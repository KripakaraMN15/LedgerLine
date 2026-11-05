import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from opentelemetry.trace import StatusCode

from vendorops.api.app import create_app
from vendorops.models import TaskStatus


def _task_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "goal": "Review vendor V-42 and reconcile invoices.",
        "idempotency_key": "phase-1-task",
        "scenario": "happy_path",
    }
    payload.update(overrides)
    return payload


async def _submit(
    client: httpx.AsyncClient,
    payload: dict[str, object],
    *,
    latency_ms: int | None = None,
) -> httpx.Response:
    data = dict(payload)
    scenario = str(data.pop("scenario", "happy_path"))
    headers = {"X-Mock-Scenario": scenario}
    if latency_ms is not None:
        headers["X-Mock-Latency-Ms"] = str(latency_ms)
    return await client.post("/tasks", json=data, headers=headers)


async def _wait_for_status(
    client: httpx.AsyncClient,
    task_id: str,
    target_status: str,
    timeout: float = 10.0,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        response = await client.get(f"/tasks/{task_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] == target_status:
            return payload
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(f"Timed out waiting for task {task_id!r} to reach {target_status!r}")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_happy_path_task_reaches_done_and_records_transition_history(
    api_harness: Any,
) -> None:
    client = api_harness.client
    response = await _submit(client, _task_payload())
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(client, response.json()["task_id"], "done")
    assert final_task["status"] == "done"
    assert "created -> running -> done" in final_task["transition_history"]


@pytest.mark.asyncio
async def test_concurrency_100_tasks_finish_within_10_seconds(
    api_harness: Any,
) -> None:
    client = api_harness.client
    started = asyncio.get_running_loop().time()
    responses = await asyncio.gather(
        *[
            _submit(
                client,
                _task_payload(
                    goal=f"Submit vendor task {index}",
                    idempotency_key=f"concurrency-{index}",
                ),
                latency_ms=200,
            )
            for index in range(100)
        ]
    )
    ids = [response.json()["task_id"] for response in responses if response.status_code == 202]
    assert len(ids) == 100
    await asyncio.gather(
        *[_wait_for_status(client, task_id, "done", timeout=12.0) for task_id in ids]
    )
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 10.0, f"Concurrent execution took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_slow_tool_call_does_not_block_other_tasks(api_harness: Any) -> None:
    client = api_harness.client
    slow_response = await _submit(
        client,
        _task_payload(goal="Slow task", scenario="slow_tool", idempotency_key="slow-tool"),
    )
    assert slow_response.status_code == 202
    fast_response = await _submit(
        client,
        _task_payload(goal="Fast task", idempotency_key="fast-task"),
    )
    assert fast_response.status_code == 202

    fast_task = await _wait_for_status(client, fast_response.json()["task_id"], "done", 4.0)
    assert fast_task["status"] == "done"
    slow_task = await _wait_for_status(
        client, slow_response.json()["task_id"], "done", timeout=12.0
    )
    assert slow_task["status"] == "done"


@pytest.mark.asyncio
async def test_bad_params_retry_uses_validation_error_text_in_retry_prompt(
    api_harness: Any,
) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(
            goal="Trigger bad params",
            scenario="bad_params_then_ok",
            idempotency_key="bad-params",
        ),
    )
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(api_harness.client, response.json()["task_id"], "done")
    assert final_task["status"] == "done"
    assert final_task["tool_attempts"][0]["attempt"] == 2
    assert final_task["tool_attempts"][0]["validation_error"]


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_failed_task_not_hung(api_harness: Any) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(
            goal="Always bad params",
            scenario="always_bad_params",
            idempotency_key="always-bad",
        ),
    )
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(
        api_harness.client, response.json()["task_id"], "failed"
    )
    assert final_task["status"] == "failed"
    assert final_task["result"] == "ToolResult(ok=False)"


@pytest.mark.asyncio
async def test_budget_kill_switch_marks_task_killed_with_budget_exceeded_reason(
    api_harness: Any,
) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(
            goal="Expensive scenario",
            scenario="expensive_budget",
            max_usd=0.01,
            idempotency_key="budget-kill",
        ),
    )
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(api_harness.client, response.json()["task_id"], "killed")
    assert final_task["status"] == "killed"
    assert final_task["reason"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_loop_detection_kills_task_after_three_repeated_calls(
    api_harness: Any,
) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(
            goal="Repeat same call",
            scenario="repeat_same_call",
            idempotency_key="loop-detect",
        ),
    )
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(api_harness.client, response.json()["task_id"], "killed")
    assert final_task["status"] == "killed"
    assert final_task["reason"] == "loop_detected"


@pytest.mark.asyncio
async def test_idempotent_requests_return_same_task_id_and_create_one_task_record(
    api_harness: Any,
) -> None:
    client = api_harness.client
    payload = _task_payload(goal="Idempotent run", idempotency_key="duplicate-key")
    first = await _submit(client, payload)
    second = await _submit(client, payload)
    assert first.status_code == 202, first.text
    assert second.status_code == 200, second.text
    assert first.json()["task_id"] == second.json()["task_id"]
    assert await api_harness.repository.count_tasks_by_idempotency_key("duplicate-key") == 1


@pytest.mark.asyncio
async def test_restart_recovery_never_leaves_task_stuck_in_running_state(
    api_harness: Any,
) -> None:
    task = await api_harness.repository.create_task(
        idempotency_key="restart-key",
        goal="Recover a running task abandoned by the previous process",
    )
    task = await api_harness.repository.transition_task(
        task.id,
        TaskStatus.RUNNING,
        "started",
    )
    api_harness.repository.records[task.id] = replace(
        task,
        updated_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    restarted_app = create_app(
        repository=api_harness.repository,
        api_key="test-api-key",
        llm_client=api_harness.llm_client,
        vendor_api_client=api_harness.vendor_api_client,
        recovery_stale_seconds=1,
        shutdown_timeout=0.1,
    )
    async with restarted_app.router.lifespan_context(restarted_app):
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            recovered = await api_harness.repository.get_task(task.id)
            if recovered.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.KILLED}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("Startup recovery left the task in a non-terminal state")
            await asyncio.sleep(0.02)
    assert recovered.status is TaskStatus.DONE


@pytest.mark.asyncio
async def test_completed_task_contains_one_trace_with_required_spans_and_attributes(
    api_harness: Any,
) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(goal="Tracing scenario", idempotency_key="trace-key"),
    )
    assert response.status_code == 202, response.text
    final_task = await _wait_for_status(api_harness.client, response.json()["task_id"], "done")
    assert final_task["trace_id"]
    trace_id = final_task["trace_id"]
    spans = [
        span
        for span in api_harness.exporter.get_finished_spans()
        if f"{span.context.trace_id:032x}" == trace_id
    ]
    task_spans = [span for span in spans if span.name == "task"]
    assert len(task_spans) == 1
    assert any(span.name == "llm.call" for span in spans)
    assert any(
        span.name == "tool.call" and span.attributes.get("tool") == "get_vendor"
        for span in spans
    )
    assert all(span.status.status_code is not StatusCode.ERROR for span in task_spans)


@pytest.mark.asyncio
async def test_sigterm_allows_inflight_tasks_to_reach_checkpoint_before_exit(api_harness: Any) -> None:
    response = await _submit(
        api_harness.client,
        _task_payload(goal="Graceful shutdown", scenario="slow_tool", idempotency_key="shutdown-key"),
        latency_ms=250,
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    task = await api_harness.client.get(f"/tasks/{task_id}")
    deadline = asyncio.get_running_loop().time() + 2.0
    while task.json()["status"] == "created":
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("Task did not start before the checkpoint deadline")
        await asyncio.sleep(0.01)
        task = await api_harness.client.get(f"/tasks/{task_id}")
    assert task.status_code == 200
    assert task.json()["status"] in {"running", "awaiting_approval", "done"}
    assert task.json()["signals"]["checkpoint"]
    await api_harness.app.state.runner.shutdown()
    persisted = await api_harness.client.get(f"/tasks/{task_id}")
    assert persisted.json()["status"] == "done"
    assert persisted.json()["signals"]["checkpoint"]
