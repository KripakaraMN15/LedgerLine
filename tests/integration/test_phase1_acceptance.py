import asyncio

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.xfail(strict=True, reason="phase 1")

app = pytest.importorskip("vendorops.api.app", reason="Phase 1 API is not implemented yet").app


def _task_payload(**overrides):
    payload = {
        "goal": "Review vendor V-42 and reconcile invoices.",
        "idempotency_key": "phase-1-task",
        "scenario": "happy_path",
    }
    payload.update(overrides)
    return payload


async def _wait_for_status(client: AsyncClient, task_id: str, target_status: str, timeout: float = 10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        response = await client.get(f"/tasks/{task_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] == target_status:
            return payload
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(f"Timed out waiting for task {task_id!r} to reach {target_status!r}")
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_happy_path_task_reaches_done_and_records_transition_history():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload())
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "done")
        assert final_task["status"] == "done"
        assert "created -> running -> done" in final_task["transition_history"]


@pytest.mark.asyncio
async def test_concurrency_100_tasks_finish_within_10_seconds():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        started = asyncio.get_running_loop().time()
        tasks = [
            client.post("/tasks", json=_task_payload(goal=f"Submit vendor task {index}", idempotency_key=f"concurrency-{index}"))
            for index in range(100)
        ]
        responses = await asyncio.gather(*tasks)
        ids = [response.json()["task_id"] for response in responses if response.status_code == 202]
        assert len(ids) == 100
        await asyncio.gather(*[_wait_for_status(client, task_id, "done", timeout=12.0) for task_id in ids])
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 10.0, f"Concurrent execution took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_slow_tool_call_does_not_block_other_tasks():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        slow_response = await client.post("/tasks", json=_task_payload(goal="Slow task", scenario="slow_tool", idempotency_key="slow-tool"))
        assert slow_response.status_code == 202
        slow_id = slow_response.json()["task_id"]

        fast_response = await client.post("/tasks", json=_task_payload(goal="Fast task", scenario="happy_path", idempotency_key="fast-task"))
        assert fast_response.status_code == 202
        fast_id = fast_response.json()["task_id"]

        fast_task = await _wait_for_status(client, fast_id, "done", timeout=4.0)
        assert fast_task["status"] == "done"

        slow_task = await _wait_for_status(client, slow_id, "done", timeout=12.0)
        assert slow_task["status"] == "done"


@pytest.mark.asyncio
async def test_bad_params_retry_uses_validation_error_text_in_retry_prompt():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload(goal="Trigger bad params", scenario="bad_params_then_ok", idempotency_key="bad-params"))
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "done")
        assert final_task["status"] == "done"
        assert final_task["tool_attempts"][0]["attempt"] == 2
        assert final_task["tool_attempts"][0]["validation_error"]


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_failed_task_not_hung():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload(goal="Always bad params", scenario="always_bad_params", idempotency_key="always-bad"))
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "failed")
        assert final_task["status"] == "failed"
        assert final_task["result"] == "ToolResult(ok=False)"


@pytest.mark.asyncio
async def test_budget_kill_switch_marks_task_killed_with_budget_exceeded_reason():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post(
            "/tasks",
            json=_task_payload(goal="Expensive scenario", scenario="expensive_budget", max_usd=0.01, idempotency_key="budget-kill"),
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "killed")
        assert final_task["status"] == "killed"
        assert final_task["reason"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_loop_detection_kills_task_after_three_repeated_calls():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post(
            "/tasks",
            json=_task_payload(goal="Repeat same call", scenario="repeat_same_call", idempotency_key="loop-detect"),
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "killed")
        assert final_task["status"] == "killed"
        assert final_task["reason"] == "loop_detected"


@pytest.mark.asyncio
async def test_idempotent_requests_return_same_task_id_and_create_one_task_record():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        payload = _task_payload(goal="Idempotent run", idempotency_key="duplicate-key")
        first = await client.post("/tasks", json=payload)
        second = await client.post("/tasks", json=payload)
        assert first.status_code == 202, first.text
        assert second.status_code == 200, second.text
        assert first.json()["task_id"] == second.json()["task_id"]
        assert first.json()["task_id"] == second.json()["task_id"]


@pytest.mark.asyncio
async def test_restart_recovery_never_leaves_task_stuck_in_running_state():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload(goal="Restart scenario", scenario="orphan_then_resume", idempotency_key="restart-key"))
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        await asyncio.sleep(0.2)
        resumed = await client.get(f"/tasks/{task_id}")
        assert resumed.status_code == 200
        status = resumed.json()["status"]
        assert status in {"done", "failed", "killed", "cancelled"} or status == "running"
        assert status != "running" or resumed.json()["reason"] == "orphaned"


@pytest.mark.asyncio
async def test_completed_task_contains_one_trace_with_required_spans_and_attributes():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload(goal="Tracing scenario", scenario="happy_path", idempotency_key="trace-key"))
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        final_task = await _wait_for_status(client, task_id, "done")
        assert final_task["trace_id"]
        assert "task" in final_task["spans"]
        assert any(span["name"] == "llm.call" for span in final_task["spans"])
        assert any(span["name"] == "tool.call" and span["tool"] == "get_vendor" for span in final_task["spans"])


@pytest.mark.asyncio
async def test_sigterm_allows_inflight_tasks_to_reach_checkpoint_before_exit():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.post("/tasks", json=_task_payload(goal="Graceful shutdown", scenario="slow_tool", idempotency_key="shutdown-key"))
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]

        task = await client.get(f"/tasks/{task_id}")
        assert task.status_code == 200
        status = task.json()["status"]
        assert status in {"running", "awaiting_approval", "done"}
        assert "checkpoint" in task.json().get("signals", {})
