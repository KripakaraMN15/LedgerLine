from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import httpx
import pytest

from vendorops.api.app import create_app
from vendorops.models import TaskStatus


@pytest.mark.asyncio
async def test_authentication_requires_a_valid_key(api_harness: Any) -> None:
    client = api_harness.client
    missing = await client.get("/tasks/not-a-uuid", headers={"X-API-Key": ""})
    invalid = await client.get(
        "/tasks/not-a-uuid",
        headers={"X-API-Key": "incorrect"},
    )
    health = await client.get("/healthz", headers={"X-API-Key": ""})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert health.status_code == 200
    assert missing.json() == {
        "error": {"code": "unauthorized", "message": "Invalid API key"}
    }


@pytest.mark.asyncio
async def test_missing_configured_key_fails_closed(api_harness: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("VENDOROPS_API_KEY", raising=False)
    app = create_app(repository=api_harness.repository, api_key="")
    with pytest.raises(RuntimeError, match="VENDOROPS_API_KEY must be configured"):
        async with app.router.lifespan_context(app):
            pytest.fail("Application lifespan started without an API key")


@pytest.mark.asyncio
async def test_duplicate_key_returns_existing_task_and_budget_is_capped(
    api_harness: Any,
) -> None:
    client = api_harness.client
    body = {"goal": "Check vendor invoices", "idempotency_key": "api-duplicate"}
    first = await client.post("/tasks", json=body)
    duplicate = await client.post("/tasks", json=body)
    too_expensive = await client.post(
        "/tasks",
        json={**body, "idempotency_key": "too-expensive", "max_usd": 1000},
    )
    different_goal = await client.post(
        "/tasks",
        json={"goal": "Different goal", "idempotency_key": "api-duplicate"},
    )

    assert first.status_code == 202
    assert duplicate.status_code == 200
    assert first.json()["task_id"] == duplicate.json()["task_id"]
    assert too_expensive.status_code == 202
    expensive_task = await api_harness.repository.get_task(too_expensive.json()["task_id"])
    assert expensive_task.state["budget"]["max_usd"] == 0.5
    assert different_goal.status_code == 409
    assert different_goal.json()["error"]["code"] == "idempotency_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("starting_status", "expected_status"),
    [
        (TaskStatus.CREATED, 409),
        (TaskStatus.RUNNING, 200),
        (TaskStatus.AWAITING_APPROVAL, 200),
        (TaskStatus.DONE, 409),
        (TaskStatus.FAILED, 409),
        (TaskStatus.KILLED, 409),
        (TaskStatus.CANCELLED, 409),
    ],
)
async def test_cancel_obeys_lifecycle_transitions(
    api_harness: Any,
    starting_status: TaskStatus,
    expected_status: int,
) -> None:
    repository = api_harness.repository
    task = await repository.create_task(
        idempotency_key=f"cancel-{starting_status.value}-{uuid4()}",
        goal="Cancel test task",
    )
    if starting_status is not TaskStatus.CREATED:
        task = await repository.transition_task(task.id, TaskStatus.RUNNING, "started")
    if starting_status is TaskStatus.AWAITING_APPROVAL:
        task = await repository.transition_task(
            task.id, TaskStatus.AWAITING_APPROVAL, "approval_required"
        )
    elif starting_status in {
        TaskStatus.DONE,
        TaskStatus.FAILED,
        TaskStatus.KILLED,
        TaskStatus.CANCELLED,
    }:
        task = await repository.transition_task(task.id, starting_status, "terminal")

    response = await api_harness.client.post(f"/tasks/{task.id}/cancel")
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json()["status"] == "cancelled"
    else:
        assert response.json()["error"]["code"] == "invalid_transition"


@pytest.mark.asyncio
async def test_readiness_reports_database_failure(api_harness: Any) -> None:
    api_harness.repository.available = False
    ready = await api_harness.client.get("/readyz")
    health = await api_harness.client.get("/healthz")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "not_ready"
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_oversized_request_body_is_rejected(api_harness: Any) -> None:
    payload = b'{"goal":"' + b"x" * (64 * 1024) + b'","idempotency_key":"large"}'

    async def chunks():
        yield payload[:1024]
        yield payload[1024:]

    response = await api_harness.client.post(
        "/tasks",
        content=chunks(),
        headers={
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_goal_length_and_unknown_body_fields_are_rejected(api_harness: Any) -> None:
    too_long = await api_harness.client.post(
        "/tasks",
        json={"goal": "x" * 2001, "idempotency_key": "too-long"},
    )
    extra_field = await api_harness.client.post(
        "/tasks",
        json={
            "goal": "Valid goal",
            "idempotency_key": "extra",
            "scenario": "happy_path",
        },
    )
    assert too_long.status_code == 422
    assert extra_field.status_code == 422


@pytest.mark.asyncio
async def test_task_status_response_exposes_event_history_and_trace_id(
    api_harness: Any,
) -> None:
    submitted = await api_harness.client.post(
        "/tasks",
        json={"goal": "Get status", "idempotency_key": "status"},
    )
    task_id = submitted.json()["task_id"]
    for _ in range(100):
        response = await api_harness.client.get(f"/tasks/{task_id}")
        if response.json()["status"] == "done":
            break
        await asyncio.sleep(0.01)
    data = response.json()
    assert data["trace_id"]
    assert data["budget"]["steps_taken"] >= 1
    assert data["event_history"]


@pytest.mark.asyncio
async def test_test_headers_are_ignored_when_hooks_are_disabled(
    api_harness: Any,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("ENABLE_TEST_HOOKS", raising=False)
    app = create_app(
        repository=api_harness.repository,
        api_key="test-api-key",
        llm_client=api_harness.llm_client,
        vendor_api_client=api_harness.vendor_api_client,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-api-key"},
        ) as client:
            response = await client.post(
                "/tasks",
                headers={
                    "X-Mock-Scenario": "expensive_budget",
                    "X-Mock-Latency-Ms": "30000",
                },
                json={"goal": "Hooks disabled", "idempotency_key": "hooks-disabled"},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]
            state = await api_harness.repository.get_task(task_id)
            assert state.state["llm_scenario"] is None
            assert state.state["llm_latency_ms"] is None
            deadline = asyncio.get_running_loop().time() + 2.0
            while True:
                status = await api_harness.repository.get_task(task_id)
                if status.status is TaskStatus.DONE:
                    break
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_created_task_is_recovered_at_startup_with_configurable_staleness(
    api_harness: Any,
) -> None:
    task = await api_harness.repository.create_task(
        idempotency_key="stale-created",
        goal="Recover task created before process restart",
        state={},
    )
    app = create_app(
        repository=api_harness.repository,
        api_key="test-api-key",
        llm_client=api_harness.llm_client,
        vendor_api_client=api_harness.vendor_api_client,
        recovery_stale_seconds=300,
        shutdown_timeout=0.1,
    )
    async with app.router.lifespan_context(app):
        for _ in range(300):
            recovered = await api_harness.repository.get_task(task.id)
            if recovered.status in {
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.KILLED,
            }:
                break
            await asyncio.sleep(0.01)
    assert recovered.status is TaskStatus.DONE


@pytest.mark.asyncio
async def test_unhandled_error_is_generic_and_includes_request_id(api_harness: Any) -> None:
    async def broken_endpoint() -> None:
        raise RuntimeError("sensitive exception payload")

    api_harness.app.add_api_route("/_test-error", broken_endpoint, methods=["GET"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_harness.app, raise_app_exceptions=False),
        base_url="http://testserver",
        headers={"X-API-Key": "test-api-key"},
    ) as client:
        response = await client.get("/_test-error")
    assert response.status_code == 500
    assert "sensitive exception payload" not in response.text
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_cancel_stops_active_task_and_writes_cancelled(api_harness: Any) -> None:
    response = await api_harness.client.post(
        "/tasks",
        headers={"X-Mock-Scenario": "slow_tool", "X-Mock-Latency-Ms": "5000"},
        json={"goal": "Cancel active work", "idempotency_key": "cancel-active"},
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    for _ in range(100):
        record = await api_harness.repository.get_task(task_id)
        if record.status is TaskStatus.RUNNING:
            break
        await asyncio.sleep(0.01)
    cancelled = await api_harness.client.post(f"/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    for _ in range(100):
        record = await api_harness.repository.get_task(task_id)
        active = api_harness.app.state.runner._tasks.get(task_id)
        if active is None:
            break
        await asyncio.sleep(0.01)
    assert record.status is TaskStatus.CANCELLED
    assert active is None


@pytest.mark.asyncio
async def test_cancel_racing_with_completion_leaves_a_valid_terminal_state(
    api_harness: Any,
) -> None:
    response = await api_harness.client.post(
        "/tasks",
        json={"goal": "Race cancellation against completion", "idempotency_key": "cancel-race"},
    )
    task_id = response.json()["task_id"]

    async def cancel() -> httpx.Response:
        return await api_harness.client.post(f"/tasks/{task_id}/cancel")

    async def wait_for_terminal() -> TaskStatus:
        deadline = asyncio.get_running_loop().time() + 3.0
        while True:
            current = await api_harness.repository.get_task(task_id)
            if current.status in {
                TaskStatus.DONE,
                TaskStatus.CANCELLED,
                TaskStatus.FAILED,
                TaskStatus.KILLED,
            }:
                return current.status
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)

    cancel_response, terminal_status = await asyncio.gather(cancel(), wait_for_terminal())
    assert terminal_status in {TaskStatus.DONE, TaskStatus.CANCELLED}
    assert cancel_response.status_code in {200, 409}
