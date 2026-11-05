import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel

from mocks.mock_llm.app import app as mock_llm_app
from mocks.mock_vendor_api.app import app as mock_vendor_app
from vendorops.agents.react import ReactAgent
from vendorops.llm.client import LLMClient
from vendorops.models import (
    Budget,
    TaskState,
    TaskStatus,
    ToolCall,
    ToolResult,
)
from vendorops.orchestrator.lifecycle import (
    TaskEventRecord,
    TaskNotFound,
    TaskRecord,
    create_task_engine,
    validate_transition,
)
from vendorops.orchestrator.runner import RunnerClosed, TaskRunner
from vendorops.tools import (
    ToolDefinition,
    ToolRegistry,
    ToolRouter,
    VendorAPIClient,
    build_vendor_registry,
)


class MemoryRepository:
    def __init__(self) -> None:
        self.records: dict[UUID, TaskRecord] = {}
        self.events: list[TaskEventRecord] = []

    def seed(
        self,
        *,
        goal: str,
        status: TaskStatus = TaskStatus.CREATED,
        state: dict[str, Any] | None = None,
        updated_at: datetime | None = None,
    ) -> TaskRecord:
        task_id = uuid4()
        now = updated_at or datetime.now(UTC)
        record = TaskRecord(
            id=task_id,
            idempotency_key=str(task_id),
            goal=goal,
            status=status,
            state=state or {},
            trace_id=None,
            created_at=now,
            updated_at=now,
        )
        self.records[task_id] = record
        return record

    async def get_task(self, task_id: UUID | str) -> TaskRecord:
        parsed_id = UUID(str(task_id))
        try:
            return self.records[parsed_id]
        except KeyError as error:
            raise TaskNotFound(parsed_id) from error

    async def checkpoint_task_state(
        self,
        task_id: UUID | str,
        state: dict[str, Any],
    ) -> TaskRecord:
        record = await self.get_task(task_id)
        updated = replace(record, state=state, updated_at=datetime.now(UTC))
        self.records[record.id] = updated
        return updated

    async def transition_task(
        self,
        task_id: UUID | str,
        target: TaskStatus,
        reason: str,
        state: dict[str, Any] | None = None,
    ) -> TaskRecord:
        record = await self.get_task(task_id)
        validate_transition(record.status, target)
        now = datetime.now(UTC)
        updated = replace(
            record,
            status=target,
            state=state if state is not None else record.state,
            updated_at=now,
        )
        self.records[record.id] = updated
        self.events.append(
            TaskEventRecord(record.id, record.status, target, reason, now)
        )
        return updated

    async def find_running_tasks(self, stale_before: datetime) -> list[TaskRecord]:
        return [
            record
            for record in self.records.values()
            if record.status is TaskStatus.RUNNING and record.updated_at <= stale_before
        ]


class Input(BaseModel):
    vendor_id: str


class Output(BaseModel):
    vendor_id: str
    status: str


class GoalTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, slow_goal: str | None = None) -> None:
        self.slow_goal = slow_goal
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.calls.append(payload)
        messages = payload["messages"]
        if any(message.get("role") == "tool" for message in messages):
            message = {"role": "assistant", "content": "Task completed."}
            finish_reason = "stop"
        else:
            goal = next(
                (message["content"] for message in messages if message.get("role") == "user"),
                "",
            )
            vendor_id = (
                "slow"
                if self.slow_goal is not None and self.slow_goal in goal
                else "fail"
                if "fail" in goal.lower()
                else "fast"
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{len(self.calls)}",
                        "type": "function",
                        "function": {
                            "name": "get_vendor",
                            "arguments": json.dumps({"vendor_id": vendor_id}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async def aclose(self) -> None:
        return None


@pytest_asyncio.fixture
async def mock_runner() -> AsyncIterator[tuple[MemoryRepository, TaskRunner, httpx.AsyncClient]]:
    repository = MemoryRepository()
    llm_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app),
        base_url="http://mock-llm",
    )
    vendor_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_vendor_app),
        base_url="http://mock-vendor",
    )
    await vendor_http.post(
        "/admin/faults",
        json={"FAIL_404": False, "FAIL_TIMEOUT": False, "RETURN_INJECTION": False},
    )
    llm = LLMClient("http://mock-llm", client=llm_http)
    router = ToolRouter(
        build_vendor_registry(VendorAPIClient("http://mock-vendor", client=vendor_http)),
        llm,
    )
    agent = ReactAgent(llm, router, repository)
    runner = TaskRunner(repository, agent)
    try:
        yield repository, runner, llm_http
    finally:
        await runner.shutdown()
        await llm_http.aclose()
        await vendor_http.aclose()


def _task_state(record: TaskRecord, *, budget: Budget | None = None) -> TaskState:
    return TaskState(
        task_id=str(record.id),
        status=record.status,
        goal=record.goal,
        budget=budget or Budget(),
        updated_at=record.updated_at,
    )


@pytest.mark.asyncio
async def test_one_hundred_tasks_complete_concurrently_under_ten_seconds() -> None:
    repository = MemoryRepository()
    llm_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app),
        base_url="http://mock-llm",
        headers={"X-Scenario": "happy_path", "X-Latency-Ms": "200"},
    )
    vendor_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_vendor_app),
        base_url="http://mock-vendor",
    )
    await vendor_http.post(
        "/admin/faults",
        json={"FAIL_404": False, "FAIL_TIMEOUT": False, "RETURN_INJECTION": False},
    )
    llm = LLMClient("http://mock-llm", client=llm_http)
    agent = ReactAgent(
        llm,
        ToolRouter(
            build_vendor_registry(VendorAPIClient("http://mock-vendor", client=vendor_http)),
            llm,
        ),
        repository,
        headers={"X-Scenario": "happy_path", "X-Latency-Ms": "200"},
    )
    runner = TaskRunner(repository, agent, concurrency_limit=200)
    records = [repository.seed(goal=f"Concurrent task {index}") for index in range(100)]

    started = asyncio.get_running_loop().time()
    results = await asyncio.gather(*(runner.submit(record.id) for record in records))
    elapsed = asyncio.get_running_loop().time() - started
    await runner.shutdown()
    await llm_http.aclose()
    await vendor_http.aclose()

    assert elapsed < 10
    assert all(result is not None and result.status is TaskStatus.DONE for result in results)


@pytest.mark.asyncio
async def test_slow_tool_handler_does_not_delay_other_tasks() -> None:
    repository = MemoryRepository()
    transport = GoalTransport(slow_goal="slow")
    llm_http = httpx.AsyncClient(transport=transport, base_url="http://mock-llm")
    entered_slow = asyncio.Event()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        if args["vendor_id"] == "slow":
            entered_slow.set()
            await asyncio.sleep(5)
        return {"vendor_id": args["vendor_id"], "status": "active"}

    registry = ToolRegistry()
    registry.register(ToolDefinition("get_vendor", "Get vendor.", Input, Output, handler))
    llm = LLMClient("http://mock-llm", client=llm_http)
    agent = ReactAgent(llm, ToolRouter(registry, llm), repository)
    runner = TaskRunner(repository, agent, concurrency_limit=2, shutdown_timeout=0)
    slow = repository.seed(goal="slow task")
    fast = repository.seed(goal="fast task")

    slow_task = runner.submit(slow.id)
    fast_task = runner.submit(fast.id)
    await asyncio.wait_for(entered_slow.wait(), timeout=1)
    fast_result = await asyncio.wait_for(fast_task, timeout=1)

    assert fast_result is not None and fast_result.status is TaskStatus.DONE
    assert not slow_task.done()
    await runner.shutdown()
    await llm_http.aclose()
    assert (await repository.get_task(slow.id)).status is TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_budget_exceeded_maps_to_killed_with_reason(
    mock_runner: tuple[MemoryRepository, TaskRunner, httpx.AsyncClient],
) -> None:
    repository, runner, client = mock_runner
    runner._agent._headers["X-Scenario"] = "expensive_budget"
    record = repository.seed(goal="expensive task")
    state = _task_state(record, budget=Budget(max_usd=0.01))
    await repository.checkpoint_task_state(record.id, state.model_dump(mode="json"))

    result = await runner.submit(record.id)
    await client.aclose()

    assert result is not None and result.status is TaskStatus.KILLED
    assert any(event.reason == "budget_exceeded" for event in repository.events)


@pytest.mark.asyncio
async def test_loop_detection_maps_to_killed_with_reason(
    mock_runner: tuple[MemoryRepository, TaskRunner, httpx.AsyncClient],
) -> None:
    repository, runner, client = mock_runner
    runner._agent._headers["X-Scenario"] = "repeat_same_call"
    record = repository.seed(goal="repeat call")

    result = await runner.submit(record.id)
    await client.aclose()

    assert result is not None and result.status is TaskStatus.KILLED
    assert any(event.reason == "loop_detected" for event in repository.events)


@pytest.mark.asyncio
async def test_handler_exception_fails_only_that_task() -> None:
    repository = MemoryRepository()
    llm_http = httpx.AsyncClient(transport=GoalTransport(), base_url="http://mock-llm")

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        if args["vendor_id"] == "fail":
            raise RuntimeError("handler failed")
        return {"vendor_id": args["vendor_id"], "status": "active"}

    registry = ToolRegistry()
    registry.register(ToolDefinition("get_vendor", "Get vendor.", Input, Output, handler))
    llm = LLMClient("http://mock-llm", client=llm_http)
    agent = ReactAgent(llm, ToolRouter(registry, llm), repository)
    runner = TaskRunner(repository, agent, concurrency_limit=2)
    failed = repository.seed(goal="fail this task")
    succeeded = repository.seed(goal="succeed this task")

    results = await asyncio.gather(runner.submit(failed.id), runner.submit(succeeded.id))
    await llm_http.aclose()

    assert results[0] is not None and results[0].status is TaskStatus.FAILED
    assert results[1] is not None and results[1].status is TaskStatus.DONE


@pytest.mark.asyncio
async def test_shutdown_cancellation_keeps_pending_call_for_resume() -> None:
    repository = MemoryRepository()
    llm_http = httpx.AsyncClient(transport=GoalTransport(), base_url="http://mock-llm")
    entered = asyncio.Event()
    calls = 0

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.sleep(5)
        return {"vendor_id": args["vendor_id"], "status": "active"}

    registry = ToolRegistry()
    registry.register(ToolDefinition("get_vendor", "Get vendor.", Input, Output, handler))
    llm = LLMClient("http://mock-llm", client=llm_http)
    agent = ReactAgent(llm, ToolRouter(registry, llm), repository)
    runner = TaskRunner(repository, agent, shutdown_timeout=0.01)
    record = repository.seed(goal="resume me")

    runner.submit(record.id)
    await asyncio.wait_for(entered.wait(), timeout=1)
    await runner.shutdown()
    with pytest.raises(RunnerClosed):
        runner.submit(record.id)

    checkpoint = await repository.get_task(record.id)
    assert checkpoint.status is TaskStatus.RUNNING
    assert checkpoint.state["pending_tool_call"]["id"] == "call-1"
    assert len(checkpoint.state["history"]) == 1

    resumed_llm = LLMClient("http://mock-llm", client=llm_http)
    resumed_agent = ReactAgent(resumed_llm, ToolRouter(registry, resumed_llm), repository)
    resumed_runner = TaskRunner(repository, resumed_agent)
    resumed = await resumed_runner.submit(record.id)

    assert resumed is not None and resumed.status is TaskStatus.DONE
    assert calls == 2
    await llm_http.aclose()


@pytest.mark.asyncio
async def test_recovery_resumes_stale_running_task_without_replaying_completed_tool() -> None:
    repository = MemoryRepository()
    record = repository.seed(
        goal="continue after completed tool",
        status=TaskStatus.RUNNING,
        updated_at=datetime.now(UTC) - timedelta(hours=1),
    )
    completed_call = ToolCall(
        id="previous-call",
        tool="get_vendor",
        args={"vendor_id": "fast"},
    )
    state = TaskState(
        task_id=str(record.id),
        status=TaskStatus.RUNNING,
        goal=record.goal,
        budget=Budget(),
        history=[
            completed_call,
            ToolResult(
                call_id="previous-call",
                ok=True,
                output={"vendor_id": "fast", "status": "active"},
                latency_ms=1,
            ),
        ],
        messages=[
            {"role": "user", "content": record.goal},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "previous-call",
                        "type": "function",
                        "function": {
                            "name": "get_vendor",
                            "arguments": '{"vendor_id":"fast"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "previous-call",
                "name": "get_vendor",
                "content": '{"ok":true,"output":{"vendor_id":"fast","status":"active"}}',
            },
        ],
        updated_at=record.updated_at,
    )
    await repository.checkpoint_task_state(record.id, state.model_dump(mode="json"))
    llm_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app),
        base_url="http://mock-llm",
    )
    calls = 0

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"vendor_id": args["vendor_id"], "status": "active"}

    registry = ToolRegistry()
    registry.register(ToolDefinition("get_vendor", "Get vendor.", Input, Output, handler))
    llm = LLMClient("http://mock-llm", client=llm_http)
    agent = ReactAgent(llm, ToolRouter(registry, llm), repository)
    runner = TaskRunner(repository, agent)

    scheduled = await runner.recover_stale(datetime.now(UTC))
    await runner.wait_idle()
    restored = await repository.get_task(record.id)
    await llm_http.aclose()

    assert scheduled == 1
    assert restored.status is TaskStatus.DONE
    assert calls == 0
    assert len(restored.state["history"]) == 2


@pytest.mark.asyncio
async def test_recovery_marks_unreadable_stale_checkpoint_failed_as_orphaned() -> None:
    repository = MemoryRepository()
    record = repository.seed(
        goal="bad checkpoint",
        status=TaskStatus.RUNNING,
        state={"invalid": "checkpoint"},
        updated_at=datetime.now(UTC) - timedelta(hours=1),
    )
    llm_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app),
        base_url="http://mock-llm",
    )
    llm = LLMClient("http://mock-llm", client=llm_http)
    router = ToolRouter(ToolRegistry(), llm)
    runner = TaskRunner(
        repository,
        ReactAgent(llm, router, repository),
    )

    scheduled = await runner.recover_stale(datetime.now(UTC))
    restored = await repository.get_task(record.id)
    await llm_http.aclose()

    assert scheduled == 0
    assert restored.status is TaskStatus.FAILED
    assert any(event.reason == "orphaned" for event in repository.events)


@pytest.mark.asyncio
async def test_task_engine_defaults_to_a_deliberately_sized_connection_pool() -> None:
    engine = create_task_engine("postgresql+asyncpg://vendorops:vendorops@localhost/vendorops")

    assert engine.pool.size() == 10
    await engine.dispose()


@pytest.mark.asyncio
async def test_recovery_uses_a_five_minute_staleness_threshold() -> None:
    repository = MemoryRepository()
    now = datetime.now(UTC)
    stale = repository.seed(
        goal="stale task",
        status=TaskStatus.RUNNING,
        updated_at=now - timedelta(minutes=6),
    )
    stale_state = TaskState(
        task_id=str(stale.id),
        status=TaskStatus.RUNNING,
        goal=stale.goal,
        budget=Budget(),
        final_answer="Already completed before the checkpoint.",
        updated_at=stale.updated_at,
    ).model_dump(mode="json")
    repository.records[stale.id] = replace(stale, state=stale_state)
    fresh = repository.seed(
        goal="healthy task",
        status=TaskStatus.RUNNING,
        updated_at=now - timedelta(minutes=1),
    )
    fresh_state = TaskState(
        task_id=str(fresh.id),
        status=TaskStatus.RUNNING,
        goal=fresh.goal,
        budget=Budget(),
        final_answer="Still active.",
        updated_at=fresh.updated_at,
    ).model_dump(mode="json")
    repository.records[fresh.id] = replace(fresh, state=fresh_state)
    llm_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app),
        base_url="http://mock-llm",
    )
    llm = LLMClient("http://mock-llm", client=llm_http)
    runner = TaskRunner(
        repository,
        ReactAgent(llm, ToolRouter(ToolRegistry(), llm), repository),
    )

    scheduled = await runner.recover_stale()
    await runner.wait_idle()
    await llm_http.aclose()

    assert scheduled == 1
    assert (await repository.get_task(stale.id)).status is TaskStatus.DONE
    assert (await repository.get_task(fresh.id)).status is TaskStatus.RUNNING
