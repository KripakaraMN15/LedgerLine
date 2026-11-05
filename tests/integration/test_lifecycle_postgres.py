import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from mocks.mock_llm.app import app as mock_llm_app
from mocks.mock_vendor_api.app import app as mock_vendor_app
from vendorops.agents.react import ReactAgent
from vendorops.llm.client import LLMClient
from vendorops.models import TaskStatus
from vendorops.orchestrator.lifecycle import (
    TaskRepository,
    apply_migrations,
    create_task_engine,
)
from vendorops.orchestrator.runner import TaskRunner
from vendorops.tools import ToolRouter, VendorAPIClient, build_vendor_registry


@pytest_asyncio.fixture
async def repository() -> AsyncIterator[TaskRepository]:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail("REQUIRE_DB=1 but TEST_DATABASE_URL is not configured")
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL lifecycle integration tests")
    engine: AsyncEngine = create_task_engine(database_url)
    await apply_migrations(engine)
    yield TaskRepository(engine)
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_idempotent_inserts_return_one_task(
    repository: TaskRepository,
) -> None:
    idempotency_key = f"concurrent-{uuid4()}"
    records = await asyncio.gather(
        *[
            repository.create_task(
                idempotency_key=idempotency_key,
                goal=f"Concurrent request {index}",
                state={"request": index},
            )
            for index in range(24)
        ]
    )

    task_ids = {record.id for record in records}
    assert len(task_ids) == 1
    stored = await repository.get_task(records[0].id)
    assert stored.idempotency_key == idempotency_key
    assert stored.status is TaskStatus.CREATED
    assert await repository.count_tasks_by_idempotency_key(idempotency_key) == 1
    events = await repository.list_task_events(stored.id)
    assert len(events) == 1
    assert events[0].from_status is TaskStatus.CREATED
    assert events[0].to_status is TaskStatus.CREATED
    assert events[0].reason == "created"


@pytest.mark.asyncio
async def test_state_transition_and_event_are_persisted_together(
    repository: TaskRepository,
) -> None:
    task = await repository.create_task(
        idempotency_key=f"transition-{uuid4()}",
        goal="Exercise lifecycle persistence",
    )

    running = await repository.transition_task(task.id, TaskStatus.RUNNING, "started")
    done = await repository.transition_task(running.id, TaskStatus.DONE, "finished")
    events = await repository.list_task_events(task.id)

    assert done.status is TaskStatus.DONE
    assert [(event.from_status, event.to_status) for event in events] == [
        (TaskStatus.CREATED, TaskStatus.CREATED),
        (TaskStatus.CREATED, TaskStatus.RUNNING),
        (TaskStatus.RUNNING, TaskStatus.DONE),
    ]


@pytest.mark.asyncio
async def test_task_checkpoint_updates_state_without_a_status_transition(
    repository: TaskRepository,
) -> None:
    task = await repository.create_task(
        idempotency_key=f"checkpoint-{uuid4()}",
        goal="Persist an agent checkpoint",
    )
    running = await repository.transition_task(task.id, TaskStatus.RUNNING, "started")
    checkpoint = await repository.checkpoint_task_state(
        task.id,
        {"messages": [{"role": "user", "content": "resume"}], "iteration": 1},
    )
    events = await repository.list_task_events(task.id)

    assert checkpoint.status is TaskStatus.RUNNING
    assert checkpoint.state["iteration"] == 1
    assert checkpoint.updated_at >= running.updated_at
    assert len(events) == 2


@pytest.mark.asyncio
async def test_task_trace_id_is_persisted(repository: TaskRepository) -> None:
    task = await repository.create_task(
        idempotency_key=f"trace-id-{uuid4()}",
        goal="Persist trace context",
    )

    updated = await repository.set_trace_id(task.id, "0a" * 16)

    assert updated.trace_id == "0a" * 16
    assert (await repository.get_task(task.id)).trace_id == "0a" * 16


@pytest.mark.asyncio
async def test_running_task_recovery_query_filters_by_staleness(
    repository: TaskRepository,
) -> None:
    task = await repository.create_task(
        idempotency_key=f"stale-{uuid4()}",
        goal="Find running task during recovery",
    )
    await repository.transition_task(task.id, TaskStatus.RUNNING, "started")

    stale_tasks = await repository.find_running_tasks(
        datetime.now(UTC) + timedelta(seconds=1)
    )

    assert task.id in {stale_task.id for stale_task in stale_tasks}


@pytest.mark.asyncio
async def test_one_hundred_tasks_complete_under_ten_seconds_with_postgres(
    repository: TaskRepository,
) -> None:
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
    headers = {"X-Scenario": "happy_path", "X-Latency-Ms": "200"}
    llm = LLMClient("http://mock-llm", client=llm_http)
    router = ToolRouter(
        build_vendor_registry(VendorAPIClient("http://mock-vendor", client=vendor_http)),
        llm,
    )
    agent = ReactAgent(llm, router, repository, headers=headers)
    runner = TaskRunner(repository, agent, concurrency_limit=200)
    tasks = await asyncio.gather(
        *[
            repository.create_task(
                idempotency_key=f"postgres-concurrency-{uuid4()}",
                goal=f"Concurrent PostgreSQL task {index}",
            )
            for index in range(100)
        ]
    )

    try:
        started = asyncio.get_running_loop().time()
        results = await runner.run_tasks([task.id for task in tasks])
        elapsed = asyncio.get_running_loop().time() - started
    finally:
        await runner.shutdown()
        await llm_http.aclose()
        await vendor_http.aclose()

    assert elapsed < 10
    assert all(result is not None and result.status is TaskStatus.DONE for result in results)
