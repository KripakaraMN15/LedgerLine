import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from vendorops.models import TaskStatus
from vendorops.orchestrator.lifecycle import TaskRepository, apply_migrations


@pytest_asyncio.fixture
async def repository() -> AsyncIterator[TaskRepository]:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail("REQUIRE_DB=1 but TEST_DATABASE_URL is not configured")
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL lifecycle integration tests")
    engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
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
