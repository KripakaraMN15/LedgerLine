from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest_asyncio
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mocks.mock_llm.app import app as mock_llm_app
from mocks.mock_vendor_api.app import app as mock_vendor_app
from vendorops.api.app import create_app
from vendorops.llm.client import LLMClient
from vendorops.models import TaskStatus
from vendorops.orchestrator.lifecycle import (
    TaskEventRecord,
    TaskNotFound,
    TaskRecord,
    TaskRepository,
    validate_transition,
)
from vendorops.telemetry import configure_tracing
from vendorops.tools import VendorAPIClient


class MemoryTaskRepository:
    def __init__(self) -> None:
        self.records: dict[UUID, TaskRecord] = {}
        self.events: list[TaskEventRecord] = []
        self.available = True

    async def create_task(
        self,
        idempotency_key: str,
        goal: str,
        state: dict[str, Any] | None = None,
        trace_id: str | None = None,
        task_id: UUID | None = None,
    ) -> TaskRecord:
        existing = next(
            (row for row in self.records.values() if row.idempotency_key == idempotency_key),
            None,
        )
        if existing is not None:
            return replace(existing, was_created=False)
        now = datetime.now(UTC)
        task_id = task_id or uuid4()
        record = TaskRecord(
            id=task_id,
            idempotency_key=idempotency_key,
            goal=goal,
            status=TaskStatus.CREATED,
            state=state or {},
            trace_id=trace_id,
            created_at=now,
            updated_at=now,
            was_created=True,
        )
        self.records[task_id] = record
        self.events.append(
            TaskEventRecord(task_id, TaskStatus.CREATED, TaskStatus.CREATED, "created", now)
        )
        return record

    async def get_task(self, task_id: UUID | str) -> TaskRecord:
        parsed_id = UUID(str(task_id))
        try:
            return self.records[parsed_id]
        except KeyError as error:
            raise TaskNotFound(parsed_id) from error

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
        self.events.append(TaskEventRecord(record.id, record.status, target, reason, now))
        return updated

    async def checkpoint_task_state(
        self, task_id: UUID | str, state: dict[str, Any]
    ) -> TaskRecord:
        record = await self.get_task(task_id)
        updated = replace(record, state=state, updated_at=datetime.now(UTC))
        self.records[record.id] = updated
        return updated

    async def set_trace_id(self, task_id: UUID | str, trace_id: str) -> TaskRecord:
        record = await self.get_task(task_id)
        updated = replace(record, trace_id=record.trace_id or trace_id)
        self.records[record.id] = updated
        return updated

    async def list_task_events(self, task_id: UUID | str) -> list[TaskEventRecord]:
        parsed_id = UUID(str(task_id))
        return [event for event in self.events if event.task_id == parsed_id]

    async def find_running_tasks(self, stale_before: datetime) -> list[TaskRecord]:
        return [
            record
            for record in self.records.values()
            if record.status is TaskStatus.RUNNING and record.updated_at < stale_before
        ]

    async def find_recoverable_tasks(self, stale_before: datetime) -> list[TaskRecord]:
        return [
            record
            for record in self.records.values()
            if record.status is TaskStatus.CREATED
            or (
                record.status is TaskStatus.RUNNING
                and record.updated_at <= stale_before
            )
        ]

    async def ping(self) -> None:
        if not self.available:
            raise ConnectionError("database unavailable")

    async def count_tasks_by_idempotency_key(self, idempotency_key: str) -> int:
        return sum(
            record.idempotency_key == idempotency_key
            for record in self.records.values()
        )


@dataclass
class APIHarness:
    app: Any
    client: httpx.AsyncClient
    repository: MemoryTaskRepository
    exporter: InMemorySpanExporter
    llm_client: LLMClient
    vendor_api_client: VendorAPIClient


@pytest_asyncio.fixture
async def api_harness(monkeypatch: Any) -> AsyncIterator[APIHarness]:
    monkeypatch.setenv("ENABLE_TEST_HOOKS", "1")
    repository = MemoryTaskRepository()
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter, set_global=False)
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
    vendor = VendorAPIClient("http://mock-vendor", client=vendor_http)
    app = create_app(
        repository=cast(TaskRepository, repository),
        api_key="test-api-key",
        llm_client=llm,
        vendor_api_client=vendor,
        rate_limit_requests=10_000,
        max_budget_steps=100,
        shutdown_timeout=2.0,
    )
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
                headers={"X-API-Key": "test-api-key"},
            ) as client:
                yield APIHarness(app, client, repository, exporter, llm, vendor)
    finally:
        await llm_http.aclose()
        await vendor_http.aclose()
