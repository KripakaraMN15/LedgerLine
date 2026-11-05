import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine

from vendorops.models import TaskStatus

logger = logging.getLogger(__name__)

LEGAL_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.RUNNING}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.DONE,
            TaskStatus.FAILED,
            TaskStatus.KILLED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.AWAITING_APPROVAL: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.FAILED,
            TaskStatus.KILLED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.DONE: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.KILLED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransition(Exception):
    def __init__(self, source: TaskStatus, target: TaskStatus) -> None:
        self.source = source
        self.target = target
        super().__init__(f"Task transition {source.value} -> {target.value} is not allowed")


class TaskNotFound(Exception):
    def __init__(self, task_id: UUID) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id} was not found")


@dataclass(frozen=True)
class TaskRecord:
    id: UUID
    idempotency_key: str
    goal: str
    status: TaskStatus
    state: dict[str, Any]
    trace_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TaskEventRecord:
    task_id: UUID
    from_status: TaskStatus
    to_status: TaskStatus
    reason: str
    created_at: datetime


def validate_transition(source: TaskStatus, target: TaskStatus) -> None:
    if target not in LEGAL_TRANSITIONS[source]:
        logger.warning(
            "Rejected illegal task transition",
            extra={"from_status": source.value, "to_status": target.value},
        )
        raise InvalidTransition(source, target)


def _task_from_row(row: RowMapping) -> TaskRecord:
    state = row["state"]
    if not isinstance(state, dict):
        raise TypeError("Persisted task state must be a JSON object")
    return TaskRecord(
        id=UUID(str(row["id"])),
        idempotency_key=str(row["idempotency_key"]),
        goal=str(row["goal"]),
        status=TaskStatus(str(row["status"])),
        state=state,
        trace_id=row["trace_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _event_from_row(row: RowMapping) -> TaskEventRecord:
    return TaskEventRecord(
        task_id=UUID(str(row["task_id"])),
        from_status=TaskStatus(str(row["from_status"])),
        to_status=TaskStatus(str(row["to_status"])),
        reason=str(row["reason"]),
        created_at=row["created_at"],
    )


class TaskRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_task(
        self,
        idempotency_key: str,
        goal: str,
        state: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> TaskRecord:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if not goal.strip():
            raise ValueError("goal must not be empty")

        task_id = uuid4()
        initial_state = state if state is not None else {}
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    """
                    INSERT INTO tasks (
                        id, idempotency_key, goal, status, state, trace_id
                    )
                    VALUES (
                        :id, :idempotency_key, :goal, :status,
                        CAST(:state AS jsonb), :trace_id
                    )
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING id, idempotency_key, goal, status, state,
                              trace_id, created_at, updated_at
                    """
                ),
                {
                    "id": task_id,
                    "idempotency_key": idempotency_key,
                    "goal": goal,
                    "status": TaskStatus.CREATED.value,
                    "state": json.dumps(initial_state),
                    "trace_id": trace_id,
                },
            )
            row = result.mappings().one_or_none()
            was_created = row is not None
            if row is None:
                result = await connection.execute(
                    text(
                        """
                        SELECT id, idempotency_key, goal, status, state,
                               trace_id, created_at, updated_at
                        FROM tasks
                        WHERE idempotency_key = :idempotency_key
                        """
                    ),
                    {"idempotency_key": idempotency_key},
                )
                row = result.mappings().one_or_none()
            if row is None:
                raise RuntimeError("Idempotent task insert produced no task row")
            if was_created:
                await connection.execute(
                    text(
                        """
                        INSERT INTO task_events (
                            task_id, from_status, to_status, reason
                        )
                        VALUES (
                            :task_id, :status, :status, 'created'
                        )
                        """
                    ),
                    {"task_id": row["id"], "status": TaskStatus.CREATED.value},
                )
            return _task_from_row(row)

    async def get_task(self, task_id: UUID | str) -> TaskRecord:
        parsed_id = UUID(str(task_id))
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT id, idempotency_key, goal, status, state,
                           trace_id, created_at, updated_at
                    FROM tasks
                    WHERE id = :id
                    """
                ),
                {"id": parsed_id},
            )
            row = result.mappings().one_or_none()
        if row is None:
            raise TaskNotFound(parsed_id)
        return _task_from_row(row)

    async def transition_task(
        self,
        task_id: UUID | str,
        target: TaskStatus,
        reason: str,
        state: dict[str, Any] | None = None,
    ) -> TaskRecord:
        parsed_id = UUID(str(task_id))
        if not reason.strip():
            raise ValueError("reason must not be empty")

        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT id, idempotency_key, goal, status, state,
                           trace_id, created_at, updated_at
                    FROM tasks
                    WHERE id = :id
                    FOR UPDATE
                    """
                ),
                {"id": parsed_id},
            )
            current_row = result.mappings().one_or_none()
            if current_row is None:
                raise TaskNotFound(parsed_id)

            current = _task_from_row(current_row)
            validate_transition(current.status, target)

            state_update = ""
            parameters: dict[str, Any] = {
                "id": parsed_id,
                "status": target.value,
            }
            if state is not None:
                state_update = ", state = CAST(:state AS jsonb)"
                parameters["state"] = json.dumps(state)

            result = await connection.execute(
                text(
                    f"""
                    UPDATE tasks
                    SET status = :status, updated_at = now(){state_update}
                    WHERE id = :id
                    RETURNING id, idempotency_key, goal, status, state,
                              trace_id, created_at, updated_at
                    """
                ),
                parameters,
            )
            updated_row = result.mappings().one()
            await connection.execute(
                text(
                    """
                    INSERT INTO task_events (
                        task_id, from_status, to_status, reason
                    )
                    VALUES (:task_id, :from_status, :to_status, :reason)
                    """
                ),
                {
                    "task_id": parsed_id,
                    "from_status": current.status.value,
                    "to_status": target.value,
                    "reason": reason,
                },
            )
            return _task_from_row(updated_row)

    async def find_running_tasks(self, stale_before: datetime) -> list[TaskRecord]:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT id, idempotency_key, goal, status, state,
                           trace_id, created_at, updated_at
                    FROM tasks
                    WHERE status = :status AND updated_at <= :stale_before
                    ORDER BY updated_at, id
                    """
                ),
                {
                    "status": TaskStatus.RUNNING.value,
                    "stale_before": stale_before,
                },
            )
            rows = result.mappings().all()
        return [_task_from_row(row) for row in rows]

    async def list_task_events(self, task_id: UUID | str) -> list[TaskEventRecord]:
        parsed_id = UUID(str(task_id))
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT task_id, from_status, to_status, reason, created_at
                    FROM task_events
                    WHERE task_id = :task_id
                    ORDER BY created_at, id
                    """
                ),
                {"task_id": parsed_id},
            )
            rows = result.mappings().all()
        return [_event_from_row(row) for row in rows]

    async def count_tasks_by_idempotency_key(self, idempotency_key: str) -> int:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text("SELECT count(*) FROM tasks WHERE idempotency_key = :idempotency_key"),
                {"idempotency_key": idempotency_key},
            )
            count = result.scalar_one()
        return int(count)


async def apply_migrations(engine: AsyncEngine, directory: Path | None = None) -> None:
    migration_directory = (
        directory if directory is not None else Path(__file__).resolve().parents[3] / "migrations"
    )
    migration_files = sorted(migration_directory.glob("*.sql"))
    if not migration_files:
        raise FileNotFoundError(f"No SQL migrations found in {migration_directory}")
    async with engine.begin() as connection:
        for migration_file in migration_files:
            content = migration_file.read_text(encoding="utf-8")
            statements = [statement.strip() for statement in content.split(";")]
            for statement in statements:
                if statement:
                    await connection.exec_driver_sql(statement)
