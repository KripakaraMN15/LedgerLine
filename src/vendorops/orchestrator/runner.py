import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta
from types import FrameType
from uuid import UUID

from vendorops.agents.react import ReactAgent
from vendorops.llm.client import BudgetExceeded
from vendorops.llm.loop import LoopDetected
from vendorops.models import Budget, TaskState, TaskStatus
from vendorops.orchestrator.lifecycle import TaskRecord, TaskRepository

logger = logging.getLogger(__name__)


class RunnerClosed(Exception):
    pass


class TaskRunner:
    def __init__(
        self,
        repository: TaskRepository,
        agent: ReactAgent,
        *,
        concurrency_limit: int = 200,
        shutdown_timeout: float = 30.0,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> None:
        if concurrency_limit < 1:
            raise ValueError("concurrency_limit must be at least 1")
        if shutdown_timeout < 0:
            raise ValueError("shutdown_timeout must be non-negative")
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self._repository = repository
        self._agent = agent
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._shutdown_timeout = shutdown_timeout
        self._stale_after = stale_after
        self._accepting = True
        self._tasks: dict[str, asyncio.Task[TaskRecord | None]] = {}
        self._shutdown_task: asyncio.Task[None] | None = None
        self._signal_handler_installed = False

    @property
    def accepting(self) -> bool:
        return self._accepting

    def submit(self, task_id: UUID | str) -> asyncio.Task[TaskRecord | None]:
        if not self._accepting:
            raise RunnerClosed("Task runner is shutting down")
        key = str(task_id)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            return existing
        task = asyncio.create_task(self._run_one(key), name=f"vendorops-task-{key}")
        self._tasks[key] = task
        task.add_done_callback(lambda completed: self._discard_task(key, completed))
        return task

    async def run_tasks(self, task_ids: list[UUID | str]) -> list[TaskRecord | None]:
        return list(await asyncio.gather(*(self.submit(task_id) for task_id in task_ids)))

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)

    async def shutdown(self) -> None:
        self._accepting = False
        active = tuple(self._tasks.values())
        if not active:
            return
        _, pending = await asyncio.wait(active, timeout=self._shutdown_timeout)
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def recover_stale(self, stale_before: datetime | None = None) -> int:
        if stale_before is None:
            stale_before = datetime.now(UTC) - self._stale_after
        stale_tasks = await self._repository.find_running_tasks(stale_before)
        scheduled = 0
        for record in stale_tasks:
            try:
                self._state_from_record(record)
            except Exception:
                logger.exception("Could not restore stale task checkpoint")
                try:
                    await self._repository.transition_task(
                        record.id,
                        TaskStatus.FAILED,
                        "orphaned",
                        state=record.state,
                    )
                except Exception:
                    logger.exception(
                        "Could not mark stale task as orphaned",
                        extra={"task_id": str(record.id)},
                    )
                continue
            try:
                self.submit(record.id)
            except RunnerClosed:
                logger.exception("Runner closed before stale task could be resumed")
            else:
                scheduled += 1
        return scheduled

    def install_sigterm_handler(self) -> None:
        if self._signal_handler_installed:
            return
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, self._schedule_shutdown)
        except NotImplementedError:
            signal.signal(signal.SIGTERM, self._signal_callback)
        self._signal_handler_installed = True

    async def wait_for_shutdown(self) -> None:
        if self._shutdown_task is None:
            await asyncio.Future()
        else:
            await self._shutdown_task

    def _schedule_shutdown(self) -> None:
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self.shutdown())

    def _signal_callback(self, _signum: int, _frame: FrameType | None) -> None:
        asyncio.get_running_loop().call_soon_threadsafe(self._schedule_shutdown)

    def _discard_task(
        self,
        task_id: str,
        task: asyncio.Task[TaskRecord | None],
    ) -> None:
        if self._tasks.get(task_id) is task:
            self._tasks.pop(task_id, None)

    async def _run_one(self, task_id: str) -> TaskRecord | None:
        async with self._semaphore:
            state: TaskState | None = None
            try:
                record = await self._repository.get_task(task_id)
                if record.status in {
                    TaskStatus.DONE,
                    TaskStatus.FAILED,
                    TaskStatus.KILLED,
                    TaskStatus.CANCELLED,
                }:
                    return record
                state = self._state_from_record(record)
                if record.status is TaskStatus.CREATED:
                    state.status = TaskStatus.RUNNING
                    record = await self._repository.transition_task(
                        record.id,
                        TaskStatus.RUNNING,
                        "started",
                        state=state.model_dump(mode="json"),
                    )
                state.status = TaskStatus.RUNNING
                await self._agent.run(state)
                state.status = TaskStatus.DONE
                return await self._repository.transition_task(
                    record.id,
                    TaskStatus.DONE,
                    "completed",
                    state=state.model_dump(mode="json"),
                )
            except asyncio.CancelledError:
                raise
            except BudgetExceeded as error:
                return await self._finish(task_id, state, TaskStatus.KILLED, error.reason)
            except LoopDetected:
                return await self._finish(task_id, state, TaskStatus.KILLED, "loop_detected")
            except Exception as error:
                logger.exception("Task execution failed", extra={"task_id": task_id})
                return await self._finish(
                    task_id,
                    state,
                    TaskStatus.FAILED,
                    str(error) or type(error).__name__,
                )

    async def _finish(
        self,
        task_id: str,
        state: TaskState | None,
        status: TaskStatus,
        reason: str,
    ) -> TaskRecord | None:
        try:
            record = await self._repository.get_task(task_id)
            if record.status in {
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.KILLED,
                TaskStatus.CANCELLED,
            }:
                return record
            if state is None:
                state = self._state_from_record(record)
            state.status = status
            return await self._repository.transition_task(
                record.id,
                status,
                reason,
                state=state.model_dump(mode="json"),
            )
        except Exception:
            logger.exception(
                "Could not persist task terminal status",
                extra={"task_id": task_id, "status": status.value, "reason": reason},
            )
            return None

    @staticmethod
    def _state_from_record(record: TaskRecord) -> TaskState:
        if record.state:
            state = TaskState.model_validate(record.state)
            if state.task_id != str(record.id):
                raise ValueError("Checkpoint task ID does not match task record")
            state.status = record.status
            state.goal = record.goal
            return state
        return TaskState(
            task_id=str(record.id),
            status=record.status,
            goal=record.goal,
            budget=Budget(),
            updated_at=record.updated_at or datetime.now(UTC),
        )
