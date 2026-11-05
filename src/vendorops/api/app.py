from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vendorops.agents.react import ReactAgent
from vendorops.llm.client import LLMClient
from vendorops.models import Budget, TaskState, TaskStatus
from vendorops.orchestrator.lifecycle import (
    InvalidTransition,
    TaskNotFound,
    TaskRepository,
    apply_migrations,
    create_task_engine,
)
from vendorops.orchestrator.runner import RunnerClosed, TaskRunner
from vendorops.tools import ToolRouter, VendorAPIClient, build_vendor_registry

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class TaskSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=2000)
    idempotency_key: str = Field(min_length=1, max_length=200)
    max_usd: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, ge=1)

    @field_validator("goal", "idempotency_key")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("max_usd")
    @classmethod
    def reject_non_finite_budget(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("max_usd must be finite")
        return value


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self._app(scope, receive, send)
            return

        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                async def replay_disconnected(
                    disconnect: Message = message,
                ) -> Message:
                    if messages:
                        return messages.pop(0)
                    return disconnect

                await self._app(scope, replay_disconnected, send)
                return
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self._max_body_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "request_too_large",
                                "message": "Request body exceeds the allowed size",
                            }
                        },
                    )
                    await response(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break

        async def replay() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self._app(scope, replay, send)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        request_id = str(uuid4())
        state["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self._app(scope, receive, send_with_request_id)


class APIKeyAndRateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        api_key: str | None,
        request_limit: int,
        window_seconds: float,
    ) -> None:
        self._app = app
        self._api_key = api_key
        self._request_limit = request_limit
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in {"/healthz", "/readyz"}:
            await self._app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        if self._api_key is None:
            await self._reject(
                scope,
                receive,
                send,
                503,
                "api_key_not_configured",
                "API authentication is not configured",
            )
            return
        supplied = headers.get(b"x-api-key", b"").decode("latin-1")
        if not supplied or not secrets.compare_digest(supplied, self._api_key):
            await self._reject(scope, receive, send, 401, "unauthorized", "Invalid API key")
            return

        key_hash = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
        now = time.monotonic()
        async with self._lock:
            recent = [
                timestamp
                for timestamp in self._requests.get(key_hash, [])
                if now - timestamp < self._window_seconds
            ]
            if len(recent) >= self._request_limit:
                self._requests[key_hash] = recent
                limited = True
            else:
                recent.append(now)
                self._requests[key_hash] = recent
                limited = False
        if limited:
            await self._reject(
                scope,
                receive,
                send,
                429,
                "rate_limited",
                "Request rate limit exceeded",
            )
            return
        await self._app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": message}},
        )
        await response(scope, receive, send)


def _environment_number(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _error_body(
    code: str,
    message: str,
    *,
    request_id: str | None = None,
) -> dict[str, dict[str, str]]:
    error = {"code": code, "message": message}
    if request_id is not None:
        error["request_id"] = request_id
    return {"error": error}


def create_app(
    *,
    database_url: str | None = None,
    api_key: str | None = None,
    repository: TaskRepository | None = None,
    llm_client: LLMClient | None = None,
    vendor_api_client: VendorAPIClient | None = None,
    llm_http_client: httpx.AsyncClient | None = None,
    vendor_http_client: httpx.AsyncClient | None = None,
    max_body_bytes: int = 64 * 1024,
    max_goal_length: int = 2000,
    rate_limit_requests: int = 120,
    rate_limit_window_seconds: float = 60.0,
    max_budget_usd: float | None = None,
    max_budget_steps: int = 100,
    shutdown_timeout: float | None = None,
    recovery_stale_seconds: float | None = None,
) -> FastAPI:
    if max_body_bytes < 1 or max_goal_length < 1:
        raise ValueError("Body and goal limits must be positive")
    if rate_limit_requests < 1 or rate_limit_window_seconds <= 0:
        raise ValueError("Rate limits must be positive")
    if max_budget_steps < 1:
        raise ValueError("max_budget_steps must be positive")
    configured_shutdown_timeout = (
        shutdown_timeout
        if shutdown_timeout is not None
        else float(os.environ.get("TASK_SHUTDOWN_TIMEOUT_SECONDS", "30"))
    )
    if not math.isfinite(configured_shutdown_timeout) or configured_shutdown_timeout < 0:
        raise ValueError("shutdown_timeout must be non-negative")
    configured_recovery_stale_seconds = (
        recovery_stale_seconds
        if recovery_stale_seconds is not None
        else float(os.environ.get("TASK_RECOVERY_STALE_SECONDS", "300"))
    )
    if (
        not math.isfinite(configured_recovery_stale_seconds)
        or configured_recovery_stale_seconds <= 0
    ):
        raise ValueError("recovery_stale_seconds must be finite and positive")

    configured_key = api_key if api_key is not None else os.environ.get("VENDOROPS_API_KEY")
    configured_key = configured_key if configured_key and configured_key.strip() else None
    configured_database_url = database_url or os.environ.get("DATABASE_URL")
    budget_limit = (
        max_budget_usd
        if max_budget_usd is not None
        else _environment_number("VENDOROPS_MAX_BUDGET_USD", 0.50)
    )
    if not math.isfinite(budget_limit) or budget_limit <= 0:
        raise ValueError("max_budget_usd must be finite and positive")
    step_limit = int(os.environ.get("VENDOROPS_MAX_STEPS", str(max_budget_steps)))
    if step_limit < 1:
        raise ValueError("VENDOROPS_MAX_STEPS must be positive")
    max_budget_steps = min(max_budget_steps, step_limit)
    engine: AsyncEngine | None = None
    if repository is None and configured_database_url:
        pool_size = int(os.environ.get("DATABASE_POOL_SIZE", "10"))
        pool_overflow = int(os.environ.get("DATABASE_MAX_OVERFLOW", "10"))
        engine = create_task_engine(
            configured_database_url,
            pool_size=pool_size,
            max_overflow=pool_overflow,
        )
        repository = TaskRepository(engine)

    owns_llm_client = llm_client is None
    owns_vendor_api_client = vendor_api_client is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_repository = repository
        active_llm_client = llm_client
        active_vendor_api_client = vendor_api_client
        runner: TaskRunner | None = None
        try:
            if configured_key is None:
                raise RuntimeError("VENDOROPS_API_KEY must be configured")
            if engine is not None:
                await apply_migrations(engine)
            if active_repository is not None:
                if active_llm_client is None:
                    active_llm_client = LLMClient(
                        os.environ.get("LLM_BASE_URL", "http://mock-llm:8001"),
                        client=llm_http_client,
                    )
                if active_vendor_api_client is None:
                    active_vendor_api_client = VendorAPIClient(
                        os.environ.get(
                            "VENDOR_API_BASE_URL", "http://mock-vendor:8002"
                        ),
                        client=vendor_http_client,
                    )
                router = ToolRouter(
                    build_vendor_registry(active_vendor_api_client),
                    active_llm_client,
                )
                agent = ReactAgent(active_llm_client, router, active_repository)
                runner = TaskRunner(
                    active_repository,
                    agent,
                    concurrency_limit=int(os.environ.get("TASK_CONCURRENCY", "200")),
                    shutdown_timeout=configured_shutdown_timeout,
                    stale_after=timedelta(seconds=configured_recovery_stale_seconds),
                )
                application.state.runner = runner
                await runner.recover_stale()
            else:
                application.state.runner = None
            yield
        finally:
            try:
                if runner is not None:
                    await runner.shutdown()
            finally:
                try:
                    if owns_llm_client and active_llm_client is not None:
                        await active_llm_client.aclose()
                finally:
                    try:
                        if owns_vendor_api_client and active_vendor_api_client is not None:
                            await active_vendor_api_client.aclose()
                    finally:
                        if engine is not None:
                            await engine.dispose()

    app = FastAPI(
        title="VendorOps API",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.repository = repository
    app.state.engine = engine
    app.state.runner = None
    app.state.enable_test_hooks = os.environ.get("ENABLE_TEST_HOOKS") == "1"
    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=max_body_bytes)
    app.add_middleware(
        APIKeyAndRateLimitMiddleware,
        api_key=configured_key,
        request_limit=rate_limit_requests,
        window_seconds=rate_limit_window_seconds,
    )
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, error: HTTPException) -> JSONResponse:
        code = {
            404: "not_found",
            405: "method_not_allowed",
        }.get(error.status_code, "http_error")
        message = "Resource was not found" if error.status_code == 404 else "Request failed"
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(code, message),
        )

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, error: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_error_body(error.code, error.message),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body("invalid_request", "Request validation failed"),
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(_request: Request, error: Exception) -> JSONResponse:
        logger.error("Unhandled API request error: %s", type(error).__name__)
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": _request.state.request_id},
            content=_error_body(
                "internal_error",
                "An internal error occurred",
                request_id=_request.state.request_id,
            ),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        active_repository: TaskRepository | None = request.app.state.repository
        active_runner: TaskRunner | None = request.app.state.runner
        if active_repository is None or active_runner is None:
            return JSONResponse(
                status_code=503,
                content=_error_body("not_ready", "Database or task runner is unavailable"),
            )
        try:
            await active_repository.ping()
        except Exception as error:
            logger.warning("Readiness database check failed: %s", type(error).__name__)
            return JSONResponse(
                status_code=503,
                content=_error_body("not_ready", "Database connectivity check failed"),
            )
        if not active_runner.accepting:
            return JSONResponse(
                status_code=503,
                content=_error_body("not_ready", "Task runner is not accepting work"),
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.post("/tasks")
    async def create_task(request: Request, payload: TaskSubmission) -> JSONResponse:
        active_repository: TaskRepository | None = request.app.state.repository
        active_runner: TaskRunner | None = request.app.state.runner
        if active_repository is None or active_runner is None:
            raise APIError(503, "not_ready", "Task service is unavailable")

        if len(payload.goal) > max_goal_length:
            raise APIError(422, "invalid_goal", "Goal exceeds the allowed length")
        scenario = (
            request.headers.get("x-mock-scenario")
            if request.app.state.enable_test_hooks
            else None
        )
        latency_header = (
            request.headers.get("x-mock-latency-ms")
            if request.app.state.enable_test_hooks
            else None
        )
        latency_ms: int | None = None
        if latency_header is not None:
            try:
                latency_ms = int(latency_header)
            except ValueError as error:
                raise APIError(400, "invalid_header", "Mock latency must be an integer") from error
            if latency_ms < 0 or latency_ms > 30_000:
                raise APIError(400, "invalid_header", "Mock latency is outside the allowed range")

        task_id = uuid4()
        budget = Budget(
            max_usd=min(payload.max_usd, budget_limit)
            if payload.max_usd is not None
            else budget_limit,
            max_steps=min(payload.max_steps, max_budget_steps)
            if payload.max_steps is not None
            else max_budget_steps,
        )
        state = TaskState(
            task_id=str(task_id),
            status=TaskStatus.CREATED,
            goal=payload.goal,
            budget=budget,
            llm_scenario=scenario,
            llm_latency_ms=latency_ms,
            updated_at=datetime.now(UTC),
        )
        try:
            record = await active_repository.create_task(
                idempotency_key=payload.idempotency_key,
                goal=payload.goal,
                state=state.model_dump(mode="json"),
                task_id=task_id,
            )
            if not record.was_created and record.goal != payload.goal:
                raise APIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for a different goal",
                )
            active_runner.submit(record.id)
        except APIError:
            raise
        except RunnerClosed as error:
            raise APIError(503, "shutting_down", "Task service is shutting down") from error
        except Exception as error:
            logger.exception("Could not create or enqueue task")
            raise APIError(503, "task_unavailable", "Task could not be accepted") from error

        return JSONResponse(
            status_code=202 if record.was_created else 200,
            content={"task_id": str(record.id), "status": record.status.value},
        )

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: UUID, request: Request) -> dict[str, Any]:
        active_repository: TaskRepository | None = request.app.state.repository
        if active_repository is None:
            raise APIError(503, "not_ready", "Task service is unavailable")
        try:
            record = await active_repository.get_task(task_id)
            events = await active_repository.list_task_events(task_id)
        except TaskNotFound as error:
            raise APIError(404, "task_not_found", "Task was not found") from error
        except Exception as error:
            logger.exception("Could not load task status")
            raise APIError(503, "task_unavailable", "Task status is unavailable") from error

        state = record.state
        event_history = [
            {
                "from_status": event.from_status.value,
                "to_status": event.to_status.value,
                "reason": event.reason,
                "created_at": event.created_at.isoformat(),
            }
            for event in events
        ]
        transition_history = ["created"] + [
            event.to_status.value
            for event in events
            if event.from_status is not event.to_status
        ]
        calls = [
            item
            for item in state.get("history", [])
            if isinstance(item, dict) and "tool" in item and "attempt" in item
        ]
        tool_attempts: list[dict[str, Any]] = []
        if calls:
            tool_attempts.append(
                {
                    "attempt": max(int(call.get("attempt", 1)) for call in calls),
                    "validation_error": any(
                        int(call.get("attempt", 1)) > 1 for call in calls
                    ),
                }
            )
        last_reason = event_history[-1]["reason"] if event_history else None
        result = next(
            (
                item
                for item in reversed(state.get("history", []))
                if isinstance(item, dict) and "ok" in item
            ),
            None,
        )
        return {
            "task_id": str(record.id),
            "status": record.status.value,
            "budget": state.get("budget"),
            "final_answer": state.get("final_answer"),
            "trace_id": record.trace_id,
            "event_history": event_history,
            "transition_history": " -> ".join(transition_history),
            "reason": last_reason,
            "tool_attempts": tool_attempts,
            "result": (
                f"ToolResult(ok={result.get('ok')})"
                if isinstance(result, dict) and "ok" in result
                else None
            ),
            "signals": {"checkpoint": state.get("updated_at")},
        }

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: UUID, request: Request) -> JSONResponse:
        active_repository: TaskRepository | None = request.app.state.repository
        active_runner: TaskRunner | None = request.app.state.runner
        if active_repository is None or active_runner is None:
            raise APIError(503, "not_ready", "Task service is unavailable")
        try:
            record = await active_runner.cancel_task(task_id)
        except TaskNotFound as error:
            raise APIError(404, "task_not_found", "Task was not found") from error
        except InvalidTransition as error:
            raise APIError(
                409,
                "invalid_transition",
                f"Task cannot be cancelled from {error.source.value}",
            ) from error
        except Exception as error:
            logger.exception("Could not cancel task")
            raise APIError(503, "task_unavailable", "Task could not be cancelled") from error
        return JSONResponse(
            status_code=200,
            content={"task_id": str(record.id), "status": record.status.value},
        )

    return app


app = create_app()
