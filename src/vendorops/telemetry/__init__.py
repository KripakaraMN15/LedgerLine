"""OpenTelemetry configuration and privacy-conscious span helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Link, Span, SpanContext, Status, StatusCode, TraceFlags

logger = logging.getLogger(__name__)
_provider: TracerProvider = TracerProvider()
_global_provider_installed = False


def configure_tracing(
    *,
    exporter: SpanExporter | None = None,
    service_name: str | None = None,
    set_global: bool = True,
) -> TracerProvider:
    """Configure tracing with an injected exporter or an environment-driven OTLP exporter."""
    global _provider, _global_provider_installed

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": service_name or os.environ.get("OTEL_SERVICE_NAME", "vendorops")}
        )
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        traces_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        endpoint = traces_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint and not traces_endpoint and not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint.rstrip('/')}/v1/traces"
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").lower()
        if endpoint and protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            headers = _otlp_headers(
                os.environ.get(
                    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
                    os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""),
                )
            )
            exporter_timeout_ms = int(
                os.environ.get(
                    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
                    os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "2000"),
                )
            )
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=endpoint,
                        headers=headers,
                        timeout=exporter_timeout_ms / 1000,
                    ),
                    export_timeout_millis=float(
                        os.environ.get("OTEL_BSP_EXPORT_TIMEOUT", "1500")
                    ),
                )
            )
        elif endpoint:
            raise ValueError(f"Unsupported OTLP protocol: {protocol}")

    _provider = provider
    if set_global and not _global_provider_installed:
        trace.set_tracer_provider(provider)
        _global_provider_installed = True
    return provider


def shutdown_tracing(timeout_millis: int = 1500) -> None:
    flush_tracing(timeout_millis)
    shutdown_thread = threading.Thread(target=_provider.shutdown, daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(timeout_millis / 1000)
    if shutdown_thread.is_alive():
        logger.error("OpenTelemetry provider shutdown exceeded %dms", timeout_millis)


def flush_tracing(timeout_millis: int = 1500) -> bool:
    if timeout_millis <= 0:
        raise ValueError("timeout_millis must be positive")
    try:
        flushed = _provider.force_flush(timeout_millis=timeout_millis)
    except Exception:
        logger.exception("Could not flush OpenTelemetry spans")
        return False
    if not flushed:
        logger.error("OpenTelemetry span flush exceeded %dms", timeout_millis)
    return flushed


def _otlp_headers(raw_headers: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for pair in raw_headers.split(","):
        key, separator, value = pair.partition("=")
        if separator and key.strip():
            headers[key.strip()] = value.strip()
    return headers


def _tracer() -> trace.Tracer:
    return _provider.get_tracer("vendorops")


def _safe_reason(reason: str) -> str:
    if reason and len(reason) <= 64 and all(
        character.isalnum() or character in "_.-" for character in reason
    ):
        return reason
    return "unspecified"


def _safe_tool_name(tool: str) -> str:
    if len(tool) <= 64 and re.fullmatch(r"[A-Za-z0-9_.-]+", tool):
        return tool
    return "unknown"


def args_hash(args: Mapping[str, Any]) -> str:
    import json

    serialized = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
    span_attributes: dict[str, Any] = {}
    if attributes is not None:
        span_attributes.update(attributes)
    with _tracer().start_as_current_span(
        name,
        attributes=span_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as error:
            _set_error(current, error)
            raise


def _set_error(current: Span, error: Exception) -> None:
    set_span_error(current, type(error).__name__)


def set_span_error(current: Span, error_type: str) -> None:
    current.set_attribute("exception.type", error_type)
    current.set_status(Status(StatusCode.ERROR, error_type))


@contextmanager
def task_span(
    task_id: str,
    goal: str,
    *,
    linked_trace_id: str | None = None,
) -> Iterator[Span]:
    links: list[Link] = []
    attributes: dict[str, Any] = {
        "task_id": task_id,
        "goal": hashlib.sha256(goal.encode("utf-8")).hexdigest(),
        "goal_size": len(goal),
    }
    if linked_trace_id:
        normalized = linked_trace_id.removeprefix("0x")
        try:
            parsed_trace_id = int(normalized, 16)
        except ValueError as error:
            raise ValueError("linked_trace_id must be hexadecimal") from error
        if parsed_trace_id <= 0 or parsed_trace_id >= 2**128:
            raise ValueError("linked_trace_id is outside the OpenTelemetry range")
        parent = SpanContext(
            trace_id=parsed_trace_id,
            span_id=secrets.randbits(64) or 1,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        links.append(Link(parent))
        attributes["previous_trace_id"] = f"{parsed_trace_id:032x}"
    with _tracer().start_as_current_span(
        "task",
        context=Context(),
        attributes=attributes,
        links=links,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as error:
            _set_error(current, error)
            raise


@contextmanager
def llm_call_span(
    *,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Iterator[Span]:
    with span(
        "llm.call",
        {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    ) as current:
        yield current


@contextmanager
def tool_call_span(tool: str, args: Mapping[str, Any], attempt: int) -> Iterator[Span]:
    with span(
        "tool.call",
        {"tool": _safe_tool_name(tool), "args_hash": args_hash(args), "attempt": attempt},
    ) as current:
        yield current


@contextmanager
def state_transition_span(from_status: str, to_status: str, reason: str) -> Iterator[Span]:
    with span(
        "state.transition",
        {
            "from": from_status,
            "to": to_status,
            "reason": _safe_reason(reason),
        },
    ) as current:
        yield current


@contextmanager
def killswitch_check_span(
    projected_usd: float,
    limit_usd: float,
    *,
    tripped: bool,
) -> Iterator[Span]:
    with span(
        "killswitch.check",
        {
            "projected_usd": projected_usd,
            "limit_usd": limit_usd,
            "tripped": tripped,
        },
    ) as current:
        yield current


@contextmanager
def context_assemble_span(
    *,
    budget_tokens: int,
    system_tokens: int,
    docs_tokens: int,
    tools_tokens: int,
    dropped_docs: int,
) -> Iterator[Span]:
    with span(
        "context.assemble",
        {
            "budget_tokens": budget_tokens,
            "system_tokens": system_tokens,
            "docs_tokens": docs_tokens,
            "tools_tokens": tools_tokens,
            "dropped_docs": dropped_docs,
        },
    ) as current:
        yield current


__all__ = [
    "args_hash",
    "configure_tracing",
    "context_assemble_span",
    "flush_tracing",
    "killswitch_check_span",
    "llm_call_span",
    "shutdown_tracing",
    "set_span_error",
    "span",
    "state_transition_span",
    "task_span",
    "tool_call_span",
]

configure_tracing()
