import asyncio
import json
import os
import random
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response

from mocks.shared_scenarios import SCENARIO_VENDOR_ID

app = FastAPI(title="VendorOps Mock LLM")
SUPPORTED_SCENARIOS = {
    "default",
    "happy_path",
    "bad_params_then_ok",
    "always_bad_params",
    "schema_violating_args",
    "repeat_same_call",
    "expensive_budget",
    "slow_tool",
    "http_429",
    "http_429_then_ok",
    "http_429_with_usage_then_ok",
    "http_429_retry_after_then_ok",
    "rate_limit",
    "429",
    "http_500",
    "http_500_then_ok",
    "server_error",
    "500",
    "http_400",
    "client_error",
    "400",
    "http_401",
    "http_404",
    "timeout",
    "timed_out",
    "malformed_json",
    "malformed",
}


def _environment_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _scenario_response(scenario: str, messages: list[Any]) -> tuple[str, str]:
    vendor_id = SCENARIO_VENDOR_ID
    arguments: dict[str, Any] = {"vendor_id": vendor_id}

    if scenario in {"bad_params_then_ok", "always_bad_params", "schema_violating_args"}:
        has_prior_tool_call = any(
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and "tool_calls" in message
            for message in messages
        )
        has_validation_feedback = any(
            isinstance(message, dict)
            and isinstance(message.get("content"), str)
            and any(word in message["content"].lower() for word in ("validation", "error", "required"))
            for message in messages
        )
        retry_with_error = has_prior_tool_call and has_validation_feedback
        if scenario != "bad_params_then_ok" or not retry_with_error:
            arguments = {"vendor_id": 42}

    tool = "get_vendor"
    if scenario == "repeat_same_call":
        arguments = {"vendor_id": vendor_id}
    elif scenario in {"happy_path", "default", "slow_tool", "expensive_budget"}:
        tool = "get_vendor"

    return tool, json.dumps(arguments)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict[str, Any],
    x_scenario: str | None = Header(default=None, alias="X-Scenario"),
    x_latency_ms: int | None = Header(default=None, alias="X-Latency-Ms"),
    x_mock_attempt: int = Header(default=1, alias="X-Mock-Attempt"),
) -> Response:
    scenario = (x_scenario or "default").strip().lower().replace("-", "_")
    if scenario not in SUPPORTED_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")

    if scenario in {
        "http_429_then_ok",
        "http_429_with_usage_then_ok",
        "http_429_retry_after_then_ok",
        "429_then_ok",
    } and x_mock_attempt > 1:
        scenario = "default"
    if scenario == "http_500_then_ok" and x_mock_attempt > 1:
        scenario = "default"
    if scenario in {
        "http_429",
        "http_429_then_ok",
        "http_429_with_usage_then_ok",
        "http_429_retry_after_then_ok",
        "rate_limit",
        "429",
        "429_then_ok",
    }:
        content: dict[str, Any] = {
            "error": {"message": "Mock rate limit exceeded", "type": "rate_limit_error"}
        }
        if scenario == "http_429_with_usage_then_ok":
            content["usage"] = {"prompt_tokens": 128, "completion_tokens": 32}
        return JSONResponse(
            status_code=429,
            content=content,
            headers={"Retry-After": "3"} if scenario == "http_429_retry_after_then_ok" else None,
        )
    if scenario in {"http_500", "http_500_then_ok", "server_error", "500"}:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Mock internal server error", "type": "server_error"}},
        )
    if scenario in {"http_400", "client_error", "400"}:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Mock invalid request", "type": "invalid_request_error"}},
        )
    if scenario == "http_401":
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Mock unauthorized", "type": "authentication_error"}},
        )
    if scenario == "http_404":
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Mock resource not found", "type": "not_found_error"}},
        )
    if scenario in {"malformed_json", "malformed"}:
        return Response(content="{", status_code=200, media_type="application/json")
    if scenario in {"timeout", "timed_out"}:
        await asyncio.sleep(_environment_int("MOCK_LLM_TIMEOUT_MS", 30_000) / 1000)

    latency_min_ms = _environment_int("MOCK_LLM_LATENCY_MIN_MS", 0)
    latency_max_ms = _environment_int("MOCK_LLM_LATENCY_MAX_MS", latency_min_ms)
    if latency_min_ms < 0 or latency_max_ms < latency_min_ms:
        raise HTTPException(
            status_code=500, detail="MOCK_LLM_LATENCY_MIN_MS/MAX_MS must define a non-negative range"
        )
    latency_ms = (
        x_latency_ms
        if x_latency_ms is not None
        else random.randint(latency_min_ms, latency_max_ms)
    )
    if latency_ms < 0:
        raise HTTPException(status_code=400, detail="X-Latency-Ms must be non-negative")
    if scenario == "slow_tool" and x_latency_ms is None:
        latency_ms += _environment_int("MOCK_LLM_SLOW_MS", 5_000)
    await asyncio.sleep(latency_ms / 1000)

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "'messages' must be a list", "type": "invalid_request_error"}},
        )

    completed_tool = scenario != "repeat_same_call" and any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and isinstance(message.get("tool_call_id"), str)
        for message in messages
    )
    tool, arguments = _scenario_response(scenario, messages)
    prompt_tokens = _environment_int("MOCK_LLM_PROMPT_TOKENS", 128)
    completion_tokens = _environment_int("MOCK_LLM_COMPLETION_TOKENS", 32)
    if scenario == "expensive_budget":
        prompt_tokens = _environment_int("MOCK_LLM_EXPENSIVE_PROMPT_TOKENS", 10_000)
        completion_tokens = _environment_int("MOCK_LLM_EXPENSIVE_COMPLETION_TOKENS", 10_000)

    assistant_message: dict[str, Any]
    finish_reason: str
    if completed_tool:
        assistant_message = {"role": "assistant", "content": "Task completed."}
        finish_reason = "stop"
    else:
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-mock-1",
                    "type": "function",
                    "function": {"name": tool, "arguments": arguments},
                }
            ],
        }
        finish_reason = "tool_calls"

    response = {
        "id": f"chatcmpl-mock-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "mock-llm"),
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return JSONResponse(content=response)
