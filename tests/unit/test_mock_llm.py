import asyncio
import json
from typing import Any

import httpx
import pytest

from mocks.mock_llm.app import app


def _payload(messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": messages or [{"role": "user", "content": "Look up vendor V-42"}],
    }


@pytest.mark.asyncio
async def test_default_scenario_returns_openai_tool_call_and_token_usage() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "chat.completion"
    assert result["model"] == "test-model"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_vendor"
    assert json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "vendor_id": "V-42"
    }
    assert result["usage"]["total_tokens"] == (
        result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
    )


@pytest.mark.asyncio
async def test_bad_params_then_ok_returns_valid_args_after_validation_feedback() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        first = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "bad_params_then_ok"}, json=_payload()
        )
        retried_messages: list[dict[str, Any]] = [
            {"role": "assistant", "tool_calls": [{"id": "call-mock-1"}]},
            {"role": "tool", "content": "Validation error: vendor_id must be a string"},
        ]
        retry = await client.post(
            "/v1/chat/completions",
            headers={"X-Scenario": "bad_params_then_ok"},
            json=_payload(retried_messages),
        )

    first_args = json.loads(first.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    retry_args = json.loads(retry.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert first_args["vendor_id"] == 42
    assert retry_args["vendor_id"] == "V-42"


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["always_bad_params", "schema_violating_args"])
async def test_invalid_argument_scenarios_return_schema_violating_args(scenario: str) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": scenario}, json=_payload()
        )

    arguments = json.loads(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert arguments["vendor_id"] == 42


@pytest.mark.asyncio
async def test_repeat_same_call_scenario_is_stable() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        first = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "repeat_same_call"}, json=_payload()
        )
        second = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "repeat_same_call"}, json=_payload()
        )

    first_call = first.json()["choices"][0]["message"]["tool_calls"][0]
    second_call = second.json()["choices"][0]["message"]["tool_calls"][0]
    assert first_call["function"] == second_call["function"]


@pytest.mark.asyncio
async def test_expensive_scenario_reports_configurable_high_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_EXPENSIVE_PROMPT_TOKENS", "1234")
    monkeypatch.setenv("MOCK_LLM_EXPENSIVE_COMPLETION_TOKENS", "567")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "expensive_budget"}, json=_payload()
        )

    assert response.json()["usage"] == {
        "prompt_tokens": 1234,
        "completion_tokens": 567,
        "total_tokens": 1801,
    }


@pytest.mark.asyncio
async def test_normal_token_counts_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_PROMPT_TOKENS", "256")
    monkeypatch.setenv("MOCK_LLM_COMPLETION_TOKENS", "64")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.json()["usage"] == {
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "total_tokens": 320,
    }


@pytest.mark.asyncio
async def test_latency_distribution_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_LATENCY_MIN_MS", "15")
    monkeypatch.setenv("MOCK_LLM_LATENCY_MAX_MS", "15")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        started = asyncio.get_running_loop().time()
        response = await client.post("/v1/chat/completions", json=_payload())
        elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 200
    assert elapsed >= 0.015


@pytest.mark.asyncio
async def test_per_request_latency_header_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_LATENCY_MIN_MS", "0")
    monkeypatch.setenv("MOCK_LLM_LATENCY_MAX_MS", "0")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        started = asyncio.get_running_loop().time()
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Latency-Ms": "20"},
            json=_payload(),
        )
        elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 200
    assert elapsed >= 0.01


@pytest.mark.asyncio
async def test_unknown_scenario_returns_clear_bad_request() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Scenario": "misspelled_scenario"},
            json=_payload(),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown scenario: misspelled_scenario"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_status"),
    [("http_429", 429), ("http_500", 500)],
)
async def test_http_failure_scenarios(scenario: str, expected_status: int) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": scenario}, json=_payload()
        )

    assert response.status_code == expected_status
    assert response.json()["error"]


@pytest.mark.asyncio
async def test_malformed_json_scenario_returns_invalid_json() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "malformed_json"}, json=_payload()
        )

    assert response.status_code == 200
    with pytest.raises(json.JSONDecodeError):
        response.json()


@pytest.mark.asyncio
async def test_timeout_scenario_exceeds_short_client_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_TIMEOUT_MS", "100")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client.post("/v1/chat/completions", headers={"X-Scenario": "timeout"}, json=_payload()),
                timeout=0.01,
            )


@pytest.mark.asyncio
async def test_slow_tool_scenario_waits_for_configured_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM_SLOW_MS", "15")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        started = asyncio.get_running_loop().time()
        response = await client.post(
            "/v1/chat/completions", headers={"X-Scenario": "slow_tool"}, json=_payload()
        )
        elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 200
    assert elapsed >= 0.015
