import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from mocks.mock_llm.app import app as mock_llm_app
from mocks.mock_vendor_api.app import app as mock_vendor_app
from mocks.shared_scenarios import SCENARIO_VENDOR_ID
from vendorops.llm.client import BudgetExceeded, LLMClient
from vendorops.models import Budget, TaskState, TaskStatus, ToolCall, ToolResult
from vendorops.tools import (
    ToolDefinition,
    ToolRegistry,
    ToolRouter,
    VendorAPIClient,
    build_vendor_registry,
)
from vendorops.tools.vendor import GetVendorInput, GetVendorOutput


class RecordingTransport(httpx.ASGITransport):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app=app)
        self.payloads: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        return await super().handle_async_request(request)


class ScriptedLLMTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, json=self.responses[len(self.payloads) - 1])

    async def aclose(self) -> None:
        return None


@pytest_asyncio.fixture
async def tool_harness() -> AsyncIterator[
    tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient]
]:
    vendor_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_vendor_app),
        base_url="http://mock-vendor",
    )
    await vendor_http.post(
        "/admin/faults",
        json={"FAIL_404": False, "FAIL_TIMEOUT": False, "RETURN_INJECTION": False},
    )
    vendor = VendorAPIClient("http://mock-vendor", client=vendor_http, retry_base_delay=0)
    llm_transport = RecordingTransport(mock_llm_app)
    llm_http = httpx.AsyncClient(transport=llm_transport, base_url="http://mock-llm")
    llm = LLMClient("http://mock-llm", client=llm_http, retry_base_delay=0)
    router = ToolRouter(build_vendor_registry(vendor), llm)
    state = TaskState(
        task_id="tool-test",
        status=TaskStatus.RUNNING,
        goal="Get vendor details",
        budget=Budget(max_usd=1.0, max_steps=10),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    try:
        yield router, state, llm_transport, vendor_http
    finally:
        await llm_http.aclose()
        await vendor_http.aclose()


@pytest.mark.asyncio
async def test_tool_succeeds_on_first_attempt_and_supplies_json_schemas(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, transport, _ = tool_harness
    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "happy_path"},
    )

    assert result.ok is True
    assert result.output is not None
    assert result.output["id"] == SCENARIO_VENDOR_ID
    assert result.latency_ms >= 0
    assert isinstance(state.history[0], ToolCall)
    assert state.history[0].attempt == 1
    assert transport.payloads[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_vendor",
                "description": "Get vendor details by vendor ID.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {"vendor_id": {"minLength": 1, "title": "Vendor Id", "type": "string"}},
                    "required": ["vendor_id"],
                    "title": "GetVendorInput",
                    "type": "object",
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_invoices",
                "description": "List invoices for a vendor by vendor ID.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {"vendor_id": {"minLength": 1, "title": "Vendor Id", "type": "string"}},
                    "required": ["vendor_id"],
                    "title": "ListInvoicesInput",
                    "type": "object",
                },
            },
        },
    ]


@pytest.mark.asyncio
async def test_bad_arguments_retry_with_validation_error_and_succeed_on_attempt_two(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, transport, _ = tool_harness
    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "bad_params_then_ok"},
    )

    assert result.ok is True
    assert [call.attempt for call in state.history if isinstance(call, ToolCall)] == [1, 2]
    second_prompt = transport.payloads[1]["messages"]
    assert any(
        message.get("role") == "user"
        and "validation error" in message.get("content", "").lower()
        for message in second_prompt
    )
    assert state.budget.steps_taken == 2
    results = [item for item in state.history if isinstance(item, ToolResult)]
    assert len(results) == 1
    assert results[0].ok is True


@pytest.mark.asyncio
async def test_bad_arguments_exhaust_retries_as_tool_failure(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, _, _ = tool_harness
    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "always_bad_params"},
    )

    assert result.ok is False
    assert result.error is not None
    assert "validation error" in result.error.lower()
    assert [call.attempt for call in state.history if isinstance(call, ToolCall)] == [1, 2, 3]
    assert len([item for item in state.history if isinstance(item, ToolResult)]) == 1


@pytest.mark.asyncio
async def test_validation_retry_budget_blocks_before_retry_exhaustion(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, transport, _ = tool_harness
    state.budget.max_usd = 0.00045

    with pytest.raises(BudgetExceeded, match="budget_exceeded"):
        await router.route(
            state,
            [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
            headers={"X-Scenario": "always_bad_params"},
        )

    assert len(transport.payloads) == 1
    assert state.budget.steps_taken == 1
    assert [call.attempt for call in state.history if isinstance(call, ToolCall)] == [1]


@pytest.mark.asyncio
async def test_invalid_args_never_invoke_registered_handler(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, transport, _ = tool_harness
    invocations: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        invocations.append(args)
        return {
            "id": args["vendor_id"],
            "name": "Example",
            "status": "active",
            "note": "",
        }

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "get_vendor",
            "Get vendor details.",
            GetVendorInput,
            GetVendorOutput,
            handler,
        )
    )
    llm_http = httpx.AsyncClient(transport=transport, base_url="http://mock-llm")
    router = ToolRouter(registry, LLMClient("http://mock-llm", client=llm_http))

    result = await router.route(
        state,
        [{"role": "user", "content": "Get vendor"}],
        headers={"X-Scenario": "always_bad_params"},
    )
    await llm_http.aclose()

    assert result.ok is False
    assert invocations == []
    assert len([call for call in state.history if isinstance(call, ToolCall)]) == 3


@pytest.mark.asyncio
async def test_validation_feedback_is_bounded_and_excludes_raw_arguments(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, _, _ = tool_harness
    marker = "attacker-controlled-value"
    invalid_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-invalid",
                            "type": "function",
                            "function": {
                                "name": "get_vendor",
                                "arguments": json.dumps(
                                    {
                                        "vendor_id": 42,
                                        "untrusted_extra": marker * 1000,
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    valid_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-valid",
                            "type": "function",
                            "function": {
                                "name": "get_vendor",
                                "arguments": json.dumps({"vendor_id": SCENARIO_VENDOR_ID}),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    transport = ScriptedLLMTransport([invalid_response, valid_response])
    llm_http = httpx.AsyncClient(transport=transport, base_url="http://mock-llm")
    vendor_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_vendor_app),
        base_url="http://mock-vendor",
    )
    router = ToolRouter(
        build_vendor_registry(VendorAPIClient("http://mock-vendor", client=vendor_http)),
        LLMClient("http://mock-llm", client=llm_http),
    )

    result = await router.route(state, [{"role": "user", "content": "Get vendor"}])
    await llm_http.aclose()
    await vendor_http.aclose()

    assert result.ok is True
    retry_prompt = transport.payloads[1]["messages"]
    feedback = retry_prompt[-1]["content"]
    assert len(feedback) <= 500
    assert marker not in feedback


@pytest.mark.asyncio
async def test_validation_retry_still_obeys_llm_step_limit(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, transport, _ = tool_harness
    state.budget.max_steps = 1

    with pytest.raises(BudgetExceeded, match="step_limit"):
        await router.route(
            state,
            [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
            headers={"X-Scenario": "bad_params_then_ok"},
        )

    assert len(transport.payloads) == 1
    assert state.budget.steps_taken == 1


@pytest.mark.asyncio
async def test_list_invoices_tool_returns_validated_output(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, _, vendor_http = tool_harness
    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-invoices",
                            "type": "function",
                            "function": {
                                "name": "list_invoices",
                                "arguments": '{"vendor_id":"V-100"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    llm_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=response_payload)
        ),
        base_url="http://mock-llm",
    )
    vendor = VendorAPIClient("http://mock-vendor", client=vendor_http)
    router = ToolRouter(build_vendor_registry(vendor), LLMClient("http://mock-llm", client=llm_http))

    result = await router.route(state, [{"role": "user", "content": "List invoices"}])
    await llm_http.aclose()

    assert result.ok is True
    assert result.output is not None
    assert result.output["vendor_id"] == "V-100"
    assert len(result.output["invoices"]) == 2


@pytest.mark.asyncio
async def test_unknown_tool_name_returns_failure_without_raising(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, _, vendor_http = tool_harness
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-unknown",
                            "type": "function",
                            "function": {"name": "delete_vendor", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    llm_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        base_url="http://mock-llm",
    )
    vendor = VendorAPIClient("http://mock-vendor", client=vendor_http)
    router = ToolRouter(build_vendor_registry(vendor), LLMClient("http://mock-llm", client=llm_http))
    result = await router.route(state, [{"role": "user", "content": "Do something"}])
    await llm_http.aclose()

    assert result.ok is False
    assert result.call_id == "call-unknown"
    assert result.error == "Unknown tool: delete_vendor"


@pytest.mark.asyncio
async def test_malformed_llm_json_returns_failure(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, _, _ = tool_harness
    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "malformed_json"},
    )

    assert result.ok is False
    assert result.error is not None
    assert "malformed json" in result.error.lower()


class TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Mock vendor API timed out", request=request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_router_caps_total_tool_call_duration(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, transport, _ = tool_harness
    llm_http = httpx.AsyncClient(transport=transport, base_url="http://mock-llm")
    router = ToolRouter(
        ToolRegistry(),
        LLMClient("http://mock-llm", client=llm_http),
        total_timeout=0.01,
    )
    started = asyncio.get_running_loop().time()

    result = await router.route(
        state,
        [{"role": "user", "content": "Get vendor"}],
        headers={"X-Scenario": "happy_path", "X-Latency-Ms": "1000"},
    )
    elapsed = asyncio.get_running_loop().time() - started
    await llm_http.aclose()

    assert result.ok is False
    assert "time limit" in (result.error or "")
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_router_propagates_cancellation_from_tool_handler(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, transport, _ = tool_harness

    async def cancel_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise asyncio.CancelledError

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "get_vendor",
            "Get vendor details.",
            GetVendorInput,
            GetVendorOutput,
            cancel_handler,
        )
    )
    llm_http = httpx.AsyncClient(transport=transport, base_url="http://mock-llm")
    router = ToolRouter(registry, LLMClient("http://mock-llm", client=llm_http))

    with pytest.raises(asyncio.CancelledError):
        await router.route(
            state,
            [{"role": "user", "content": "Get vendor"}],
            headers={"X-Scenario": "happy_path"},
        )
    await llm_http.aclose()


@pytest.mark.asyncio
async def test_vendor_timeout_returns_tool_failure(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    _, state, llm_transport, _ = tool_harness
    timeout_http = httpx.AsyncClient(transport=TimeoutTransport(), base_url="http://mock-vendor")
    vendor = VendorAPIClient(
        "http://mock-vendor",
        client=timeout_http,
        max_attempts=1,
        timeout=0.01,
    )
    llm_http = httpx.AsyncClient(
        transport=llm_transport,
        base_url="http://mock-llm",
    )
    router = ToolRouter(build_vendor_registry(vendor), LLMClient("http://mock-llm", client=llm_http))
    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "happy_path"},
    )
    await llm_http.aclose()
    await timeout_http.aclose()

    assert result.ok is False
    assert "timed out" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_mock_vendor_timeout_fault_returns_tool_failure(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state, llm_transport, vendor_http = tool_harness
    monkeypatch.setenv("MOCK_VENDOR_TIMEOUT_MS", "1")
    await vendor_http.post("/admin/faults", json={"FAIL_TIMEOUT": True})
    llm_http = httpx.AsyncClient(transport=llm_transport, base_url="http://mock-llm")
    vendor = VendorAPIClient(
        "http://mock-vendor",
        client=vendor_http,
        max_attempts=1,
        timeout=1.0,
    )
    router = ToolRouter(
        build_vendor_registry(vendor),
        LLMClient("http://mock-llm", client=llm_http),
    )

    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "happy_path"},
    )
    await llm_http.aclose()

    assert result.ok is False
    assert "504" in (result.error or "")


@pytest.mark.asyncio
async def test_vendor_404_returns_tool_failure(
    tool_harness: tuple[ToolRouter, TaskState, RecordingTransport, httpx.AsyncClient],
) -> None:
    router, state, _, vendor_http = tool_harness
    await vendor_http.post("/admin/faults", json={"FAIL_404": True})

    result = await router.route(
        state,
        [{"role": "user", "content": f"Get vendor {SCENARIO_VENDOR_ID}"}],
        headers={"X-Scenario": "happy_path"},
    )

    assert result.ok is False
    assert result.error is not None
    assert "404" in result.error
