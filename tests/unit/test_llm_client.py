import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from mocks.mock_llm.app import app as mock_llm_app
from vendorops.llm.client import BudgetExceeded, LLMAPIError, LLMClient
from vendorops.llm.cost import ModelPricing, PriceTable, load_price_table
from vendorops.llm.loop import LoopDetected, ToolLoopDetector
from vendorops.models import Budget, TaskState, TaskStatus

TEST_PRICES = PriceTable(
    {
        "mock-llm": ModelPricing(1.0, 2.0),
        "mock-llm-cheap": ModelPricing(0.1, 0.2),
    }
)


def _state(*, max_usd: float = 1.0, max_steps: int = 20, steps_taken: int = 0) -> TaskState:
    return TaskState(
        task_id="test-task",
        status=TaskStatus.RUNNING,
        goal="Test LLM calls",
        budget=Budget(max_usd=max_usd, max_steps=max_steps, steps_taken=steps_taken),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _client(*, max_attempts: int = 3, timeout: float = 1.0) -> LLMClient:
    transport = httpx.ASGITransport(app=mock_llm_app)
    http_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://mock",
        timeout=timeout,
    )
    return LLMClient(
        "http://mock",
        client=http_client,
        max_attempts=max_attempts,
        retry_base_delay=0,
        retry_max_delay=0,
        prices=TEST_PRICES,
    )


def test_price_table_loads_configured_models_and_calculates_cost() -> None:
    prices = load_price_table()

    assert prices.calculate(
        "mock-llm",
        {"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
    ) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_normal_call_updates_task_budget_with_actual_usage() -> None:
    state = _state()
    async with _client() as client:
        result = await client.complete(
            state,
            [{"role": "user", "content": "Find vendor V-42"}],
            headers={"X-Scenario": "happy_path"},
            estimated_prompt_tokens=128,
            estimated_completion_tokens=32,
        )

    assert result.response["object"] == "chat.completion"
    assert result.model == "mock-llm"
    assert result.attempts == 1
    assert result.cost_usd == pytest.approx((128 * 1.0 + 32 * 2.0) / 1_000_000)
    assert state.budget.spent_usd == result.cost_usd
    assert state.budget.steps_taken == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["http_429_then_ok", "http_500_then_ok"])
async def test_client_retries_transient_errors_then_succeeds(scenario: str) -> None:
    state = _state()
    async with _client(max_attempts=2) as client:
        result = await client.complete(
            state,
            [{"role": "user", "content": "Retry request"}],
            headers={"X-Scenario": scenario},
        )

    assert result.attempts == 2
    assert result.response["usage"]["prompt_tokens"] > 0
    assert state.budget.steps_taken == 1


@pytest.mark.asyncio
async def test_client_does_not_retry_other_4xx_responses() -> None:
    state = _state()
    async with _client(max_attempts=4) as client:
        with pytest.raises(LLMAPIError) as error:
            await client.complete(
                state,
                [{"role": "user", "content": "Bad request"}],
                headers={"X-Scenario": "http_400"},
            )

    assert error.value.status_code == 400
    assert error.value.attempts == 1
    assert state.budget.steps_taken == 0


@pytest.mark.asyncio
async def test_client_timeout_is_reported() -> None:
    state = _state()
    async with _client(timeout=0.01) as client:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                client.complete(
                    state,
                    [{"role": "user", "content": "Wait"}],
                    headers={"X-Scenario": "timeout"},
                ),
                timeout=0.01,
            )


@pytest.mark.asyncio
async def test_exact_budget_boundary_is_allowed_and_accounted() -> None:
    expected_cost = TEST_PRICES.estimate("mock-llm", 128, 32)
    state = _state(max_usd=expected_cost)
    async with _client() as client:
        result = await client.complete(
            state,
            [{"role": "user", "content": "Boundary"}],
            estimated_prompt_tokens=128,
            estimated_completion_tokens=32,
        )

    assert result.cost_usd == expected_cost
    assert state.budget.spent_usd == expected_cost
    assert state.budget.steps_taken == 1


@pytest.mark.asyncio
async def test_call_is_rejected_when_step_limit_is_reached() -> None:
    state = _state(max_steps=2, steps_taken=2)
    async with _client() as client:
        with pytest.raises(BudgetExceeded) as error:
            await client.complete(state, [{"role": "user", "content": "Over step limit"}])

    assert error.value.reason == "step_limit"
    assert state.budget.steps_taken == 2


@pytest.mark.asyncio
async def test_expensive_primary_uses_fallback_model_when_its_cost_fits() -> None:
    state = _state(max_usd=0.001)
    async with _client() as client:
        result = await client.complete(
            state,
            [{"role": "user", "content": "Use fallback"}],
            model="mock-llm",
            fallback_model="mock-llm-cheap",
            fallback_on_budget=True,
            estimated_prompt_tokens=1_000,
            estimated_completion_tokens=1_000,
        )

    assert result.model == "mock-llm-cheap"
    assert result.response["model"] == "mock-llm-cheap"
    assert state.budget.spent_usd < state.budget.max_usd
    assert state.budget.steps_taken == 1


@pytest.mark.asyncio
async def test_budget_exceeded_when_projected_cost_is_over_limit() -> None:
    state = _state(max_usd=0.0001)
    async with _client() as client:
        with pytest.raises(BudgetExceeded) as error:
            await client.complete(
                state,
                [{"role": "user", "content": "Too expensive"}],
                estimated_prompt_tokens=100,
                estimated_completion_tokens=100,
            )

    assert error.value.reason == "budget_exceeded"
    assert state.budget.steps_taken == 0


def test_loop_detector_raises_on_third_consecutive_identical_call() -> None:
    detector = ToolLoopDetector()

    detector.observe("get_vendor", {"vendor_id": "V-42"})
    detector.observe("get_vendor", {"vendor_id": "V-42"})
    with pytest.raises(LoopDetected) as error:
        detector.observe("get_vendor", {"vendor_id": "V-42"})

    assert error.value.repeats == 3


def test_different_call_resets_consecutive_loop_count() -> None:
    detector = ToolLoopDetector()
    call = {"vendor_id": "V-42"}

    detector.observe("get_vendor", call)
    detector.observe("get_vendor", call)
    detector.observe("list_invoices", call)
    detector.observe("get_vendor", call)
    detector.observe("get_vendor", call)

    with pytest.raises(LoopDetected) as error:
        detector.observe("get_vendor", call)
    assert error.value.repeats == 3


def test_different_arguments_reset_consecutive_loop_count() -> None:
    detector = ToolLoopDetector()

    detector.observe("get_vendor", {"vendor_id": "V-42"})
    detector.observe("get_vendor", {"vendor_id": "V-42"})
    detector.observe("get_vendor", {"vendor_id": "V-99"})
    detector.observe("get_vendor", {"vendor_id": "V-42"})

    assert detector.observe("get_vendor", {"vendor_id": "V-42"})
