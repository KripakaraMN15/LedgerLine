from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from vendorops.llm.cost import PriceTable, load_price_table
from vendorops.models import TaskState
from vendorops.telemetry import killswitch_check_span, llm_call_span


class BudgetExceeded(Exception):
    def __init__(self, projected_usd: float, limit_usd: float, reason: str) -> None:
        self.projected_usd = projected_usd
        self.limit_usd = limit_usd
        self.reason = reason
        super().__init__(
            f"LLM call blocked ({reason}): projected ${projected_usd:.8f}, "
            f"budget limit ${limit_usd:.8f}"
        )


class LLMResponseError(Exception):
    pass


class LLMAPIError(Exception):
    def __init__(self, status_code: int, attempts: int, message: str) -> None:
        self.status_code = status_code
        self.attempts = attempts
        super().__init__(message)


@dataclass(frozen=True)
class LLMCompletion:
    response: dict[str, Any]
    model: str
    cost_usd: float
    attempts: int


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    prompt: Any = messages if tools is None else {"messages": messages, "tools": tools}
    serialized = json.dumps(prompt, separators=(",", ":"), ensure_ascii=True)
    return max(1, math.ceil(len(serialized) / 4))


class LLMClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_attempts: int = 3,
        retry_base_delay: float = 0.25,
        retry_max_delay: float = 4.0,
        prices: PriceTable | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if timeout <= 0 or retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("Timeout and retry delays must be non-negative")
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._api_key = api_key
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._prices = prices or load_price_table()
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _record_usage(
        self,
        state: TaskState,
        model: str,
        usage: Mapping[str, Any],
        *,
        successful_call: bool,
    ) -> float:
        cost = self._prices.calculate(model, usage)
        state.budget.spent_usd += cost
        if successful_call:
            state.budget.steps_taken += 1
        return cost

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(tzinfo=UTC)
                        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
        ceiling = min(
            self._retry_max_delay,
            self._retry_base_delay * (2 ** (attempt - 1)),
        )
        return self._jitter(0.0, ceiling)

    async def complete(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        model: str = "mock-llm",
        fallback_model: str | None = None,
        fallback_on_budget: bool = False,
        headers: Mapping[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        estimated_prompt_tokens: int | None = None,
        estimated_completion_tokens: int | None = None,
    ) -> LLMCompletion:
        prompt_tokens = (
            estimated_prompt_tokens
            if estimated_prompt_tokens is not None
            else estimate_prompt_tokens(messages, tools)
        )
        completion_tokens = (
            estimated_completion_tokens
            if estimated_completion_tokens is not None
            else max_tokens
        )
        initial_spend = state.budget.spent_usd
        started = time.perf_counter()
        attempts = 1
        selected_model = model
        with llm_call_span(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ) as current_span:
            try:
                completion = await self._complete(
                    state,
                    messages,
                    model=model,
                    fallback_model=fallback_model,
                    fallback_on_budget=fallback_on_budget,
                    headers=headers,
                    tools=tools,
                    max_tokens=max_tokens,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    estimated_completion_tokens=estimated_completion_tokens,
                )
                attempts = completion.attempts
                selected_model = completion.model
                usage = completion.response.get("usage")
                if isinstance(usage, Mapping):
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(
                        usage.get("completion_tokens", completion_tokens)
                    )
                current_span.set_attribute("usd", completion.cost_usd)
                return completion
            except Exception as error:
                attempts = error.attempts if isinstance(error, LLMAPIError) else 1
                current_span.set_attribute("error", type(error).__name__)
                raise
            finally:
                current_span.set_attribute("model", selected_model)
                current_span.set_attribute("prompt_tokens", prompt_tokens)
                current_span.set_attribute("completion_tokens", completion_tokens)
                current_span.set_attribute("usd", state.budget.spent_usd - initial_spend)
                current_span.set_attribute("latency_ms", round((time.perf_counter() - started) * 1000))
                current_span.set_attribute("attempt", attempts)

    async def _complete(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        model: str = "mock-llm",
        fallback_model: str | None = None,
        fallback_on_budget: bool = False,
        headers: Mapping[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 128,
        estimated_prompt_tokens: int | None = None,
        estimated_completion_tokens: int | None = None,
    ) -> LLMCompletion:
        budget = state.budget
        if budget.steps_taken >= budget.max_steps:
            with killswitch_check_span(
                budget.spent_usd, budget.max_usd, tripped=True
            ):
                raise BudgetExceeded(budget.spent_usd, budget.max_usd, "step_limit")

        prompt_tokens = (
            estimated_prompt_tokens
            if estimated_prompt_tokens is not None
            else estimate_prompt_tokens(messages, tools)
        )
        completion_tokens = (
            estimated_completion_tokens
            if estimated_completion_tokens is not None
            else max_tokens
        )
        if prompt_tokens < 0 or completion_tokens < 0 or max_tokens < 0:
            raise ValueError("Token estimates and max_tokens must be non-negative")

        selected_model = model
        request_headers = dict(headers or {})
        if self._api_key is not None:
            request_headers["Authorization"] = f"Bearer {self._api_key}"
        fallback_used = False

        for attempt in range(1, self._max_attempts + 1):
            if budget.steps_taken >= budget.max_steps:
                with killswitch_check_span(
                    budget.spent_usd, budget.max_usd, tripped=True
                ):
                    raise BudgetExceeded(budget.spent_usd, budget.max_usd, "step_limit")
            projected = budget.spent_usd + self._prices.estimate(
                selected_model, prompt_tokens, completion_tokens
            )
            over_budget = projected > budget.max_usd
            with killswitch_check_span(
                projected, budget.max_usd, tripped=over_budget
            ):
                pass
            if over_budget:
                primary_cost = self._prices.estimate(model, prompt_tokens, completion_tokens)
                fallback_cost = (
                    self._prices.estimate(fallback_model, prompt_tokens, completion_tokens)
                    if fallback_on_budget and fallback_model is not None and not fallback_used
                    else math.inf
                )
                fallback_projected = budget.spent_usd + fallback_cost
                if (
                    fallback_model is not None
                    and selected_model != fallback_model
                    and fallback_cost < primary_cost
                    and fallback_projected <= budget.max_usd
                ):
                    selected_model = fallback_model
                    fallback_used = True
                else:
                    raise BudgetExceeded(projected, budget.max_usd, "budget_exceeded")

            request_headers["X-Mock-Attempt"] = str(attempt)
            payload = {
                "model": selected_model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if tools is not None:
                payload["tools"] = tools
            try:
                response = await self._client.post(
                    self._url,
                    json=payload,
                    headers=request_headers,
                    timeout=self._timeout,
                )
            except httpx.TransportError as error:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(None, attempt))
                    continue
                reason = "timed out" if isinstance(error, httpx.TimeoutException) else "failed"
                raise LLMAPIError(0, attempt, f"LLM request {reason}") from error

            try:
                result: Any = response.json()
            except json.JSONDecodeError:
                result = None
            if response.is_success and result is None:
                raise LLMResponseError("LLM returned malformed JSON")
            usage = result.get("usage") if isinstance(result, dict) else None
            if isinstance(usage, dict):
                actual_cost = self._record_usage(
                    state,
                    selected_model,
                    usage,
                    successful_call=response.is_success,
                )
            elif response.is_success:
                raise LLMResponseError("LLM response is missing usage data")
            else:
                actual_cost = 0.0
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(response, attempt))
                    continue
                raise LLMAPIError(
                    response.status_code,
                    attempt,
                    f"LLM request failed after {attempt} attempts with HTTP "
                    f"{response.status_code}",
                )
            if response.is_error:
                raise LLMAPIError(
                    response.status_code,
                    attempt,
                    f"LLM request failed with HTTP {response.status_code}",
                )
            if result is None:
                raise LLMResponseError("LLM returned malformed JSON")
            if not isinstance(result, dict):
                raise LLMResponseError("LLM response must be a JSON object")
            usage = result.get("usage")
            if not isinstance(usage, dict):
                raise LLMResponseError("LLM response is missing usage data")
            return LLMCompletion(result, selected_model, actual_cost, attempt)

        raise RuntimeError("LLM retry loop exited without a response")
