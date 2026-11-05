from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from vendorops.llm.cost import PriceTable, load_price_table
from vendorops.models import TaskState


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


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, separators=(",", ":"), ensure_ascii=True)
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

    async def complete(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        model: str = "mock-llm",
        fallback_model: str | None = None,
        fallback_on_budget: bool = False,
        headers: Mapping[str, str] | None = None,
        max_tokens: int = 128,
        estimated_prompt_tokens: int | None = None,
        estimated_completion_tokens: int | None = None,
    ) -> LLMCompletion:
        budget = state.budget
        if budget.steps_taken >= budget.max_steps:
            raise BudgetExceeded(budget.spent_usd, budget.max_usd, "step_limit")

        prompt_tokens = (
            estimated_prompt_tokens
            if estimated_prompt_tokens is not None
            else estimate_prompt_tokens(messages)
        )
        completion_tokens = (
            estimated_completion_tokens
            if estimated_completion_tokens is not None
            else max_tokens
        )
        if prompt_tokens < 0 or completion_tokens < 0 or max_tokens < 0:
            raise ValueError("Token estimates and max_tokens must be non-negative")

        projected = budget.spent_usd + self._prices.estimate(
            model, prompt_tokens, completion_tokens
        )
        selected_model = model
        if projected > budget.max_usd:
            fallback_cost = (
                self._prices.estimate(fallback_model, prompt_tokens, completion_tokens)
                if fallback_on_budget and fallback_model is not None
                else math.inf
            )
            fallback_projected = budget.spent_usd + fallback_cost
            if (
                fallback_model is not None
                and fallback_cost < projected - budget.spent_usd
                and fallback_projected <= budget.max_usd
            ):
                selected_model = fallback_model
            else:
                raise BudgetExceeded(projected, budget.max_usd, "budget_exceeded")

        request_headers = dict(headers or {})
        if self._api_key is not None:
            request_headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, self._max_attempts + 1):
            request_headers["X-Mock-Attempt"] = str(attempt)
            try:
                response = await self._client.post(
                    self._url,
                    json=payload,
                    headers=request_headers,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as error:
                raise LLMAPIError(0, attempt, "LLM request timed out") from error
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < self._max_attempts:
                    ceiling = min(
                        self._retry_max_delay,
                        self._retry_base_delay * (2 ** (attempt - 1)),
                    )
                    await asyncio.sleep(random.uniform(0, ceiling))
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
            try:
                result: Any = response.json()
            except json.JSONDecodeError as error:
                raise LLMResponseError("LLM returned malformed JSON") from error
            if not isinstance(result, dict):
                raise LLMResponseError("LLM response must be a JSON object")
            usage = result.get("usage")
            if not isinstance(usage, dict):
                raise LLMResponseError("LLM response is missing usage data")
            actual_cost = self._prices.calculate(selected_model, usage)
            budget.spent_usd += actual_cost
            budget.steps_taken += 1
            return LLMCompletion(result, selected_model, actual_cost, attempt)

        raise RuntimeError("LLM retry loop exited without a response")
