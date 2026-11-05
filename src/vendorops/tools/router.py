import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import ValidationError

from vendorops.context import AssembledContext, ContextBudgetError
from vendorops.llm.client import BudgetExceeded, LLMClient
from vendorops.llm.loop import LoopDetected, ToolLoopDetector
from vendorops.models import TaskState, ToolCall, ToolResult
from vendorops.telemetry import set_span_error, tool_call_span
from vendorops.tools.registry import ToolRegistry

MAX_VALIDATION_FEEDBACK_CHARS = 500
MAX_TOOL_REPAIR_ATTEMPTS = 3
Checkpoint = Callable[[], Awaitable[None]]
PromptAssembler = Callable[
    [TaskState, list[dict[str, Any]], list[dict[str, Any]]],
    AssembledContext,
]


class ToolRouter:
    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: LLMClient,
        *,
        max_tool_retries: int = 2,
        max_tool_repair_attempts: int = MAX_TOOL_REPAIR_ATTEMPTS,
        total_timeout: float = 30.0,
    ) -> None:
        if max_tool_retries < 0:
            raise ValueError("max_tool_retries must be non-negative")
        if max_tool_repair_attempts < 0:
            raise ValueError("max_tool_repair_attempts must be non-negative")
        if total_timeout <= 0:
            raise ValueError("total_timeout must be positive")
        self._registry = registry
        self._llm_client = llm_client
        self._max_tool_retries = max_tool_retries
        self._max_tool_repair_attempts = max_tool_repair_attempts
        self._total_timeout = total_timeout

    def schemas(self) -> list[dict[str, Any]]:
        return self._registry.openai_tools()

    async def route(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        headers: dict[str, str] | None = None,
        model: str = "mock-llm",
        initial_response: dict[str, Any] | None = None,
        loop_detector: ToolLoopDetector | None = None,
        checkpoint: Checkpoint | None = None,
        prompt_assembler: PromptAssembler | None = None,
    ) -> ToolResult:
        try:
            async with asyncio.timeout(self._total_timeout):
                return await self._route(
                    state,
                    messages,
                    headers=headers,
                    model=model,
                    initial_response=initial_response,
                    loop_detector=loop_detector,
                    checkpoint=checkpoint,
                    prompt_assembler=prompt_assembler,
                )
        except TimeoutError:
            pending_call = state.pending_tool_call
            state.pending_tool_call = None
            return self._failure(
                state,
                pending_call.id if pending_call is not None else "",
                f"Tool call exceeded {self._total_timeout:g}s time limit",
                int(self._total_timeout * 1000),
            )

    async def _route(
        self,
        state: TaskState,
        messages: list[dict[str, Any]],
        *,
        headers: dict[str, str] | None,
        model: str,
        initial_response: dict[str, Any] | None,
        loop_detector: ToolLoopDetector | None,
        checkpoint: Checkpoint | None,
        prompt_assembler: PromptAssembler | None,
    ) -> ToolResult:
        try:
            prompt = messages
            schemas = self._registry.openai_tools()
        except Exception as error:
            return self._failure(state, "", str(error), 0)
        attempt = 0
        validation_retries = 0
        repair_attempts = 0
        while True:
            attempt += 1
            try:
                if initial_response is not None:
                    response = initial_response
                    initial_response = None
                else:
                    assembled = (
                        prompt_assembler(state, prompt, schemas)
                        if prompt_assembler is not None
                        else None
                    )
                    completion = await self._llm_client.complete(
                        state,
                        assembled.messages if assembled is not None else prompt,
                        model=model,
                        headers=headers,
                        tools=assembled.tools if assembled is not None else schemas,
                        estimated_prompt_tokens=(
                            assembled.report.total_tokens
                            if assembled is not None
                            else None
                        ),
                    )
                    response = completion.response
                assistant_message, call_id, tool_name, raw_arguments = self._tool_call(response)
                arguments = self._parse_arguments(raw_arguments)
            except (BudgetExceeded, ContextBudgetError, LoopDetected):
                raise
            except Exception as error:
                return self._failure(state, "", str(error), 0)

            tool = self._registry.get(tool_name)
            if tool is None:
                with tool_call_span(tool_name, arguments, attempt) as current_span:
                    current_span.set_attribute("ok", False)
                    current_span.set_attribute("latency_ms", 0)
                    current_span.set_attribute("error", "UnknownTool")
                    set_span_error(current_span, "UnknownTool")
                self._record_call(state, call_id, tool_name, arguments, attempt)
                return self._failure(state, call_id, f"Unknown tool: {tool_name}", 0)

            try:
                tool.input_model.model_validate(arguments)
            except ValidationError as error:
                with tool_call_span(tool_name, arguments, attempt) as current_span:
                    current_span.set_attribute("ok", False)
                    current_span.set_attribute("latency_ms", 0)
                    current_span.set_attribute("error", "ValidationError")
                    set_span_error(current_span, "ValidationError")
                self._record_call(state, call_id, tool_name, arguments, attempt)
                error_text = self._validation_feedback(error)
                if validation_retries >= self._max_tool_retries:
                    return self._failure(state, call_id, error_text, 0)
                validation_retries += 1
                prompt.extend(
                    [
                        assistant_message,
                        {"role": "user", "content": error_text},
                    ]
                )
                if checkpoint is not None:
                    await checkpoint()
                continue

            self._record_call(state, call_id, tool_name, arguments, attempt)
            if loop_detector is not None:
                try:
                    loop_detector.observe(tool_name, arguments)
                except LoopDetected as error:
                    with tool_call_span(tool_name, arguments, attempt) as current_span:
                        current_span.set_attribute("ok", False)
                        current_span.set_attribute("latency_ms", 0)
                        current_span.set_attribute("error", type(error).__name__)
                        set_span_error(current_span, type(error).__name__)
                    raise
            tool_call = ToolCall(
                id=call_id,
                tool=tool_name,
                args=arguments,
                attempt=attempt,
            )
            state.loop_history.append(tool_call)
            state.loop_history = state.loop_history[-2:]
            state.pending_tool_call = tool_call
            prompt.append(assistant_message)
            if checkpoint is not None:
                await checkpoint()
            result = await self.execute_pending(
                state,
                tool_call,
                messages=prompt,
                checkpoint=checkpoint,
            )
            if result.ok or not result.retryable:
                return result
            if repair_attempts >= self._max_tool_repair_attempts:
                return result
            repair_attempts += 1
            prompt.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.tool,
                    "content": json.dumps(
                        {
                            "ok": False,
                            "error": self._repair_feedback(result),
                        },
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                }
            )
            if checkpoint is not None:
                await checkpoint()

    async def execute_pending(
        self,
        state: TaskState,
        tool_call: ToolCall,
        *,
        messages: list[dict[str, Any]],
        checkpoint: Checkpoint | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        with tool_call_span(tool_call.tool, tool_call.args, tool_call.attempt) as current_span:
            try:
                async with asyncio.timeout(self._total_timeout):
                    result = await self._execute_pending(
                        state,
                        tool_call,
                        messages=messages,
                        checkpoint=checkpoint,
                    )
            except asyncio.CancelledError:
                current_span.set_attribute("cancelled", True)
                current_span.set_attribute("latency_ms", self._latency_ms(started))
                raise
            except TimeoutError:
                state.pending_tool_call = None
                result = self._failure(
                    state,
                    tool_call.id,
                    f"Tool call exceeded {self._total_timeout:g}s time limit",
                    int(self._total_timeout * 1000),
                    retryable=True,
                )
                if checkpoint is not None:
                    await checkpoint()
            current_span.set_attribute("ok", result.ok)
            current_span.set_attribute("latency_ms", self._latency_ms(started))
            if result.ok:
                current_span.set_attribute(
                    "output_size",
                    len(
                        json.dumps(
                            result.output,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                )
            elif result.error:
                current_span.set_attribute("error", "ToolExecutionError")
                set_span_error(current_span, "ToolExecutionError")
            return result

    async def _execute_pending(
        self,
        state: TaskState,
        tool_call: ToolCall,
        *,
        messages: list[dict[str, Any]],
        checkpoint: Checkpoint | None,
    ) -> ToolResult:
        tool = self._registry.get(tool_call.tool)
        if tool is None:
            state.pending_tool_call = None
            result = self._failure(
                state,
                tool_call.id,
                f"Unknown tool: {tool_call.tool}",
                0,
            )
            if checkpoint is not None:
                await checkpoint()
            return result
        try:
            validated = tool.input_model.model_validate(tool_call.args)
        except ValidationError as error:
            state.pending_tool_call = None
            result = self._failure(
                state,
                tool_call.id,
                self._validation_feedback(error),
                0,
            )
            if checkpoint is not None:
                await checkpoint()
            return result

        started = time.perf_counter()
        try:
            raw_output = await tool.handler(validated.model_dump(mode="json"))
            output = tool.output_model.model_validate(raw_output)
            output_data = output.model_dump(mode="json")
            result = ToolResult(
                call_id=tool_call.id,
                ok=True,
                output=output_data,
                latency_ms=self._latency_ms(started),
            )
        except Exception as error:
            result = ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=str(error) or "Tool execution failed",
                latency_ms=self._latency_ms(started),
                retryable=self._is_retryable_error(error),
                error_code=(
                    error.response.status_code
                    if isinstance(error, httpx.HTTPStatusError)
                    else None
                ),
            )
        state.pending_tool_call = None
        if result.ok:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.tool,
                    "content": json.dumps(
                        {"ok": True, "output": result.output},
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                }
            )
        self._record_result(state, result)
        if checkpoint is not None:
            await checkpoint()
        return result

    @staticmethod
    def _tool_call(response: dict[str, Any]) -> tuple[dict[str, Any], str, str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("LLM response does not contain a choice")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("LLM response does not contain an assistant message")
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls or not isinstance(tool_calls[0], dict):
            raise ValueError("LLM assistant message does not contain a tool call")
        call = tool_calls[0]
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError("LLM tool call does not contain a function")
        call_id = call.get("id")
        tool_name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(call_id, str) or not isinstance(tool_name, str):
            raise ValueError("LLM tool call is missing its ID or function name")
        return message, call_id, tool_name, raw_arguments

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ValueError(f"LLM returned malformed tool arguments: {error}") from error
        if not isinstance(raw_arguments, dict):
            raise ValueError("LLM tool arguments must be a JSON object")
        return raw_arguments

    @staticmethod
    def _validation_feedback(error: ValidationError) -> str:
        details = [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors(include_input=False)
        ]
        message = "Tool argument validation error: " + json.dumps(details, ensure_ascii=True)
        if len(message) > MAX_VALIDATION_FEEDBACK_CHARS:
            return message[: MAX_VALIDATION_FEEDBACK_CHARS - 3] + "..."
        return message

    @staticmethod
    def _repair_feedback(result: ToolResult) -> str:
        if result.error_code is not None:
            summary = f"HTTP {result.error_code}"
        else:
            summary = "temporary execution error"
        message = (
            "Tool execution failed with a retryable "
            f"{summary}. Correct the tool call and try again."
        )
        if len(message) > MAX_VALIDATION_FEEDBACK_CHARS:
            return message[: MAX_VALIDATION_FEEDBACK_CHARS - 3] + "..."
        return message

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            return status_code == 429 or status_code >= 500
        if isinstance(error, (httpx.TransportError, TimeoutError)):
            return True
        return True

    @staticmethod
    def _record_call(
        state: TaskState,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        attempt: int,
    ) -> None:
        state.history.append(
            ToolCall(id=call_id, tool=tool_name, args=arguments, attempt=attempt)
        )

    @staticmethod
    def _latency_ms(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))

    @staticmethod
    def _failure(
        state: TaskState,
        call_id: str,
        error: str,
        latency_ms: int,
        *,
        retryable: bool = False,
    ) -> ToolResult:
        result = ToolResult(
            call_id=call_id,
            ok=False,
            error=error or "Tool execution failed",
            latency_ms=latency_ms,
            retryable=retryable,
        )
        return ToolRouter._record_result(state, result)

    @staticmethod
    def _record_result(state: TaskState, result: ToolResult) -> ToolResult:
        state.history.append(result)
        return result
