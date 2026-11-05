from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from vendorops.context import (
    AssembledContext,
    ContextRatios,
    RetrievedDocument,
    TokenCounter,
    assemble,
)
from vendorops.llm.client import LLMClient
from vendorops.llm.loop import ToolLoopDetector
from vendorops.models import TaskState
from vendorops.telemetry import span
from vendorops.tools.router import ToolRouter


class CheckpointRepository(Protocol):
    async def checkpoint_task_state(
        self, task_id: str, state: dict[str, Any]
    ) -> object: ...


class AgentToolError(Exception):
    pass


class ReactAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        router: ToolRouter,
        repository: CheckpointRepository,
        *,
        headers: Mapping[str, str] | None = None,
        model: str = "mock-llm",
        max_iterations: int = 100,
        system_prompt: str = (
            "You are a vendor operations assistant. Use the available tools to "
            "inspect vendor records and invoices, then provide a concise answer."
        ),
        context_limit: int = 8192,
        reserved_output_tokens: int = 1024,
        context_ratios: ContextRatios | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if context_limit < 1 or reserved_output_tokens < 0:
            raise ValueError("Context and reserved-output limits must be positive")
        if reserved_output_tokens >= context_limit:
            raise ValueError("Reserved output tokens must be below the context limit")
        self._llm_client = llm_client
        self._router = router
        self._repository = repository
        self._headers = dict(headers or {})
        self._model = model
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._context_limit = context_limit
        self._reserved_output_tokens = reserved_output_tokens
        self._context_ratios = context_ratios or ContextRatios()
        self._token_counter = token_counter

    async def run(self, state: TaskState) -> TaskState:
        with span("agent.react", {"iteration_limit": self._max_iterations}) as current_span:
            result = await self._run(state)
            current_span.set_attribute("iterations", state.budget.steps_taken)
            current_span.set_attribute("final_answer_size", len(state.final_answer or ""))
            return result

    async def _run(self, state: TaskState) -> TaskState:
        if state.final_answer is not None:
            return state
        if not state.messages:
            state.messages.append({"role": "user", "content": state.goal})
        await self._checkpoint(state)

        loop_detector = ToolLoopDetector()
        for previous_call in state.loop_history:
            loop_detector.observe(previous_call.tool, previous_call.args)

        for _ in range(self._max_iterations):
            if state.pending_tool_call is not None:
                pending = state.pending_tool_call
                result = await self._router.execute_pending(
                    state,
                    pending,
                    messages=state.messages,
                    checkpoint=lambda: self._checkpoint(state),
                )
                if not result.ok:
                    raise AgentToolError(result.error or "Tool execution failed")
                continue

            await self._checkpoint(state)
            schemas = self._router.schemas()
            assembled = self._assemble_prompt(state, state.messages, schemas)
            completion = await self._llm_client.complete(
                state,
                assembled.messages,
                model=self._model,
                headers=self._headers_for_state(state),
                tools=assembled.tools,
                estimated_prompt_tokens=assembled.report.total_tokens,
            )
            await self._checkpoint(state)
            assistant_message = self._assistant_message(completion.response)
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None:
                content = assistant_message.get("content")
                if not isinstance(content, str):
                    raise ValueError("LLM final answer must contain text")
                state.messages.append(assistant_message)
                state.final_answer = content
                await self._checkpoint(state)
                return state

            result = await self._router.route(
                state,
                state.messages,
                headers=self._headers_for_state(state),
                model=self._model,
                initial_response=completion.response,
                loop_detector=loop_detector,
                checkpoint=lambda: self._checkpoint(state),
                prompt_assembler=self._assemble_prompt,
            )
            if not result.ok:
                await self._checkpoint(state)
                raise AgentToolError(result.error or "Tool execution failed")

        raise RuntimeError(f"Agent exceeded {self._max_iterations} iterations")

    def _assemble_prompt(
        self,
        state: TaskState,
        history: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> AssembledContext:
        documents: list[RetrievedDocument] = []
        if state.findings is not None:
            for index, fact in enumerate(state.findings.facts):
                source = (
                    state.findings.sources[index]
                    if index < len(state.findings.sources)
                    else None
                )
                content = f"{fact}\nSource: {source}" if source else fact
                documents.append(
                    RetrievedDocument(
                        content=content,
                        score=1.0,
                        priority=0,
                        document_id=f"finding-{index}",
                    )
                )
        return assemble(
            system_prompt=self._system_prompt,
            tool_schemas=tool_schemas,
            documents=documents,
            history=history,
            context_limit=self._context_limit,
            reserved_output_tokens=self._reserved_output_tokens,
            ratios=self._context_ratios,
            token_counter=self._token_counter,
        )

    def _headers_for_state(self, state: TaskState) -> dict[str, str]:
        headers = dict(self._headers)
        if state.llm_scenario is not None:
            headers["X-Scenario"] = state.llm_scenario
        if state.llm_latency_ms is not None:
            headers["X-Latency-Ms"] = str(state.llm_latency_ms)
        return headers

    async def _checkpoint(self, state: TaskState) -> None:
        state.updated_at = datetime.now(UTC)
        await self._repository.checkpoint_task_state(
            state.task_id,
            state.model_dump(mode="json"),
        )

    @staticmethod
    def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("LLM response does not contain a choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("LLM response does not contain an assistant message")
        return message
