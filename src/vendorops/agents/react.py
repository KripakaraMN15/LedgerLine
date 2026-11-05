from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from vendorops.llm.client import LLMClient
from vendorops.llm.loop import ToolLoopDetector
from vendorops.models import TaskState
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
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self._llm_client = llm_client
        self._router = router
        self._repository = repository
        self._headers = dict(headers or {})
        self._model = model
        self._max_iterations = max_iterations

    async def run(self, state: TaskState) -> TaskState:
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
            completion = await self._llm_client.complete(
                state,
                state.messages,
                model=self._model,
                headers=self._headers,
                tools=self._router.schemas(),
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
                headers=self._headers,
                model=self._model,
                initial_response=completion.response,
                loop_detector=loop_detector,
                checkpoint=lambda: self._checkpoint(state),
            )
            if not result.ok:
                await self._checkpoint(state)
                raise AgentToolError(result.error or "Tool execution failed")

        raise RuntimeError(f"Agent exceeded {self._max_iterations} iterations")

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
