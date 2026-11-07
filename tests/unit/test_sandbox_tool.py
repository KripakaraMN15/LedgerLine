from datetime import UTC, datetime

import httpx
import pytest

from vendorops.llm.client import LLMClient
from vendorops.models import Budget, CodeArtifact, TaskState, TaskStatus
from vendorops.sandbox import SandboxExecutor, register_sandbox_tool
from vendorops.tools import ToolRegistry, ToolRouter


class StubSandboxExecutor(SandboxExecutor):
    async def execute(
        self,
        source: str,
        *,
        timeout_seconds: float | None = None,
    ) -> CodeArtifact:
        return CodeArtifact(source=source, stdout="executed", exit_code=0)


def test_sandbox_executor_registers_in_router_registry() -> None:
    registry = ToolRegistry()
    register_sandbox_tool(registry, StubSandboxExecutor())

    assert registry.get("execute_python") is not None
    schema = registry.openai_tools()[0]["function"]
    assert schema["name"] == "execute_python"
    assert schema["parameters"]["required"] == ["code"]
    assert schema["parameters"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_registered_sandbox_tool_dispatches_through_tool_router() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "sandbox-call",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": '{"code":"print(1 + 1)"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
        base_url="http://mock-llm",
    )
    llm = LLMClient("http://mock-llm", client=client)
    registry = ToolRegistry()
    register_sandbox_tool(registry, StubSandboxExecutor())
    router = ToolRouter(registry, llm)
    state = TaskState(
        task_id="sandbox-tool-test",
        status=TaskStatus.RUNNING,
        goal="Run a calculation",
        budget=Budget(),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = await router.route(state, [{"role": "user", "content": state.goal}])
    await client.aclose()

    assert result.ok
    assert result.output is not None
    assert result.output["stdout"] == "executed"
