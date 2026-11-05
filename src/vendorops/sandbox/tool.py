from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vendorops.models import CodeArtifact
from vendorops.sandbox.executor import SandboxExecutor
from vendorops.tools.registry import ToolDefinition, ToolRegistry


class PythonExecutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    code: str = Field(min_length=1, max_length=100_000)


def register_sandbox_tool(registry: ToolRegistry, executor: SandboxExecutor) -> None:
    async def execute_python(arguments: dict[str, Any]) -> dict[str, Any]:
        artifact = await executor.execute(str(arguments["code"]))
        return artifact.model_dump(mode="json")

    registry.register(
        ToolDefinition(
            name="execute_python",
            description="Execute Python code in an isolated, resource-limited container.",
            input_model=PythonExecutionInput,
            output_model=CodeArtifact,
            handler=execute_python,
        )
    )
