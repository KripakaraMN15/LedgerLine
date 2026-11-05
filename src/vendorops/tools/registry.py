from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model
        self.handler = handler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self._tools.values()]
