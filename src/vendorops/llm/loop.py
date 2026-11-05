import hashlib
import json
from collections.abc import Mapping
from typing import Any


class LoopDetected(Exception):
    def __init__(self, tool: str, args_hash: str, repeats: int) -> None:
        self.tool = tool
        self.args_hash = args_hash
        self.repeats = repeats
        super().__init__(f"Tool call {tool!r} repeated {repeats} times consecutively")


class ToolLoopDetector:
    def __init__(self, repeat_limit: int = 3) -> None:
        if repeat_limit < 1:
            raise ValueError("repeat_limit must be at least 1")
        self._repeat_limit = repeat_limit
        self._previous: tuple[str, str] | None = None
        self._repeats = 0

    @staticmethod
    def _args_hash(args: Mapping[str, Any]) -> str:
        serialized = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def observe(self, tool: str, args: Mapping[str, Any]) -> str:
        call = (tool, self._args_hash(args))
        if call == self._previous:
            self._repeats += 1
        else:
            self._previous = call
            self._repeats = 1
        if self._repeats >= self._repeat_limit:
            raise LoopDetected(tool, call[1], self._repeats)
        return call[1]

    def reset(self) -> None:
        self._previous = None
        self._repeats = 0
