from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    FAILED = "failed"
    KILLED = "killed"
    CANCELLED = "cancelled"


class Budget(BaseModel):
    max_usd: float = 0.50
    max_steps: int = 20
    spent_usd: float = 0.0
    steps_taken: int = 0


class ToolCall(BaseModel):
    id: str
    tool: str
    args: dict[str, Any]
    attempt: int = 1


class ToolResult(BaseModel):
    call_id: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int


class ResearchFindings(BaseModel):
    vendor_id: str
    facts: list[str]
    sources: list[str]
    untrusted: bool = True


class CodeArtifact(BaseModel):
    language: str = "python"
    source: str
    stdout: str | None = None
    exit_code: int | None = None


class ReviewVerdict(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)


class TaskState(BaseModel):
    task_id: str
    status: TaskStatus
    goal: str
    budget: Budget
    findings: ResearchFindings | None = None
    artifact: CodeArtifact | None = None
    verdict: ReviewVerdict | None = None
    history: list[ToolCall | ToolResult] = Field(default_factory=list)
    loop_history: list[ToolCall] = Field(default_factory=list)
    pending_tool_call: ToolCall | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    llm_scenario: str | None = None
    llm_latency_ms: int | None = None
    updated_at: datetime
