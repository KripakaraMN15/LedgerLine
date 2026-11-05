# VendorOps Agent Platform: SPEC

Status: v0.1 (Phase 1). This file is the source of truth. If code and spec disagree, fix one of them in the same commit.

## 1. Goal

An autonomous agent platform that executes vendor-operations tasks (onboard vendor, reconcile invoices, schedule payment) with concurrency, cost controls, typed state, human approval, and full observability. Built on free tools, structured to production standards.

**Non-goals (for now):** real payments, real vendor integrations, multi-tenant auth, UI beyond a minimal status page.

## 2. Repo layout

```
vendorops/
├── SPEC.md
├── README.md
├── docker-compose.yml          # postgres(pgvector), app, mock-llm, mock-vendor, phoenix
├── pyproject.toml              # uv or poetry; ruff, mypy --strict, pytest
├── .github/workflows/
│   ├── ci.yml                  # lint, mypy, unit + integration tests
│   └── trajectory-gate.yml     # Phase 5: replay golden trajectories
├── src/vendorops/
│   ├── api/                    # FastAPI routes, auth, health checks
│   ├── orchestrator/           # asyncio task runner, lifecycle, concurrency limits
│   ├── llm/                    # provider client, retries, kill-switch middleware
│   ├── context/                # token budget assembler
│   ├── tools/                  # registry, Pydantic schemas, router, tool impls
│   ├── agents/                 # react loop, researcher, coder, reviewer
│   ├── memory/                 # scratchpad, episodic, semantic, eviction
│   ├── security/               # sanitization proxy, slack signature verification
│   ├── sandbox/                # code executor (docker --network none / E2B)
│   ├── fallback/               # browser fallback (Playwright)
│   ├── hitl/                   # approval gate, durable state, slack webhook
│   ├── telemetry/              # OpenTelemetry setup, span helpers
│   └── models.py               # shared typed state (section 4)
├── mocks/
│   ├── mock_llm/               # scripted responses, latency + failure injection
│   ├── mock_vendor_api/        # REST API with fault flags
│   └── mock_vendor_portal/     # plain HTML legacy portal (Phase 5)
├── evals/
│   ├── golden_trajectories/    # expected tool-call DAGs as JSON
│   └── replay.py
├── tests/{unit,integration,load}/
├── docs/{runbook.md,teardowns/}
└── migrations/
```

## 3. Task lifecycle

```
CREATED -> RUNNING -> AWAITING_APPROVAL -> RUNNING -> DONE
              |              |                |
              +--------------+----------------+--> FAILED
              +-------------------------------+--> KILLED   (cost/loop limit)
              +-------------------------------+--> CANCELLED (user request)
```

Rules:
- Only the transitions above are legal. Illegal transitions raise `InvalidTransition` and are logged.
- `DONE`, `FAILED`, `KILLED`, `CANCELLED` are terminal.
- Every transition writes a row to `task_events` (task_id, from, to, reason, timestamp) in the same DB transaction as the state change.
- Task creation is idempotent: clients send an `idempotency_key`; a duplicate key returns the existing task.
- On process restart, every `CREATED` task is enqueued (it may have been inserted just before a crash); stale `RUNNING` tasks are resumed from their last checkpoint or marked `FAILED` with reason `orphaned`.
- Phase 1 runs a single app instance. Stale-task recovery uses a five-minute threshold and checkpoints refresh `updated_at`; multiple replicas require a recovery lease or `FOR UPDATE SKIP LOCKED`.

## 4. Typed state (src/vendorops/models.py)

Agents exchange these objects, never raw strings.

```python
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
    error: str | None = None          # validation or runtime error text, fed back on retry
    latency_ms: int


class ResearchFindings(BaseModel):    # Researcher -> Coder
    vendor_id: str
    facts: list[str]
    sources: list[str]
    untrusted: bool = True            # set False only by the sanitization proxy


class CodeArtifact(BaseModel):        # Coder -> Reviewer
    language: str = "python"
    source: str
    stdout: str | None = None
    exit_code: int | None = None


class ReviewVerdict(BaseModel):       # Reviewer -> orchestrator
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
    updated_at: datetime
```

## 5. Trace schema (OpenTelemetry)

One trace per task; `trace_id` is stored on the task row. Span names and required attributes:
The OTLP exporter is configured with `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or
`OTEL_EXPORTER_OTLP_ENDPOINT`), `OTEL_EXPORTER_OTLP_PROTOCOL`, and the standard OTLP headers
environment variables. Without an endpoint, telemetry uses the SDK's no-export setup; tests may
inject an in-memory exporter. Goals are represented by hashes and sizes, tool arguments by hashes,
and vendor output by sizes only. Exporting is batched and network failures never block task calls;
graceful runner shutdown performs a bounded flush. On task resume, each process run starts a new
trace linked to the original stored trace ID, which is retained in the task row. Phase 2 wires
`context.assemble` into every agent LLM request, including tool-validation retries.

| Span | Required attributes |
|---|---|
| `task` | task_id, goal, final_status, total_usd, total_steps |
| `llm.call` | model, prompt_tokens, completion_tokens, usd, latency_ms, attempt |
| `tool.call` | tool, args_hash, ok, latency_ms, attempt, error (if any) |
| `state.transition` | from, to, reason |
| `context.assemble` | budget_tokens, system_tokens, docs_tokens, tools_tokens, dropped_docs |
| `killswitch.check` | projected_usd, limit_usd, tripped |

Never put secrets or raw PII in span attributes; log hashes instead.

## 6. Cost and loop control

- `Budget` is attached to every task. Before each LLM call, middleware computes `projected = spent_usd + estimated_cost(next_call)`.
- If `projected > max_usd` or `steps_taken >= max_steps`: stop the loop, set status `KILLED`, record reason. Optionally retry once on the cheaper fallback model if `fallback_on_budget` is true and the cheaper model's projected cost fits.
- Identical tool call (same tool + args hash) repeated 3 times in a row counts as a loop and triggers `KILLED` with reason `loop_detected`.
- Cost is computed from token counts times a configurable price table (`config/prices.yaml`); the mock LLM reports realistic token counts.

## 7. Tool contract

- Every tool is registered with a Pydantic input model and output model.
- The router validates LLM-produced args against the input model. On `ValidationError`, it retries the LLM call with the error text appended, up to `max_tool_retries` (default 2). After that, the call returns `ToolResult(ok=False)`.
- After a tool execution 404, the agent feeds a bounded, sanitized error summary back to the model and allows up to three repair attempts after the original call. Every repair LLM request uses `LLMClient` and counts against the task budget and step limit. Timeouts and transient 5xx/429 errors are retried by the external client with backoff and do not trigger LLM repair; authentication failures and other 4xx responses fail without repair. Repair calls retain the original loop detector and increment the `ToolCall.attempt` value.
- Tools never raise into the agent loop. All failures become `ToolResult(ok=False, error=...)`.
- Every external call has a timeout, and retries use exponential backoff with jitter.
- Tools that move money (even mock money) must require an approved `approval_id`; the tool itself checks it, not just the agent.
- Tool execution is at-least-once across crashes: side-effecting tools must derive a stable idempotency key from `ToolCall.id` and pass it to the external service. The current vendor tools are read-only.
- Checkpoints rewrite the full messages and history JSONB state; this is acceptable for Phase 1, but payload growth must be bounded as the system scales.
- Checkpointed messages and history can contain untrusted vendor payloads. Treat them as untrusted input and sanitize them before they are used as model instructions.

## 8. Mock services

**mock_llm** (OpenAI-compatible `/v1/chat/completions`):
- Scripted responses selected by a scenario header (`X-Scenario: bad_params_then_ok`).
- Configurable latency distribution and token counts.
- Failure modes: HTTP 429, HTTP 500, timeout, malformed JSON, schema-violating args.

**mock_vendor_api**:
- Endpoints: `GET /vendors/{id}`, `GET /vendors/{id}/invoices`, `POST /payments`.
- Fault flags via env or admin endpoint: `FAIL_404`, `FAIL_TIMEOUT`, `RETURN_INJECTION` (a vendor note containing a prompt injection).

## 9. Phase 1 scope: thin vertical slice

In scope: FastAPI `POST /tasks` and `GET /tasks/{id}`, async orchestrator, one agent, two tools (`get_vendor`, `list_invoices`) via the Pydantic router, tracing to Phoenix, cost kill-switch, Postgres persistence, Docker Compose, CI.

Out of scope: memory tiers, multi-agent, sandbox, browser fallback, Slack gate, evals.

### Acceptance tests (write these first; they must fail before implementation)

1. **Happy path:** POST a task, poll until `DONE`, and verify `task_events` contains `created -> running -> done`.
2. **Concurrency:** 100 tasks submitted at once against mock_llm with 200ms latency all finish, and total wall time is under 10 seconds (proves non-blocking).
3. **Slow tool does not block:** with one task stuck on a 5s tool call, other tasks still complete in normal time.
4. **Bad params retry:** scenario `bad_params_then_ok` produces one validation failure, a retry with the error text in the prompt, then success on attempt 2.
5. **Retry exhaustion:** scenario `always_bad_params` yields `ToolResult(ok=False)` and the task ends `FAILED`, not hung.
6. **Kill-switch:** with `max_usd=0.01` and an expensive scenario, task ends `KILLED` with reason `budget_exceeded`.
7. **Loop detection:** scenario `repeat_same_call` ends `KILLED` with reason `loop_detected` after 3 repeats.
8. **Idempotency:** two POSTs with the same `idempotency_key` return the same `task_id` and create one task.
9. **Restart recovery:** kill the app mid-task, restart, and verify the task is resumed or marked `FAILED(orphaned)`, never stuck in `RUNNING`.
10. **Tracing:** a completed task has one trace containing `task`, at least one `llm.call`, and at least one `tool.call` span with the required attributes.
11. **Graceful shutdown:** SIGTERM lets in-flight tasks reach a checkpoint before the process exits.

Phase 1 is done when all 11 pass in CI and `docker compose up` runs the full stack from a clean clone.

## 10. Rules for AI coding assistants

1. Follow this spec. If something is ambiguous, ask or add a clarifying note to this file; do not guess silently.
2. Tests first. Do not modify a test to make it pass unless the spec changed.
3. `mypy --strict` and `ruff` must pass. No `# type: ignore` without a comment explaining why.
4. No blocking calls inside `async def` (no `requests`, `time.sleep`, sync DB drivers). Use `httpx.AsyncClient`, `asyncio.sleep`, `asyncpg` or SQLAlchemy async.
5. No secrets in code, tests, logs, or span attributes. Read config from environment.
6. Keep modules small and independently testable. No circular imports between top-level packages.
7. Anything in `security/`, `sandbox/`, or `hitl/` is reviewed by a human before merge.
8. Small commits, one concern each, with `main` always green.

## 11. Roadmap

| Phase | Adds | Done when |
|---|---|---|
| 1 | Vertical slice (this spec) | 11 acceptance tests pass |
| 2 | Context budgeter, self-healing loop, sandbox executor | Failing tool call is repaired in at most 3 tries; sandbox has no network |
| 3 | Researcher/Coder/Reviewer swarm, sanitization proxy | Planted injection in mock vendor note does not reach the Coder as an instruction |
| 4 | HITL Slack gate with durable state, 3-tier memory | Task paused 3 simulated days resumes exactly once after approval |
| 5 | Browser fallback, trajectory CI gate, shadow mode | PR blocked when tool-call sequence deviates from golden DAG |
| 6 | Runbook, load test report, 3 teardown posts | Published |
