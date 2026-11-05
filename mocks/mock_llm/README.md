# Mock LLM

This service exposes `POST /v1/chat/completions` with OpenAI-compatible chat
completion and tool-call response shapes. Select scripted behavior with the
`X-Scenario` header. The default scenario returns a `get_vendor` tool call.

Supported scenarios:

- `bad_params_then_ok`: returns invalid arguments until a request includes
  validation feedback, then returns valid arguments.
- `always_bad_params` and `schema_violating_args`: always return invalid
  `get_vendor` arguments.
- `repeat_same_call`: repeatedly returns the same tool call.
- `expensive_budget`: reports high token usage.
- `slow_tool`: waits for `MOCK_LLM_SLOW_MS` (5 seconds by default).
- `http_429`, `http_500`, `timeout`, and `malformed_json`: simulate the
  corresponding failure modes.

Latency and token counts can be configured with
`MOCK_LLM_LATENCY_MIN_MS`, `MOCK_LLM_LATENCY_MAX_MS`,
`MOCK_LLM_PROMPT_TOKENS`, and `MOCK_LLM_COMPLETION_TOKENS`. Timeout and
expensive-scenario token counts have corresponding `MOCK_LLM_TIMEOUT_MS`,
`MOCK_LLM_EXPENSIVE_PROMPT_TOKENS`, and
`MOCK_LLM_EXPENSIVE_COMPLETION_TOKENS` settings. A request can set
`X-Latency-Ms` to override the response delay for that call.

Unknown `X-Scenario` values return HTTP 400. The service uses asynchronous
delays for all latency and timeout behavior.

Run locally with `uvicorn mock_llm.app:app --app-dir mocks --port 8001` or use
the repository's Docker Compose service.
