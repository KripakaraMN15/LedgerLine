# VendorOps

VendorOps is the vendor-operations agent platform described in [SPEC.md](./SPEC.md).

## Telemetry

The tracing SDK initializes from standard OpenTelemetry environment variables. Set
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to the collector's trace endpoint, for example
`http://localhost:6006/v1/traces` for the Phoenix service in Docker Compose. The app service
uses `http://phoenix:6006/v1/traces` and `http/protobuf` by default. If no endpoint is configured,
spans are collected by the SDK but not exported. Unit tests can pass an in-memory span exporter to
`configure_tracing`. Exporter request timeouts follow `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` (or
`OTEL_EXPORTER_OTLP_TIMEOUT`) in milliseconds; the default is 2000 ms. After in-flight tasks stop,
the runner performs a bounded 1500 ms flush.

## Task API

Set `VENDOROPS_API_KEY` before starting the app. The API has no default key and fails closed for
protected routes and refuses to start when it is unset. `POST /tasks` accepts a goal, an idempotency key, and optional
`max_usd` / `max_steps` overrides; server limits can be configured with
`VENDOROPS_MAX_BUDGET_USD` and `VENDOROPS_MAX_STEPS`. Task status is available at
`GET /tasks/{id}`, cancellation at `POST /tasks/{id}/cancel`, and unauthenticated health probes at
`/healthz` and `/readyz`. Request bodies are limited to 64 KiB and goals to 2000 characters. The
in-memory per-key rate limiter is single-instance only; run one app instance until it is replaced
with shared rate-limit storage. `TASK_SHUTDOWN_TIMEOUT_SECONDS` controls the graceful-drain timeout
(30 seconds by default), and `TASK_RECOVERY_STALE_SECONDS` sets the stale-task recovery window
(300 seconds by default). Mock scenario and latency headers are ignored unless
`ENABLE_TEST_HOOKS=1` is explicitly set; do not enable this in production.

For Docker Compose, export a key in the shell before starting services:

```powershell
$env:VENDOROPS_API_KEY = "set-a-local-development-key"
docker compose up --build
```

The local Compose Phoenix service disables Phoenix authentication for the trace smoke check and
publishes its UI and OTLP ports on loopback only. Do not reuse this development setting for a
network-exposed Phoenix deployment.

## Sandboxed Python tool

`register_sandbox_tool(registry, SandboxExecutor())` adds `execute_python` to a tool registry used
by `ToolRouter`. Execution requires a Docker daemon and starts a fresh, resource-limited container
with no network or host mounts. Sandbox tests skip when Docker is unavailable locally; CI sets
`REQUIRE_DOCKER=1` so missing Docker fails the job.
