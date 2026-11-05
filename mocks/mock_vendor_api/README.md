# Mock vendor API

This FastAPI service provides deterministic vendor and invoice data:

- `GET /vendors/{id}`
- `GET /vendors/{id}/invoices`
- `POST /payments`

Payments require an `approval_id`; this mock acknowledges requests but does not
move money.

Fault injection can be configured at startup with the `FAIL_404`,
`FAIL_TIMEOUT`, and `RETURN_INJECTION` environment variables (set to `true`,
`1`, `yes`, or `on`), or at runtime with `POST /admin/faults`. The active
settings are available from `GET /admin/faults`. `FAIL_TIMEOUT` uses an
asynchronous delay configured by `MOCK_VENDOR_TIMEOUT_MS` (30 seconds by
default). `RETURN_INJECTION` replaces the vendor note with an adversarial
prompt-injection string. The vendor ID seeded for the scripted LLM tool
scenarios is defined in `mocks/shared_scenarios.py`. The seeded `INV-1001`
amount differs from its `expected_amount` to support reconciliation scenarios.

Run locally with `uvicorn mock_vendor_api.app:app --app-dir mocks --port 8002`
or use the repository's Docker Compose service.
