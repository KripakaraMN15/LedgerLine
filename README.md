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
