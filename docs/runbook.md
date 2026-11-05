# Runbook

## PostgreSQL lifecycle tests

Start the Compose PostgreSQL service, then run the integration tests from the
repository root:

```shell
docker compose up -d postgres
```

Set `TEST_DATABASE_URL` to
`postgresql+asyncpg://vendorops:vendorops@localhost:5432/vendorops`, install the
development dependencies, and run:

```shell
python -m pytest tests/integration/test_lifecycle_postgres.py -q
```

The test fixture applies the SQL files in `migrations/` before running tests.
When finished, stop the database with `docker compose stop postgres`.
