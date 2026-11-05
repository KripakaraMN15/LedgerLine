.PHONY: test test-db

test:
	python -m pytest tests/unit tests/integration -q

test-db:
	@set -eu; \
	container="vendorops-test-db-$$$$"; \
	docker run --detach --rm --name "$$container" \
		--env POSTGRES_DB=vendorops \
		--env POSTGRES_USER=vendorops \
		--env POSTGRES_PASSWORD=vendorops \
		--publish 127.0.0.1::5432 postgres:18 >/dev/null; \
	cleanup() { docker stop "$$container" >/dev/null 2>&1 || true; }; \
	trap cleanup EXIT INT TERM; \
	port=$$(docker port "$$container" 5432/tcp | sed 's/.*://'); \
	attempt=0; \
	until docker exec "$$container" pg_isready -U vendorops -d vendorops >/dev/null 2>&1; do \
		attempt=$$((attempt + 1)); \
		if [ "$$attempt" -ge 60 ]; then \
			docker logs "$$container"; \
			exit 1; \
		fi; \
		sleep 1; \
	done; \
	TEST_DATABASE_URL="postgresql+asyncpg://vendorops:vendorops@127.0.0.1:$$port/vendorops" \
	REQUIRE_DB=1 python -m pytest tests/unit tests/integration -q
