FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
RUN pip install --no-cache-dir .
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app

EXPOSE 8000
USER app
CMD ["uvicorn", "vendorops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
