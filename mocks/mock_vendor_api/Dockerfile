FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app
COPY --chown=app:app . .

EXPOSE 8002
USER app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
