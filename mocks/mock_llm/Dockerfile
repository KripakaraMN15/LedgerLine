FROM python:3.11-slim

WORKDIR /app
COPY mocks/mock_llm/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app
COPY --chown=app:app mocks/mock_llm/app.py ./mocks/mock_llm/app.py
COPY --chown=app:app mocks/shared_scenarios.py ./mocks/shared_scenarios.py

EXPOSE 8001
USER app
CMD ["uvicorn", "mocks.mock_llm.app:app", "--host", "0.0.0.0", "--port", "8001"]
