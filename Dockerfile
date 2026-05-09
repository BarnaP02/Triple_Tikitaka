FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-root --no-interaction

COPY app/ .

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]