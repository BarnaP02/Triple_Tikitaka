FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --only main

COPY app/ .

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
