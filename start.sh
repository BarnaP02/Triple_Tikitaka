#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# ── Prerequisites ─────────────────────────────────────────────────────────────

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in the values."
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker and try again."
    exit 1
fi

# ── Load env ──────────────────────────────────────────────────────────────────

set -a
source .env
set +a

APP_PORT=${APP_PORT:-8000}

# ── Sync dependencies ─────────────────────────────────────────────────────────

echo "Syncing dependencies..."
poetry install --no-interaction --quiet --only main,dev

# ── Start services ────────────────────────────────────────────────────────────

echo "Starting services..."
docker compose up -d --build

# ── Wait for app ──────────────────────────────────────────────────────────────

echo -n "Waiting for app"
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${APP_PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Ready → http://localhost:${APP_PORT}"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "App did not become healthy in 60 s. Check logs with: docker compose logs app"
exit 1
