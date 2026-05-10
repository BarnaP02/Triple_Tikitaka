# Triple Tikitaka

Animal sound classifier for the BirdCLEF 2025 dataset. Upload a recording, get species predictions, and optionally have an AI agent research whether those species could plausibly be present given your context.

## Prerequisites

- Python 3.12 (via [pyenv](https://github.com/pyenv/pyenv))
- [Poetry](https://python-poetry.org/)
- Docker + Docker Compose

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/BarnaP02/Triple_Tikitaka.git
cd Triple_Tikitaka
poetry install
```

**2. Pull the model and species list from DVC**

```bash
dvc pull models/best_model.pth
dvc pull dataset/birdclef-2025/train.csv
```

The full training dataset is tens of GB and not needed to run the app. These two commands pull only what is required: the trained model and the species name lookup.

**3. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Description |
|---|---|
| `APP_PORT` | Port the app will be available on (e.g. `8000`) |
| `GRAFANA_ADMIN_PASSWORD` | Any password for the Grafana dashboard |
| `GOOGLE_API_KEY` | Gemini API key — required for the `/analyze` endpoint ([get one here](https://aistudio.google.com/apikey)) |

## Running

```bash
./scripts/start.sh
```

This syncs dependencies, builds the Docker image, starts all services, and prints the URL once the app is ready. Open it in your browser and you're good to go.

## Services

Once running, the following services are available (default ports, configurable in `.env`):

| Service | URL | Description |
|---|---|---|
| App | http://localhost:8000 | Web UI and REST API |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics scraper |
| MLflow | http://localhost:5000 | Training experiment tracking |

## Example audio files

The `examples/` directory contains a small set of recordings you can use right after setup:

- `examples/labelled/` — 5 recordings from the training set, renamed to the species' common name (e.g. `Saffron_Finch.ogg`). Upload one and the model should predict that species with high confidence.
- `examples/unlabelled/` — 5 short soundscape recordings without ground-truth labels, good for exploring what the model picks up in a real field recording.

## Usage

The web UI has two modes:

- **Predict only** — runs the ML model and shows the top detected species with confidence bars.
- **Predict + Analyze** — also passes the detections to a Gemini agent with Google Search, which assesses whether the species are plausibly present given your context (location, habitat, situation). Add context in the text field for better results.

## API

The app also exposes a REST API directly:

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@recording.ogg"

# Analyze
curl -X POST http://localhost:8000/analyze \
  -F "file=@recording.ogg" \
  -F "context=Colombian Andes, ~2000m, humid montane forest"
```
