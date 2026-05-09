# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triple Tikitaka is an animal sound classification system for the [BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025) Kaggle competition. Despite the name, the dataset contains birds, amphibians, mammals, and insects — all recorded in Colombia. The system classifies species from 5-second audio chunks using an EfficientNet B0 model, then uses Gemini with Google Search grounding to assess whether the detections are plausible given the user's context.

## Quickstart

```bash
./start.sh
```

Checks `.env`, syncs deps, builds and starts all Docker services, waits for health, then prints the URL. Requires Docker to be running.

## Environment Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment (fish shell)
source .venv/bin/activate.fish

# Pull dataset and models from DVC remote
dvc pull
```

Copy `.env.example` to `.env` and fill in `GRAFANA_ADMIN_PASSWORD`, `APP_PORT`, and `GOOGLE_API_KEY` before starting services. Python version is pinned to 3.12.13 via `.python-version`.

## Running Notebooks

Notebooks in `notebooks/` define the full ML pipeline and must be run in order:

1. `01_preprocessing.ipynb` — builds chunk manifests (`dataset/train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) from the raw BirdCLEF audio and `train.csv`
2. `02_training.ipynb` — trains EfficientNet B0 with MLflow logging; supports checkpoint resume
3. `03_evaluation.ipynb` — loads `models/best_model.pth` and computes top-1/3/5 accuracy

MLflow must be running before training (`docker compose up mlflow`). Checkpoints save to `models/`.

## Services

```
MLflow UI  : http://localhost:5000   (required during training)
App / UI   : http://localhost:${APP_PORT}
Grafana    : http://localhost:3000
Prometheus : http://localhost:9090
```

## Architecture

### Data Flow

Raw `.ogg` files → chunked into 5-second clips → `*_manifest.csv` (filename + `start_sample` + `primary_label`) → `BirdCLEFDataset` loads chunks on-the-fly → mel spectrogram computed **on GPU** at training time (not pre-computed).

Audio constants used throughout (must be consistent across all stages):
- `SAMPLE_RATE=32000`, `CHUNK_DURATION=5`, `CHUNK_SAMPLES=160000`
- `N_FFT=1024`, `HOP_LENGTH=320`, `N_MELS=128`, `F_MIN=50`, `F_MAX=16000`

### Model

`timm.create_model("efficientnet_b0", pretrained=True, num_classes=N)` where N = number of species in the training split (~206). The 3-channel input is satisfied by repeating the single-channel mel spectrogram 3×.

Loss: `BCEWithLogitsLoss` (multi-label formulation even though labels are single-class). Optimizer: `AdamW` + `CosineAnnealingLR`. Early stopping patience=7.

### Checkpoint Format

Saved as `models/best_model.pth` and `models/last_checkpoint.pth`:
```python
{
    "epoch": int,
    "model_state": state_dict,
    "optimizer_state": ...,
    "scheduler_state": ...,
    "val_loss": float,
    "label2idx": dict,       # must be preserved for inference
    "mlflow_run_id": str,    # for resuming the same MLflow run
}
```

`label2idx` is stored in the checkpoint so inference does not need the training manifest.

### Inference App (`app/inference.py`)

FastAPI server with three endpoints:

- `GET /health` — returns `{"status": "ok"}`
- `POST /predict` — accepts an audio file, returns enriched predictions (all species, sorted by probability)
- `POST /analyze` — accepts an audio file + optional `context` form field; filters species above 0.5 confidence, calls Gemini 2.5 Flash with Google Search to assess plausibility, returns predictions + analysis

Both `/predict` and `/analyze` return predictions as a list of objects:
```json
[{"code": "compau", "common_name": "Common Pauraque", "scientific_name": "Nyctidromus albicollis", "probability": 1.0}]
```

Species names are resolved at startup from `dataset/birdclef-2025/train.csv` (or `TRAIN_CSV_PATH` env var in Docker).

The app also serves a single-page web UI at `/` (`app/static/index.html`) with drag-and-drop upload, optional context field, and two modes: predict-only and predict+analyze (renders Gemini markdown response).

#### Key env vars for the app
| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/best_model.pth` | Path to checkpoint |
| `TRAIN_CSV_PATH` | `dataset/birdclef-2025/train.csv` | Species name lookup |
| `GOOGLE_API_KEY` | — | Gemini API key (required for `/analyze`) |

### Docker

```bash
docker compose up -d     # start everything
docker compose logs app  # tail app logs
```

The app container mounts `./models`, `./dataset/birdclef-2025/train.csv`, `./app`, and `./configs` at runtime — the image itself contains no model weights or data. `MODEL_PATH`, `TRAIN_CSV_PATH`, and `GOOGLE_API_KEY` are forwarded from `.env` via `docker-compose.yml`.

### Data / Model Versioning

`dataset/` and `models/` are tracked by DVC (see `dataset.dvc` and `models.dvc`). Do not commit these directories directly to git. After updating models, run `dvc add models` and commit the `.dvc` file.
