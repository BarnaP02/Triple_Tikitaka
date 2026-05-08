# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triple Tikitaka is a bird call classification system for the [BirdCLEF 2025](https://www.kaggle.com/competitions/birdclef-2025) Kaggle competition. It classifies bird species from 5-second audio chunks using an EfficientNet B0 model trained on mel spectrograms.

## Environment Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment (fish shell)
source .venv/bin/activate.fish

# Pull dataset and models from DVC remote
dvc pull
```

## Running Notebooks

Notebooks in `notebooks/` define the full ML pipeline and must be run in order:

1. `01_preprocessing.ipynb` — builds chunk manifests (`dataset/train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) from the raw BirdCLEF audio and `train.csv`
2. `02_training.ipynb` — trains EfficientNet B0 with MLflow logging; supports checkpoint resume
3. `03_evaluation.ipynb` — loads `models/best_model.pth` and computes top-1/3/5 accuracy

MLflow must be running before training (see below). Checkpoints save to `models/`.

## Services

```bash
# Start all services (app, MLflow, Prometheus, Grafana)
docker compose up

# MLflow UI (required during training): http://localhost:5000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# App (inference API): http://localhost:${APP_PORT}
```

Copy `.env.example` to `.env` and fill in `GRAFANA_ADMIN_PASSWORD` and `APP_PORT` before starting services.

## Architecture

### Data Flow

Raw `.ogg` files → chunked into 5-second clips → `*_manifest.csv` (filename + `start_sample` + `primary_label`) → `BirdCLEFDataset` loads chunks on-the-fly → mel spectrogram computed **on GPU** at training time (not pre-computed).

Audio constants used throughout (must be consistent across all stages):
- `SAMPLE_RATE=32000`, `CHUNK_DURATION=5`, `CHUNK_SAMPLES=160000`
- `N_FFT=1024`, `HOP_LENGTH=320`, `N_MELS=128`, `F_MIN=50`, `F_MAX=16000`

### Model

`timm.create_model("efficientnet_b0", pretrained=True, num_classes=N)` where N = number of species in the training split (currently ~206). The 3-channel input is satisfied by repeating the single-channel mel spectrogram 3×.

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
    "label2idx": dict,          # must be preserved for inference
    "mlflow_run_id": str,       # for resuming the same MLflow run
}
```

`label2idx` is stored in the checkpoint so inference does not need the training manifest.

### Inference App

`app/` contains a FastAPI server (`inference.py`) that loads `models/best_model.pth` and serves predictions. The Docker container mounts `./app`, `./models`, and `./configs` at runtime.

### Data / Model Versioning

`dataset/` and `models/` are tracked by DVC (see `dataset.dvc` and `models.dvc`). Do not commit these directories directly to git. After updating models, run `dvc add models` and commit the `.dvc` file.
