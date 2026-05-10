# Development Process

This document describes the development timeline of the agentic system built on top of the BirdCLEF 2025 ML model. The ML model itself (training, evaluation) was built before this phase began. Everything described here was developed interactively in Claude Code.

---

## Starting point

The repository already had:
- A trained EfficientNet B0 model saved to `models/best_model.pth`
- Preprocessing, training, and evaluation notebooks
- Docker Compose infrastructure for MLflow, Prometheus, and Grafana

The goal was to build an agentic system on top of the model: accept a recording, classify it, then have an AI agent research whether the predictions make sense given the user's context.

---

## Step 1 — Project documentation (`/init`)

The session started with the `/init` command, which prompted Claude Code to read the codebase and generate `CLAUDE.md` — a file that gives future Claude instances the context they need to work in this repo (commands, architecture, data flow, checkpoint format, etc.).

---

## Step 2 — Inference endpoint (`inference: add FastAPI prediction endpoint`)

The first real task was implementing `app/inference.py` as a FastAPI server. The requirements:
- Accept an audio file upload
- Split it into non-overlapping 5-second chunks (160,000 samples each), zero-padding the last one
- Run all chunks through the model in a single batched forward pass
- Return the **maximum probability across all chunks** for each species — so a species only needs to appear once anywhere in the recording to be detected

The mel spectrogram pipeline had to exactly match what was used during training:
- `MelSpectrogram(sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128, f_min=50, f_max=16000)`
- `AmplitudeToDB(stype="power", top_db=80)`
- Min-max normalisation to `[0, 1]`
- Spectrogram repeated 3× to satisfy EfficientNet's 3-channel input

The checkpoint path defaults to `models/best_model.pth` relative to the repo root, but can be overridden with `MODEL_PATH` for Docker where the path is `/models/best_model.pth`.

An `inference_explained.md` was also written to document each step of the pipeline with the relevant code snippets.

---

## Step 3 — Agentic `/analyze` endpoint (`inference: add /analyze endpoint with Gemini-powered species research`)

The `/analyze` endpoint layers an AI research agent on top of `/predict`:
1. Run ML inference
2. Filter species above a confidence threshold
3. Pass the results and user-provided context to an LLM with web search
4. Return predictions + analysis together

**Initial plan used the Claude API** (`claude-opus-4-7` with the `web_search_20250305` server-side tool). The `pause_turn` / `end_turn` loop was understood and implemented correctly — for server-side tools Claude handles the search loop internally and only signals `pause_turn` if it hits its iteration limit, at which point you re-send the messages to continue.

**Switch to Gemini** — after getting the API key set up, it turned out the Google Cloud project behind the key had zero free-tier quota for `gemini-2.0-flash`. After testing available models, `gemini-2.5-flash` had quota and worked. Gemini's Google Search grounding is fully server-side, so the client code is much simpler than the Claude loop — one `generate_content` call returns the final grounded answer.

The `google-genai` client is initialised lazily (on first `/analyze` call) so the server starts cleanly even without `GOOGLE_API_KEY` set.

---

## Step 4 — Species name lookup (`inference: translate species codes to common/scientific names`)

The model outputs BirdCLEF species codes (`compau`, `blbwre1`, etc.), which Gemini couldn't look up. `train.csv` has `primary_label → common_name, scientific_name` for all 206 species.

At startup, `inference.py` reads `train.csv` and builds a lookup dict. Both `/predict` and `/analyze` now return enriched objects:
```json
{"code": "compau", "common_name": "Common Pauraque", "scientific_name": "Nyctidromus albicollis", "probability": 1.0}
```

The Gemini prompt uses full common and scientific names, which produced much better research output — instead of saying it couldn't find the species, Gemini could correctly look it up and reason about its range and habitat.

---

## Step 5 — Web UI (`ui: add web frontend served by FastAPI`)

A single-page HTML/JS UI served directly by FastAPI at `/`. No build step or separate server — just `app/static/index.html` mounted with `StaticFiles`.

Features:
- Drag-and-drop audio file upload
- Optional context textarea
- Two buttons: **Predict only** and **Predict + Analyze**
- Species results rendered as a bar chart with common name, italic scientific name, and colour-coded confidence bars (green ≥ 50%, indigo ≥ 10%, grey below)
- Gemini analysis rendered as Markdown via `marked.js`

---

## Step 6 — Prompt refinement (`inference: improve analyze prompt and render markdown in UI`)

Several iterations on the research prompt:

- **Initial prompt** asked whether the detected species "can coexist in Colombia." Since all dataset species are Colombian, this always produced a positive answer regardless of the recording's actual origin.
- **Revised prompt** asks the agent to assess plausibility *given the user's context* — if someone records in their European bedroom, the agent should note that Colombian species are unlikely to be present there; if they record at a zoo, the agent should recognise that captive animals can be far outside their native range.
- **"Bird call classifier" → "animal sound classifier"** — the BirdCLEF 2025 dataset includes amphibians, mammals, and insects alongside birds, so the original wording was misleading.
- **Confidence threshold** raised from 0.1 → 0.3 → 0.5 as we found that lower thresholds sent too much noise to the agent.

---

## Step 7 — Docker fixes (`docker: fix image build and add .dockerignore`)

The original Dockerfile had three issues:
1. `python:3.14-slim` — Python 3.14 does not exist; corrected to `python:3.12-slim`
2. `CMD ["uvicorn", ...]` without `poetry run` — with `in-project = true`, the venv is at `.venv/` and the system `uvicorn` does not exist inside the container; fixed by disabling virtualenv creation (`poetry config virtualenvs.create false`) so packages install into the system Python directly
3. No `.dockerignore` — the build context was 31 GB because the entire `dataset/` directory was being sent to the Docker daemon; `.dockerignore` now excludes `dataset/`, `models/`, `.venv/`, `notebooks/`, and `mlruns/`

`docker compose` was also updated to forward `MODEL_PATH`, `TRAIN_CSV_PATH`, and `GOOGLE_API_KEY` from `.env`, and to mount `train.csv` read-only into the container.

---

## Step 8 — Startup script and documentation

`start.sh` was added as a single entry point: checks for `.env` and a running Docker daemon, syncs poetry dependencies, runs `docker compose up -d --build`, and polls the `/health` endpoint until the app is ready, then prints the URL. It was later moved to `scripts/start.sh`, which required fixing the `cd "$(dirname "$0")"` to `cd "$(dirname "$0")/.."` so it resolves paths relative to the repo root instead of the `scripts/` directory.

`CLAUDE.md` was updated to reflect the current architecture, and `README.md` was written to cover prerequisites, setup, and usage for anyone cloning the repo fresh.

---

## Step 9 — Prometheus metrics and Grafana dashboard (`monitoring: wire up Prometheus metrics and Grafana dashboard`)

The Prometheus and Grafana containers were already defined in `docker-compose.yml` and Prometheus was already configured to scrape the app, but the app exposed no metrics and Grafana had no datasource or dashboards configured.

Three things were added:

**App metrics** — `prometheus-fastapi-instrumentator` was added as a dependency. Two lines in `inference.py` (`Instrumentator().instrument(app).expose(app)`) register a `/metrics` endpoint that automatically tracks HTTP request counts, latency histograms, and response sizes per endpoint.

**Grafana provisioning** — Grafana supports loading datasources and dashboards from YAML/JSON files at startup via `docker/grafana/provisioning/`. Two provisioning files were added:
- `datasources/prometheus.yml` — auto-configures Prometheus as the default datasource so no manual setup is needed after `docker compose up`
- `dashboards/dashboards.yml` + `dashboards/app-dashboard.json` — a pre-built dashboard with four panels: request rate by endpoint, request latency (p50/p95/p99), error rate (4xx/5xx), and total requests per endpoint over 24 hours

The Grafana service in `docker-compose.yml` was updated to mount `./docker/grafana/provisioning` into `/etc/grafana/provisioning`.

---

## Step 10 — Per-species detection tracking (`monitoring: add per-species detection counter`)

To make the Grafana dashboard useful beyond generic HTTP metrics, a custom Prometheus counter was added to track how often each species is predicted with high confidence.

**The metric** — `species_detections_total` is a `Counter` with two labels: `species_code` and `common_name`. It is incremented for every species whose predicted probability is ≥ 70% in both `/predict` and `/analyze` calls. The threshold (70%) is separate from `DETECT_THRESHOLD` (50%), which controls which species are sent to the Gemini agent.

**The panel** — a time series panel was added to the Grafana dashboard showing `species_detections_total` as a cumulative staircase: flat between predictions, stepping up by 1 each time a species clears the threshold. The legend table is configured to show the `last` value, which equals the true total count.

**A fix along the way** — the initial panel used `increase(species_detections_total[5m])`, which Prometheus extrapolates across the full window rather than counting discrete events. This produced non-integer values (e.g. 16.8 instead of 2) that grew on their own between predictions. Switching to the raw counter value and `last` aggregation in the legend resolved both issues.

---

## Lessons learned

- **Free-tier API quotas are project-scoped, not key-scoped.** Creating a new API key in the same Google Cloud project does not fix a zero-quota situation. The fix was to get a key from Google AI Studio directly, which auto-provisions free-tier quotas, or to switch to a model (`gemini-2.5-flash`) that had quota on the existing project.

- **Prompt framing matters as much as model capability.** The first version of the research prompt produced useless answers ("yes, these Colombian species can coexist in Colombia") because the framing assumed a fixed geography. Reframing it to evaluate plausibility *given the user's context* made the agent substantially more useful.

- **Species codes are invisible to general-purpose LLMs.** `compau` means nothing to Gemini. Mapping codes to common and scientific names before passing them to the agent was a necessary step — the quality of the analysis improved significantly once Gemini could actually look the species up.

- **Build context size is easy to overlook.** Without `.dockerignore`, Docker was sending 31 GB of audio files to the daemon on every build. Adding the ignore file reduced the context to ~750 KB and made builds near-instant.

- **`increase()` in Prometheus is not a discrete event counter.** It extrapolates the rate to fill the full query window, producing non-integer and inflated values for infrequent events. For metrics where the raw cumulative count is what matters, querying the counter directly and using `last` as the legend aggregation gives exact, stable numbers.
