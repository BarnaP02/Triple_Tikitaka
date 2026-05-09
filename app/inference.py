from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import csv
import os
import tempfile
from pathlib import Path

from google import genai
from google.genai import types
import timm
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE      = 32_000
CHUNK_DURATION   = 5
CHUNK_SAMPLES    = SAMPLE_RATE * CHUNK_DURATION
N_FFT            = 1024
HOP_LENGTH       = 320
N_MELS           = 128
F_MIN            = 50
F_MAX            = 16_000
DETECT_THRESHOLD = 0.1

_default_checkpoint = Path(__file__).parent.parent / "models" / "best_model.pth"
CHECKPOINT_PATH = os.getenv("MODEL_PATH", str(_default_checkpoint))

_default_train_csv = Path(__file__).parent.parent / "dataset" / "birdclef-2025" / "train.csv"
TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH", str(_default_train_csv))

# code -> {"common_name": ..., "scientific_name": ...}
_species_meta: dict[str, dict[str, str]] = {}
if Path(TRAIN_CSV_PATH).exists():
    with open(TRAIN_CSV_PATH, newline="") as _f:
        for _row in csv.DictReader(_f):
            _code = _row["primary_label"]
            if _code not in _species_meta:
                _species_meta[_code] = {
                    "common_name": _row["common_name"],
                    "scientific_name": _row["scientific_name"],
                }

app = FastAPI()

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
def index():
    return FileResponse(_static / "index.html")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
).to(device)
power_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)

checkpoint  = torch.load(CHECKPOINT_PATH, map_location="cpu")
label2idx   = checkpoint["label2idx"]
idx2label   = {v: k for k, v in label2idx.items()}
num_classes = len(label2idx)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes, in_chans=3)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _gemini_client


def _load_waveform(file_bytes: bytes, suffix: str) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def _run_inference(waveform: torch.Tensor) -> dict[str, float]:
    total_samples = waveform.shape[1]
    remainder = total_samples % CHUNK_SAMPLES
    if remainder:
        waveform = torch.nn.functional.pad(waveform, (0, CHUNK_SAMPLES - remainder))

    chunks = waveform.squeeze(0).reshape(-1, CHUNK_SAMPLES)  # (n_chunks, 160000)

    with torch.no_grad():
        chunks = chunks.unsqueeze(1).to(device)              # (n_chunks, 1, 160000)
        mel    = mel_transform(chunks)
        mel_db = power_to_db(mel)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        mel_db = mel_db.repeat(1, 3, 1, 1)                   # (n_chunks, 3, 128, 501)
        probs  = torch.sigmoid(model(mel_db))                 # (n_chunks, num_classes)
        max_probs = probs.max(dim=0).values.cpu().numpy()     # (num_classes,)

    return {idx2label[i]: round(float(max_probs[i]), 4) for i in range(num_classes)}


def _display_name(code: str) -> str:
    meta = _species_meta.get(code)
    if meta:
        return f"{meta['common_name']} ({meta['scientific_name']})"
    return code


def _research_species(detected: dict[str, float], user_context: str) -> str:
    species_list = ", ".join(
        f"{_display_name(code)} — {prob:.2%}"
        for code, prob in sorted(detected.items(), key=lambda x: -x[1])
    )
    prompt = (
        f"I recorded bird calls in Colombia. My ML model detected the following species "
        f"(species: confidence): {species_list}.\n\n"
        f"User context: {user_context}\n\n"
        f"Research these species and give your opinion on whether they can plausibly coexist "
        f"in Colombia. Consider their typical habitats, ranges, and whether the combination "
        f"makes ecological sense. Be concise and specific."
    )

    response = _get_gemini_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )
    return response.text


@app.get("/health")
def health():
    return {"status": "ok"}


def _enrich(predictions: dict[str, float]) -> list[dict]:
    return [
        {
            "code": code,
            "common_name": _species_meta.get(code, {}).get("common_name", code),
            "scientific_name": _species_meta.get(code, {}).get("scientific_name", ""),
            "probability": prob,
        }
        for code, prob in sorted(predictions.items(), key=lambda x: -x[1])
    ]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".ogg"
    file_bytes = await file.read()
    waveform = _load_waveform(file_bytes, suffix)
    predictions = _run_inference(waveform)
    return {"predictions": _enrich(predictions)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), context: str = Form("")):
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".ogg"
    file_bytes = await file.read()
    waveform = _load_waveform(file_bytes, suffix)
    predictions = _run_inference(waveform)

    detected = {code: prob for code, prob in predictions.items() if prob >= DETECT_THRESHOLD}
    analysis = _research_species(detected, context) if detected else "No species detected above threshold."

    return {
        "predictions": _enrich(predictions),
        "detected_above_threshold": _enrich(detected),
        "analysis": analysis,
    }
