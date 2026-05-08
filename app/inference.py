from fastapi import FastAPI, File, UploadFile
import os
import tempfile
from pathlib import Path

import timm
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE    = 32_000
CHUNK_DURATION = 5
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_DURATION
N_FFT          = 1024
HOP_LENGTH     = 320
N_MELS         = 128
F_MIN          = 50
F_MAX          = 16_000

_default_checkpoint = Path(__file__).parent.parent / "models" / "best_model.pth"
CHECKPOINT_PATH = os.getenv("MODEL_PATH", str(_default_checkpoint))

app = FastAPI()

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad the last chunk if the audio doesn't divide evenly into 5s
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

    return {
        "predictions": {
            idx2label[i]: round(float(max_probs[i]), 4)
            for i in range(num_classes)
        }
    }
