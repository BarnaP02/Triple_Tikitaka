# How inference.py works

The file is a FastAPI server with two endpoints. It loads the trained model once at startup and keeps it in memory for all subsequent requests.

## Startup (module-level)

When the server starts, three things are initialised globally:

**Audio transforms** — two torchaudio transform objects are created and moved to GPU (or CPU if no GPU is available):
- `MelSpectrogram`: converts a raw waveform into a mel spectrogram — a 2D representation where the x-axis is time, the y-axis is frequency (on a mel scale), and brightness represents energy.
- `AmplitudeToDB`: converts the spectrogram values from linear power to decibels, which better matches how human hearing (and bird call patterns) work.

```python
mel_transform = T.MelSpectrogram(sample_rate=32_000, n_fft=1024, hop_length=320, n_mels=128, f_min=50, f_max=16_000).to(device)
power_to_db   = T.AmplitudeToDB(stype="power", top_db=80).to(device)
```

**Model** — the checkpoint file (`models/best_model.pth`) is loaded. This file contains the trained weights and the `label2idx` dictionary that maps species names to integer indices. An EfficientNet B0 is rebuilt from that checkpoint. `model.eval()` switches off dropout and batch norm updates so inference is deterministic.

```python
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes, in_chans=3)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

**Label mapping** — `label2idx` (e.g. `{"compau": 3, ...}`) and its inverse `idx2label` (e.g. `{3: "compau", ...}`) are extracted from the checkpoint so the model output indices can be translated back to species names.

```python
label2idx = checkpoint["label2idx"]
idx2label = {v: k for k, v in label2idx.items()}
```

The model path defaults to `models/best_model.pth` relative to the repo root, but can be overridden with the `MODEL_PATH` environment variable (used inside Docker where the path is `/models/best_model.pth`).

## Endpoints

### `GET /health`
Returns `{"status": "ok"}`. Used to check that the server is running.

### `POST /predict`
Accepts an audio file upload and returns a probability for every species.

**Step 1 — Save and load the file**
The uploaded file is written to a temporary file on disk, loaded into a PyTorch tensor with `torchaudio.load()`, then the temp file is deleted. This is necessary because torchaudio needs a file path, not a raw byte stream.

```python
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(await file.read())
waveform, sr = torchaudio.load(tmp_path)
os.unlink(tmp_path)
```

**Step 2 — Normalise the audio**
If the file's sample rate isn't 32 000 Hz it is resampled. If the file has multiple channels (e.g. stereo) they are averaged into one mono channel.

```python
if sr != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```

**Step 3 — Chunk the audio**
The waveform is split into non-overlapping 5-second windows (160 000 samples each). If the last window is shorter than 5 seconds it is zero-padded to fill it. The result is a batch of shape `(n_chunks, 160000)`.

```python
remainder = total_samples % CHUNK_SAMPLES
if remainder:
    waveform = torch.nn.functional.pad(waveform, (0, CHUNK_SAMPLES - remainder))

chunks = waveform.squeeze(0).reshape(-1, CHUNK_SAMPLES)  # (n_chunks, 160000)
```

**Step 4 — Compute mel spectrograms**
All chunks are processed in one batched GPU call:
1. `MelSpectrogram` produces `(n_chunks, 1, 128, 501)` — 128 mel frequency bins, 501 time frames.
2. `AmplitudeToDB` converts to dB scale.
3. Min-max normalisation scales values to [0, 1] — this matches how normalisation was done during training.
4. The single channel is repeated 3 times to match the 3-channel input EfficientNet expects (it was pretrained on RGB images).

```python
mel    = mel_transform(chunks)                                         # (n_chunks, 1, 128, 501)
mel_db = power_to_db(mel)
mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
mel_db = mel_db.repeat(1, 3, 1, 1)                                    # (n_chunks, 3, 128, 501)
```

**Step 5 — Run the model**
The batch passes through EfficientNet B0 in one forward pass, producing logits of shape `(n_chunks, num_classes)`. `torch.sigmoid` converts these to independent probabilities between 0 and 1 for each species in each chunk.

```python
probs = torch.sigmoid(model(mel_db))  # (n_chunks, num_classes)
```

**Step 6 — Aggregate across chunks**
`probs.max(dim=0)` takes the highest probability seen for each species across all chunks. A species only needs to appear once anywhere in the recording to be detected, so the peak prediction across all windows is the best signal of its presence.

```python
max_probs = probs.max(dim=0).values.cpu().numpy()  # (num_classes,)
```

**Step 7 — Return**
A dictionary of `{ species_name: max_probability }` is returned for all species, e.g.:

```json
{
  "predictions": {
    "compau": 1.0,
    "blbwre1": 0.045,
    "bubwre1": 0.0162,
    ...
  }
}
```
