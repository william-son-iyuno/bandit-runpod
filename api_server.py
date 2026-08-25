import asyncio
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.models.bandit.bandit import Bandit
from src.system.inference_handler import StandardTensorChunkedInferenceHandler


FS = 48_000
STEMS = ["speech", "music", "sfx"]
CHECKPOINT = os.getenv(
    "BANDIT_CHECKPOINT", "/workspace/models/checkpoint-multi.ckpt"
)
BATCH_SIZE = int(os.getenv("BANDIT_BATCH_SIZE", "4"))
API_KEY = os.getenv("BANDIT_API_KEY")

app = FastAPI(title="Bandit v2 Source Separation API", version="0.1.0")
inference_lock = asyncio.Lock()


def load_runtime():
    model = Bandit(
        in_channels=1,
        stems=STEMS,
        band_type="musical",
        n_bands=64,
        normalize_channel_independently=False,
        treat_channel_as_feature=True,
        n_sqm_modules=8,
        emb_dim=128,
        rnn_dim=256,
        bidirectional=True,
        rnn_type="GRU",
        mlp_dim=512,
        hidden_activation="Tanh",
        complex_mask=True,
        use_freq_weights=True,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window_fn="hann_window",
        center=True,
        normalized=True,
        pad_mode="reflect",
        onesided=True,
        fs=FS,
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.cuda().eval()
    inference = StandardTensorChunkedInferenceHandler(
        chunk_size_seconds=8.0,
        hop_size_seconds=1.0,
        inference_batch_size=BATCH_SIZE,
        fs=FS,
    ).cuda()
    return model, inference


MODEL, INFERENCE = load_runtime()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "multi",
        "gpu": torch.cuda.get_device_name(0),
        "sample_rate": FS,
        "stems": STEMS,
    }


def _separate(input_path: Path, workdir: Path):
    audio, sample_rate = torchaudio.load(str(input_path))
    if audio.shape[0] > 2:
        audio = audio[:2]
    if sample_rate != FS:
        audio = torchaudio.functional.resample(audio, sample_rate, FS)
    if audio.shape[-1] < FS // 4:
        raise ValueError("Audio must be at least 0.25 seconds")

    started = time.monotonic()
    with torch.inference_mode():
        output = INFERENCE(audio.unsqueeze(0).cuda(), MODEL)

    output_paths = []
    for stem in STEMS:
        output_path = workdir / f"{stem}.flac"
        torchaudio.save(
            str(output_path),
            output["estimates"][stem]["audio"][0].cpu(),
            FS,
            format="flac",
        )
        output_paths.append(output_path)

    return output_paths, time.monotonic() - started, audio.shape[-1] / FS


@app.post("/separate")
async def separate(
    file: UploadFile = File(...), x_api_key: str | None = Header(default=None)
):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    workdir = Path(tempfile.mkdtemp(prefix="bandit-"))
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    input_path = workdir / f"input{suffix}"
    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        async with inference_lock:
            paths, elapsed, duration = await asyncio.to_thread(
                _separate, input_path, workdir
            )

        zip_path = workdir / "bandit-stems.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in paths:
                archive.write(path, path.name)
            archive.writestr(
                "metadata.json",
                (
                    '{"model":"multi","sample_rate":48000,'
                    f'"duration_seconds":{duration:.3f},'
                    f'"processing_seconds":{elapsed:.3f}}}'
                ),
            )

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="bandit-stems.zip",
            background=BackgroundTask(shutil.rmtree, workdir, True),
        )
    except (RuntimeError, ValueError) as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
