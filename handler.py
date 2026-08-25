import base64
import hashlib
import io
import os
import time
from urllib.parse import urlparse

import requests
import runpod
import soundfile as sf
import torch
import torchaudio

from src.models.bandit.bandit import Bandit
from src.system.inference_handler import StandardTensorChunkedInferenceHandler


SAMPLE_RATE = 48_000
STEMS = ["speech", "music", "sfx"]
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/opt/models/checkpoint-multi.ckpt")
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "4"))
CHECKPOINT_MD5 = "fea2868787551b0cff36cfcf7c3622a3"


def _build_model() -> Bandit:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

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
        hidden_activation_kwargs=None,
        complex_mask=True,
        use_freq_weights=True,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window_fn="hann_window",
        wkwargs=None,
        power=None,
        center=True,
        normalized=True,
        pad_mode="reflect",
        onesided=True,
        fs=SAMPLE_RATE,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    if loaded < int(len(model.state_dict()) * 0.9):
        raise RuntimeError(
            f"Checkpoint mismatch: loaded={loaded}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    model.cuda().eval()
    return model


MODEL = _build_model()
INFERENCE = StandardTensorChunkedInferenceHandler(
    chunk_size_seconds=8.0,
    hop_size_seconds=1.0,
    inference_batch_size=INFERENCE_BATCH_SIZE,
    fs=SAMPLE_RATE,
).cuda()


def _download(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("audio_url must be an HTTPS URL")

    with requests.get(url, stream=True, timeout=(10, 120), allow_redirects=True) as response:
        response.raise_for_status()
        chunks = []
        for chunk in response.iter_content(1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)


def _read_input(job_input: dict) -> bytes:
    if job_input.get("audio_url"):
        return _download(job_input["audio_url"])
    if job_input.get("audio_base64"):
        return base64.b64decode(job_input["audio_base64"], validate=True)
    raise ValueError("Provide audio_url or audio_base64")


def _decode_audio(data: bytes) -> tuple[torch.Tensor, int]:
    try:
        audio, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Unsupported or corrupt audio: {exc}") from exc

    if audio.shape[0] not in (1, 2):
        audio = audio[:2]
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)

    duration = audio.shape[-1] / SAMPLE_RATE
    if duration < 0.25:
        raise ValueError("Audio must be at least 0.25 seconds")
    return audio, sample_rate


def _encode_flac(audio: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    samples = audio.detach().float().cpu().numpy().T
    sf.write(buffer, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def _deliver_stem(stem: str, flac: bytes, upload_urls: dict) -> dict:
    upload_url = upload_urls.get(stem)
    digest = hashlib.sha256(flac).hexdigest()
    if upload_url:
        response = requests.put(
            upload_url,
            data=flac,
            headers={"Content-Type": "audio/flac"},
            timeout=(10, 600),
        )
        response.raise_for_status()
        return {
            "format": "flac",
            "sample_rate": SAMPLE_RATE,
            "delivery": "uploaded",
            "sha256": digest,
            "bytes": len(flac),
        }

    return {
        "format": "flac",
        "sample_rate": SAMPLE_RATE,
        "delivery": "inline",
        "sha256": digest,
        "bytes": len(flac),
        "audio_base64": base64.b64encode(flac).decode("ascii"),
    }


def handler(job: dict) -> dict:
    started = time.monotonic()
    try:
        job_input = job.get("input", {})
        raw = _read_input(job_input)
        audio, original_sample_rate = _decode_audio(raw)
        duration = audio.shape[-1] / SAMPLE_RATE

        with torch.inference_mode():
            output = INFERENCE(audio.unsqueeze(0).cuda(), MODEL)

        upload_urls = job_input.get("output_upload_urls", {})
        stems = {}
        for stem in STEMS:
            flac = _encode_flac(output["estimates"][stem]["audio"][0])
            stems[stem] = _deliver_stem(stem, flac, upload_urls)
        return {
            "status": "completed",
            "model": "multi",
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_sample_rate": original_sample_rate,
            "output_sample_rate": SAMPLE_RATE,
            "duration_seconds": round(duration, 3),
            "processing_seconds": round(time.monotonic() - started, 3),
            "stems": stems,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "failed", "error": "GPU_OUT_OF_MEMORY", "retryable": False}
    except (ValueError, requests.RequestException) as exc:
        return {"status": "failed", "error": str(exc), "retryable": False}
    except Exception as exc:
        return {"status": "failed", "error": f"INFERENCE_ERROR: {exc}", "retryable": False}


runpod.serverless.start({"handler": handler})
