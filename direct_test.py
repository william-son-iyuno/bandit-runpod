import json
import time

import torch
import torchaudio

from src.models.bandit.bandit import Bandit
from src.system.inference_handler import StandardTensorChunkedInferenceHandler


FS = 48_000
STEMS = ["speech", "music", "sfx"]


def main():
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
    checkpoint = torch.load(
        "/workspace/models/checkpoint-multi.ckpt", map_location="cpu"
    )
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    model.cuda().eval()

    audio, sample_rate = torchaudio.load("/workspace/input.wav")
    if sample_rate != FS:
        audio = torchaudio.functional.resample(audio, sample_rate, FS)
    inference = StandardTensorChunkedInferenceHandler(
        chunk_size_seconds=8.0,
        hop_size_seconds=1.0,
        inference_batch_size=4,
        fs=FS,
    ).cuda()

    started = time.monotonic()
    with torch.inference_mode():
        output = inference(audio.unsqueeze(0).cuda(), model)
    elapsed = time.monotonic() - started

    for stem in STEMS:
        torchaudio.save(
            f"/workspace/{stem}.wav",
            output["estimates"][stem]["audio"][0].cpu(),
            FS,
        )

    print(
        json.dumps(
            {
                "loaded_keys": len(model_state),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "input_shape": list(audio.shape),
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gb": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "outputs": STEMS,
            }
        )
    )


if __name__ == "__main__":
    main()
