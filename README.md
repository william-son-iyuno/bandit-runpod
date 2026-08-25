# Bandit v2 RunPod worker

RunPod Serverless wrapper for the multilingual Bandit v2 cinematic audio source-separation checkpoint.

The endpoint accepts `audio_url` (HTTPS) or `audio_base64` and separates the full input into `speech`, `music`, and `sfx`. There is no fixed duration limit.

For long audio, pass presigned PUT URLs in `output_upload_urls`. Without upload URLs the FLAC files are returned inline as base64 and may exceed RunPod's response payload limit.

Upstream code: https://github.com/kwatcharasupat/bandit-v2 (Apache-2.0)

Model weights: https://zenodo.org/records/12701995 (CC BY-SA 4.0)

Example input:

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "output_upload_urls": {
      "speech": "https://storage.example.com/speech.flac?presigned=...",
      "music": "https://storage.example.com/music.flac?presigned=...",
      "sfx": "https://storage.example.com/sfx.flac?presigned=..."
    }
  }
}
```
