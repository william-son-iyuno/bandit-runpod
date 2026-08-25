# Bandit v2 on RunPod

RunPod Serverless wrapper for the multilingual Bandit v2 cinematic audio source-separation checkpoint.

The active Pod API accepts an uploaded audio file and separates the full input into `speech`, `music`, and `sfx`. There is no fixed duration limit in the application.

Active API base URL:

```text
https://2kxlgz9yd1ms6c-8000.proxy.runpod.net
```

Retrieve the API key from the Pod over SSH and call the endpoint:

```bash
API_KEY=$(ssh -i ~/.runpod/ssh/RunPod-Key-Go \
  -p 51872 root@213.181.111.2 \
  'cat /workspace/.bandit-api-key')

curl -X POST \
  'https://2kxlgz9yd1ms6c-8000.proxy.runpod.net/separate' \
  -H "X-API-Key: $API_KEY" \
  -F 'file=@input.wav' \
  -o bandit-stems.zip
```

The ZIP contains `speech.flac`, `music.flac`, `sfx.flac`, and `metadata.json`.

Health check:

```bash
curl 'https://2kxlgz9yd1ms6c-8000.proxy.runpod.net/health'
```

After a Pod restart, reconnect over SSH and run:

```bash
nohup /workspace/start_pod.sh > /workspace/bandit-api.log 2>&1 &
echo $! > /workspace/bandit-api.pid
```

Upstream code: https://github.com/kwatcharasupat/bandit-v2 (Apache-2.0)

Model weights: https://zenodo.org/records/12701995 (CC BY-SA 4.0)

The Pod currently uses one RTX 4090 and costs approximately $0.74/hour while running.
