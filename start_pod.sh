#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q \
  'pytorch-lightning==2.3.0' \
  'torchmetrics==0.11.4' \
  'librosa==0.10.2.post1' \
  'fastapi>=0.115' \
  'uvicorn[standard]>=0.30' \
  'python-multipart>=0.0.9'

export PYTHONPATH=/workspace/bandit-v2
export BANDIT_CHECKPOINT=/workspace/models/checkpoint-multi.ckpt
export BANDIT_API_KEY
BANDIT_API_KEY=$(< /workspace/.bandit-api-key)

cd /workspace/bandit-v2
exec uvicorn api_server:app \
  --app-dir /workspace \
  --host 0.0.0.0 \
  --port 8000
