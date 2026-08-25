FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG BANDIT_COMMIT=d5563d9031e95fdaa3e5a73d5020b9a0df61adb6
ARG CHECKPOINT_URL=https://zenodo.org/api/records/12701995/files/checkpoint-multi.ckpt/content
ARG CHECKPOINT_MD5=fea2868787551b0cff36cfcf7c3622a3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHECKPOINT_PATH=/opt/models/checkpoint-multi.ckpt

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        torchaudio==2.0.2 \
    && pip install \
        runpod==1.12.0 soundfile==0.12.1 tqdm==4.66.4

RUN git clone https://github.com/kwatcharasupat/bandit-v2.git /tmp/bandit-v2 \
    && git -C /tmp/bandit-v2 checkout "$BANDIT_COMMIT" \
    && cp -R /tmp/bandit-v2/src /app/src \
    && cp /tmp/bandit-v2/LICENSE /app/BANDIT-LICENSE \
    && rm -rf /tmp/bandit-v2

RUN mkdir -p /opt/models \
    && curl -fL --retry 5 --retry-delay 5 "$CHECKPOINT_URL" -o "$CHECKPOINT_PATH" \
    && echo "$CHECKPOINT_MD5  $CHECKPOINT_PATH" | md5sum -c -

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
