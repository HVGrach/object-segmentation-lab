FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUPYTER_PORT=8888 \
    JUPYTER_TOKEN=lab-object-segmentation

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 && \
    pip install -r /tmp/requirements.txt

COPY . /workspace

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-lc", "jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser --allow-root --ServerApp.root_dir=/workspace --ServerApp.token=${JUPYTER_TOKEN}"]
