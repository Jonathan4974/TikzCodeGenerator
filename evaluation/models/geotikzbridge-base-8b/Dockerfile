FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MODEL_CHECKPOINT=SJY-1995/GeoTikzBridge-Base-8B
ENV MAX_NEW_TOKENS=4096
ENV TEMPERATURE=0.1
ENV TOP_P=0.95

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    dvisvgm \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    accelerate \
    bitsandbytes \
    safetensors \
    transformers \
    huggingface_hub \
    hf_transfer

RUN pip install \
    fastapi \
    uvicorn \
    python-multipart \
    pillow

RUN pip install -r https://raw.githubusercontent.com/sjy-1995/GeoTikzBridge/main/requirements/internvl_chat.txt

COPY app.py /app/app.py

EXPOSE 8007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]
