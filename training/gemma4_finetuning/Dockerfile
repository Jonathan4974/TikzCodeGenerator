FROM python:3.13.2-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget perl xz-utils fontconfig ghostscript poppler-utils dvisvgm \
    ca-certificates git curl build-essential \
    fonts-dejavu fonts-liberation fonts-noto fonts-noto-cjk \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV TEXLIVE_YEAR=2026
ENV TEXLIVE_DIR=/usr/local/texlive/2026
ENV PATH="/usr/local/texlive/2026/bin/x86_64-linux:${PATH}"

RUN mkdir -p /tmp/install-tl \
    && cd /tmp/install-tl \
    && wget -qO- https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
        | tar xzf - --strip-components=1 \
    && printf "%s\n" \
        "selected_scheme scheme-full" \
        "TEXDIR /usr/local/texlive/2026" \
        "option_doc 0" \
        "option_src 0" \
        > texlive.profile \
    && ./install-tl -profile texlive.profile \
    && rm -rf /tmp/install-tl

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir -U \
    unsloth \
    datasets transformers accelerate peft trl bitsandbytes \
    pillow numpy scipy scikit-learn scikit-image pandas pyarrow \
    torchmetrics lpips dreamsim crystalbleu pygments sacremoses \
    pymupdf protobuf sentencepiece huggingface_hub hf_transfer nltk \
    tensorboard tensorflow

CMD ["bash"]