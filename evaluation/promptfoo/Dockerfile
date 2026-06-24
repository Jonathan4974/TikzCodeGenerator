FROM node:22-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PROMPTFOO_DISABLE_TELEMETRY=1
ENV PROMPTFOO_DISABLE_UPDATE=1

# Aktuelle Vanilla TeX Live Installation
ENV TEXLIVE_YEAR=2026
ENV TEXLIVE_DIR=/usr/local/texlive/2026
ENV PATH="/usr/local/texlive/2026/bin/x86_64-linux:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    perl \
    wget \
    xz-utils \
    fontconfig \
    ghostscript \
    poppler-utils \
    dvisvgm \
    ca-certificates \
    git \
    curl \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Vanilla TeX Live full von CTAN/TUG installieren
# option_doc 0 und option_src 0: alle Pakete, aber ohne Doku und Source-Dateien
RUN mkdir -p /tmp/install-tl \
    && cd /tmp/install-tl \
    && wget -qO- https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
        | tar xzf - --strip-components=1 \
    && printf "%s\n" \
        "selected_scheme scheme-full" \
        "TEXDIR /usr/local/texlive/2026" \
        "TEXMFCONFIG ~/.texlive2026/texmf-config" \
        "TEXMFHOME ~/texmf" \
        "TEXMFLOCAL /usr/local/texlive/texmf-local" \
        "TEXMFSYSCONFIG /usr/local/texlive/2026/texmf-config" \
        "TEXMFSYSVAR /usr/local/texlive/2026/texmf-var" \
        "TEXMFVAR ~/.texlive2026/texmf-var" \
        "option_doc 0" \
        "option_src 0" \
        > texlive.profile \
    && ./install-tl -profile texlive.profile \
    && /usr/local/texlive/2026/bin/x86_64-linux/tlmgr path add \
    && rm -rf /tmp/install-tl

RUN pip3 install --break-system-packages \
    pandas \
    pyarrow \
    pillow \
    numpy \
    scikit-learn \
    scikit-image \
    nltk \
    crystalbleu \
    latexcodec \
    torchmetrics \
    pygments \
    sacremoses \
    datasets \
    huggingface_hub \
    torch \
    torchvision \
    transformers \
    sentencepiece \
    dreamsim \
    peft \
    pymupdf \
    protobuf

RUN npm install -g promptfoo

COPY . .

CMD ["sh", "-c", "promptfoo eval --watch & sleep 5 && promptfoo view --port 15500 --no"]