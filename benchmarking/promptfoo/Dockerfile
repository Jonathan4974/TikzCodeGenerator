FROM node:22-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PROMPTFOO_DISABLE_TELEMETRY=1
ENV PROMPTFOO_DISABLE_UPDATE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-fonts-recommended \
    texlive-science \
    texlive-extra-utils \
    latexmk \
    dvisvgm \
    ghostscript \
    poppler-utils \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    pandas \
    pyarrow \
    pillow \
    numpy \
    scikit-image \
    nltk \
    crystalbleu \
    latexcodec \
    torchmetrics \
    torch \
    torchvision \
    transformers \
    sentencepiece 


# Promptfoo installieren
RUN npm install -g promptfoo

COPY . .

CMD ["sh", "-c", "promptfoo eval --watch & sleep 5 && promptfoo view --port 15500 --no"]