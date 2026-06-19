## Running Ollama

Ollama can be run either with Docker Compose or with Enroot.

### Option A: Docker Compose

If you want to use Docker, start the Ollama service with:

```bash
docker compose up
```

### Option B: Enroot

Alternatively, you can run Ollama with Enroot.

#### 1. Import the Ollama image

Download and convert the official Ollama Docker image:

```bash
enroot import docker://ollama/ollama:latest
```

This creates an Enroot image file, usually named:

```text
ollama+ollama+latest.sqsh
```

#### 2. Start the Ollama server

Start the Ollama server and mount a persistent model directory:

```bash
enroot start \
  --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --mount /usr/prakt/s0030/projects/models/ollama_model_store:/ollama-model-store \
  ollama+ollama+latest.sqsh \
  serve
```

The mounted directory is used to store downloaded models persistently on the host system.

#### 3. Check downloaded models

In another terminal, check which models are available:

```bash
curl http://127.0.0.1:11434/api/tags
```

#### 4. Pull a model

Download the required model through the Ollama API:

```bash
curl http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e2b-it-qat"
  }'
```

#### 5. Test the model

After the model has been pulled, you can test it with a simple generation request:

```bash
curl http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e2b-it-qat",
    "prompt": "Erkläre kurz, was ein AIG ist.",
    "stream": false
  }'
```
