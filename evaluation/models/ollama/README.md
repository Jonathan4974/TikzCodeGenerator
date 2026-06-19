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

### Running on the GPU cluster with Slurm

To run Ollama on the GPU cluster, use the provided Slurm script. With that you can test if ollama works correctly. 
The ollama_client_test.sbatch also starts the ollama client via a fastapi. This is useful for using ollama in promptfoo.

```bash
sbatch ollama_test.sbatch
```

```bash
sbatch ollama_client_test.sbatch
```

You can check the job status with:

```bash
squeue -u s0030
```

If needed, cancel the job with:

```bash
scancel <jobid>
```

The Slurm output logs are written to the path configured in the `#SBATCH --output=...` line of `ollama_test.sbatch`.

For example, if the script contains:

```bash
#SBATCH --output=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-%j.out
```

then the log file will be written to:

```text
/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-<jobid>.out
```

You can inspect the log with:

```bash
tail -f /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-<jobid>.out
```
