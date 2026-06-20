# Evaluation

Shared metric implementations used by both benchmarking and training.
Think of this as the project's metrics library.

## What goes here
- Metric implementation scripts
- Evaluation utilities shared across milestones

## Metrics implemented
| Metric | Type | Script / location |
|---|---|---|
| CrystalBLEU (cBLEU) | Code similarity | `code_metrics.py` |
| Token Edit Distance (TED) | Code structure | `code_metrics.py` |
| DreamSim (DSim) | Perceptual image similarity | `image_metrics.py` |
| KID | Visual quality distribution | `image_metrics.py` |
| FID | Visual quality distribution | `image_metrics.py` |
| Image Structural Similarity (SSIM) | Perceptual image similarity | `promptfoo/assertions/image_structural_similarity.py` |
| LPIPS distance | Perceptual image similarity | `promptfoo/assertions/image_lpips.py` |
| DISTS distance | Perceptual image similarity | `promptfoo/assertions/image_dists.py` |
| CLIP similarity | Semantic similarity | `promptfoo/assertions/image_clip_similarity.py` |
| SigLIP similarity | Semantic similarity | `promptfoo/assertions/image_siglip_similarity.py` |
| Renderability | Code executability / render check | `promptfoo/assertions/tikz_is_renderable.py` |
| Compilation Rate | Code executability | `compiler_metrics.py` |
| Congruence Coefficient (CC) | Sketch structural similarity | `sketch_metrics.py` |

## Usage
All metrics are designed to be imported and called from benchmarking and training scripts:

```python
from evaluation.image_metrics import dreamsim, kid
from evaluation.code_metrics import cbleu, ted
from evaluation.compiler_metrics import compilation_rate
```

The promptfoo benchmark configuration at `evaluation/promptfoo/promptfooconfig.yaml` also uses additional evaluation assertions for SSIM, renderability, CLIP similarity, SigLIP similarity, LPIPS, DISTS, CrystalBLEU, and TED.


# Promptfoo Evaluation with Ollama on the I9 Cluster

This setup runs an Ollama model inside an Enroot container on a Slurm GPU node and evaluates the generated TikZ output with Promptfoo.

The setup is built around three scripts:

1. `setup_latex.sh`
2. `setup_envs.sh`
3. `ollama_promptfoo.sbatch`

The two setup scripts are executed once before running the Slurm job. The Slurm script is submitted whenever a new evaluation should be started.

## Expected Directory Structure

The setup assumes the following directory structure under the practical course user directory:

```text
/usr/prakt/<I9_USER>/projects/
├── data/
├── models/
├── tikzcodegenerator/
└── full_latex/
```

The directories are used as follows:

```text
data/
```

Contains the benchmark data, including images, reference TikZ files, captions, and manifest files.

```text
models/
```

Contains persistent model caches and downloaded model files, for example the Ollama model store, HuggingFace cache, Torch cache, DreamSim cache, and CrystalBLEU cache.

```text
full_latex/
```

Contains the local TeX Live installation.

## Scripts

### `run_setup_job_promptfoo_ollama.py`

This Python script is a small wrapper around the setup scripts and the Slurm job submission.

Since each Slurm job is limited to 8 hours, the full benchmark manifest should first be split into multiple smaller manifest files. The script can then run the evaluation split by split: it updates the Promptfoo configuration to point to the current manifest split, submits the corresponding Slurm job, waits for it to finish, and then starts the next split.

After all splits have finished, the individual results can be merged into one final benchmark result.

The python script can:

1. run `setup_latex.sh`,
2. run `setup_envs.sh`,
3. submit `ollama_promptfoo.sbatch`.

The script also passes the most important configuration values to the setup scripts and the Slurm job through environment variables:

```text
I9_USER
MODEL
TYPE
PROMPTFOO_CONFIG
```

This means that the user does not have to edit the shell scripts or the Slurm script manually for every run.

Run the full setup and submit the job:

```bash
python run_setup_job_promptfoo_ollama.py \
  --user <I9_USER> \
  --model gemma4:e2b-it-qat \
  --type img-tikz \
  --start-split 1 \
  --num-splits 10 \
  --wait-seconds 600 \
  --promptfoo-config /usr/prakt/<I9_USER>/projects/tikzcodegenerator/evaluation/promptfoo/configs/image_to_tikz_promptfooconfig_gemma_latest.yaml
```

If LaTeX and the Conda environments are already set up, skip both setup steps and only submit the Slurm job:

```bash
python run_setup_job_promptfoo_ollama.py \
  --user <I9_USER> \
  --skip-latex \
  --skip-envs \
  --model gemma4:e2b-it-qat \
  --type img-tikz \
  --start-split 1 \
  --num-splits 10 \
  --wait-seconds 600 \
  --promptfoo-config /usr/prakt/<I9_USER>/projects/tikzcodegenerator/evaluation/promptfoo/configs/image_to_tikz_promptfooconfig_gemma_latest.yaml
```

The `--promptfoo-config` argument expects an absolute path to the Promptfoo YAML configuration.


### `setup_latex.sh`

Installs a user-local TeX Live distribution into:

```text
/usr/prakt/<I9_USER>/projects/tools/texlive/2026
```

This is required because the Slurm compute nodes do not necessarily provide LaTeX, TikZ, `pdflatex`, or `dvisvgm` system-wide.

Run once:

```bash
I9_USER=<I9_USER> ./setup_latex.sh
```

### `setup_envs.sh`

Creates and configures the required Conda environments:

```text
ollama-client
promptfoo
```

The `ollama-client` environment runs the FastAPI wrapper around the Ollama API.

The `promptfoo` environment runs Promptfoo and the Python-based evaluation metrics such as CLIP similarity, LPIPS, DreamSim, SSIM, and CrystalBLEU.

Run once after `setup_latex.sh`:

```bash
I9_USER=<I9_USER> ./setup_envs.sh
```

Run this again only if dependencies change or the environments need to be repaired.

### `ollama_promptfoo.sbatch`

Runs the full evaluation on the Slurm GPU cluster.

The script:

1. requests one 24 GB GPU, preferably an RTX 6000,
2. starts the Ollama server from the Enroot image,
3. pulls the Ollama model if necessary,
4. starts the local Ollama client on port `8004`,
5. starts `promptfoo eval`,
6. starts the Promptfoo web UI on port `15500`,
7. copies the Promptfoo database back from the node-local temporary directory after the job ends.

The Promptfoo database is used from `/tmp` during the job to avoid SQLite locking issues on network filesystems. At the end of the job, it is copied back into the project result directory.

## Execution Order

Run the scripts in this order.

### 1. Install TeX Live

```bash
cd /usr/prakt/<I9_USER>/projects/tikzcodegenerator
I9_USER=<I9_USER> ./setup_latex.sh
```

### 2. Create Conda Environments

```bash
cd /usr/prakt/<I9_USER>/projects/tikzcodegenerator
I9_USER=<I9_USER> ./setup_envs.sh
```

### 3. Submit the Slurm Job

```bash
cd /usr/prakt/<I9_USER>/projects/tikzcodegenerator
I9_USER=<I9_USER> sbatch evaluation/promptfoo/ollama_promptfoo.sbatch
```

## Promptfoo Web UI

Promptfoo runs on port:

```text
15500
```

Find the node of the running job. You can find it in this log file in the first line:

```
tikzcodegenerator/evaluation/logs/ollama-<jobid>.out
```

If the job runs on `node9`, configure SSH forwarding like this:

```sshconfig
Host promptfoo-node9
    HostName atcremers<number>.in.tum.de
    User <I9_USER>
    Port 58022
    IdentityFile <path_to_key>
    IdentitiesOnly yes
    LocalForward 15500 node9:15500
```

Then connect:

```bash
ssh promptfoo-node9
```

Open locally:

```text
http://127.0.0.1:15500
```
