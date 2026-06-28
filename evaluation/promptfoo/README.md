# Promptfoo with Enroot

## 0. Convert Docker `.tar` to Enroot `.sqsh`

Run this on a machine that has **Docker + Enroot** installed. E.g. your local machine.

```bash
docker build -t promptfoo-tex .
```

Then import the Docker image into Enroot.

Adjust the image name/tag according to the output of `docker images`:

```bash
enroot import --output promptfoo-tex.sqsh dockerd://promptfoo-tex:latest
```

Copy the `.sqsh` file to the cluster.

---

## 1. Create the Enroot container on the cluster

cd ~/projects/

mkdir -p .enroot/{data,cache,tmp}

export ENROOT_DATA_PATH="$PWD/.enroot/data"
export ENROOT_CACHE_PATH="$PWD/.enroot/cache"
export ENROOT_TEMP_PATH="$PWD/.enroot/tmp"

```bash
enroot create --name promptfoo promptfoo-tex.sqsh
```

---

## 2. Create required host directories

The host-side directories used in `--mount` must already exist.

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/promptfoo

mkdir -p result/promptfoo-db
mkdir -p result/generated_images
```

Check that all required host paths exist:

```bash
ls -lah /usr/prakt/s0030/projects/models
ls -lah /usr/prakt/s0030/projects/data/benchmark_data/manifest_splits/image_manifest_1.csv
ls -lah /usr/prakt/s0030/projects/data/benchmark_data/benchmark_corpus
ls -lah /usr/prakt/s0030/projects/data/benchmark_data/images
ls -lah /usr/prakt/s0030/projects/data/benchmark_data/references
```

---

## 3. TMUX commands
Start a new tmux:
```bash
tmux new -s promptfoo_splits
```

To exit: Ctrl-b, then d

to join a running tmux:

```bash
tmux attach -t promptfoo_splits
```

List running tmux:
```bash
tmux ls
```

Stop a tmux:
```bash
exit
```
or 
```bash
tmux kill-session -t promptfoo_splits
```

## 4. Make GPUs accessible inside of the container
Run these two commands:

```bash
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

## 5. Start Promptfoo

DeTikZify must already be running.

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/promptfoo

enroot start --root --rw \
  --mount "$PWD:/app" \
  --mount "$PWD/result/promptfoo-db:/root/.promptfoo" \
  --mount "$PWD/result/generated_images:/generated_images" \
  --mount "/usr/prakt/s0030/projects/data/benchmark_data/manifest_splits:/manifest_splits" \
  --mount "/usr/prakt/s0030/projects/data/benchmark_data/benchmark_corpus:/crystalbleu_corpus" \
  --mount "/usr/prakt/s0030/projects/data/benchmark_data/images:/images" \
  --mount "/usr/prakt/s0030/projects/data/benchmark_data/references:/references" \
  --mount "/usr/prakt/s0030/projects/models:/models" \
  promptfoo \
  sh -lc '
    cd /app
    ln -sf /manifest_splits/image_manifest.csv /image_manifest.csv
    set -a
    . ./.env
    set +a
    promptfoo eval -c configs/image_to_tikz_promptfooconfig_detikzify_2_5_8b.yaml -j 1 --watch &
    sleep 5
    promptfoo view --port 15500 --no
  '
```

---

## 6. Open Promptfoo UI

If running on the same machine:

```bash
http://127.0.0.1:15500
```

If running on a cluster node, use an SSH tunnel if needed.



## Only on the local machine
```bash
docker build -t promptfoo-tex .
```

```bash
docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/result/promptfoo-db:/root/.promptfoo" \
  -v "$PWD/result/generated_images:/generated_images" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/image_manifest.csv:/image_manifest.csv" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/benchmark_corpus:/crystalbleu_corpus" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/images:/images" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/references:/references" \
  -v "/home/jonas/models:/models" \
  -p 15500:15500 \
  promptfoo-tex \
  sh -c "promptfoo eval -c configs/image_to_tikz_promptfooconfig_geotikzbridge-8b.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no"
  ```


tmux new -s promptfoo_splits
Ctrl-b, then d
tmux attach -t promptfoo_splits
tmux ls
exit
tmux kill-session -t promptfoo_splits
