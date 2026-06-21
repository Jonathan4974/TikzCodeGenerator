docker build -t tikzilla-8b-rl-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -p 8006:8006 \
  -e BASE_MODEL_PATH=nllg/TikZilla-8B-RL \
  -e QUANTIZATION=8bit \
  -e HF_TOKEN="" \
  tikzilla-8b-rl-cuda128