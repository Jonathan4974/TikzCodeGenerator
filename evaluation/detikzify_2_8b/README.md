docker build -t detikzify-2-8b-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4:/DaTikZ-V4" \
  -v "/home/jonas/models:/models/" \
  detikzify-2-8b-cuda128 \
  bash


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8001:8001 \
  detikzify-2-8b-cuda128


With 8bit quantization it need 15965MiB VRAM