docker build -t detikzify-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4:/DaTikZ-V4" \
  -v "/home/jonas/models:/models/" \
  detikzify-cuda128 \
  bash


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8000:8000 \
  detikzify-cuda128