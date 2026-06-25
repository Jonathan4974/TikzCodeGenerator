docker build -t detikzify-2-5-8b-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4:/DaTikZ-V4" \
  -v "/home/jonas/models:/models/" \
  detikzify-2-5-8b-cuda128 \
  bash


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8000:8000 \
  -e QUANTIZATION=8bit \
  detikzify-2-5-8b-cuda128


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8000:8000 \
  -e QUANTIZATION=none \
  detikzify-2-5-8b-cuda128



on the server:

apptainer build detikzify-2-5-8b.sif docker-archive:///usr/prakt/s0030/projects/tikzcodegenerator/evaluation/models/detikzify_2_5_8b/detikzify_2_5_8b.tar


apptainer run --nv \
  --bind "$PWD:/app" \
  --bind "/usr/prakt/s0030/projects/models:/models" \
  --env QUANTIZATION=none \
  --pwd /app \
  detikzify-2-5-8b.sif