docker build -t geotikzbridge-base-8b-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -p 8004:8004 \
  geotikzbridge-base-8b-cuda128
