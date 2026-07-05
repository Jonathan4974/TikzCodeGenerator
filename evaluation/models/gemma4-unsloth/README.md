docker build -t gemma4-tikz-api .

docker run --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -v /home/jonas/models:/models \
  gemma4-tikz-api