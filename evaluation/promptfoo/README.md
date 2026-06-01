docker build -t promptfoo-tex .

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/app" \
  -v "$PWD/promptfoo-db:/root/.promptfoo" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/images:/images" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/references:/references" \
  -v promptfoo-hf-cache:/root/.cache/huggingface \
  -v promptfoo-torch-cache:/root/.cache/torch \
  -p 15500:15500 \
  -e PYTHONPATH=/app \
  promptfoo-tex