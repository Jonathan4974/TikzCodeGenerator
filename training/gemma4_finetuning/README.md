docker build -t gemma4-finetune .


docker run --rm -it \
  --gpus all \
  --add-host=host.docker.internal:host-gateway \
  --shm-size=32g \
  --env-file .env \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/train-big/:/data" \
  -v "/home/jonas/models:/models" \
  -v "$PWD/generated_images:/generated_images" \
  -v "/home/jonas/PycharmProjects/tikzcodegenerator/evaluation/promptfoo/pf_utils:/pf_utils" \
  gemma4-finetune \
  bash