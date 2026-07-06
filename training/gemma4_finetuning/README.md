docker build -t gemma4-finetune .


docker run --rm -it \
  --gpus all \
  --add-host=host.docker.internal:host-gateway \
  --shm-size=32g \
  --env-file .env \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/train-big/:/data" \
  -v "/home/jonas/models:/models" \
  -v "/home/jonas/PycharmProjects/tikzcodegenerator/evaluation/promptfoo/pf_utils:/pf_utils" \
  gemma4-finetune \
  bash



tensorboard --logdir /home/jonas/models/gemma4-sft/gemma4_sft --host 0.0.0.0 --port 6006