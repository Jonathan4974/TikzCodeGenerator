docker build -t promptfoo-tex .

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/promptfoo-db:/root/.promptfoo" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/images:/images" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/references:/references" \
  -v promptfoo-hf-cache:/root/.cache/huggingface \
  -v promptfoo-torch-cache:/root/.cache/torch \
  -p 15500:15500 \
  promptfoo-tex \
  sh -c "promptfoo eval -c configs/image_to_tikz_promptfooconfig_specialized.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no"



docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/promptfoo-db:/root/.promptfoo" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/images:/images" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4/references:/references" \
  -v promptfoo-hf-cache:/root/.cache/huggingface \
  -v promptfoo-torch-cache:/root/.cache/torch \
  -p 15500:15500 \
  promptfoo-tex \
  sh -c "promptfoo eval -c configs/image_to_tikz_promptfooconfig_general_vlm.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no"

