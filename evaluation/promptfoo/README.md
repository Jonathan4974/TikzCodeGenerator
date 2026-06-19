docker build -t promptfoo-tex .

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/promptfoo-db:/root/.promptfoo" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/small_manifest.csv:/image_manifest.csv" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/images:/images" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/references:/references" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/captions:/captions" \
  -v promptfoo-hf-cache:/root/.cache/huggingface \
  -v promptfoo-torch-cache:/root/.cache/torch \
  -v promptfoo-dreamsim-cache:/root/.cache/dreamsim \
  -p 15500:15500 \
  promptfoo-tex \
  sh -c "promptfoo eval -c configs/image_to_tikz_promptfooconfig_qwen3_6_35b_latest.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no"
