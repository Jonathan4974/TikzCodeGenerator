docker build -t promptfoo-tex .

docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/result/promptfoo-db:/root/.promptfoo" \
  -v "$PWD/result/generated_images:/generated_images" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/image_manifest.csv:/image_manifest.csv" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/benchmark_corpus:/crystalbleu_corpus" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/images:/images" \
  -v "/home/jonas/Datasets/TikZ/benchmark_data/references:/references" \
  -v "/home/jonas/models:/models" \
  -p 15500:15500 \
  promptfoo-tex \
  sh -c "promptfoo eval -c configs/image_to_tikz_promptfooconfig_detikzify_2_5_8b.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no"



