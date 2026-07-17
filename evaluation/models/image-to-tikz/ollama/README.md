docker build -t ollama-tikz-api .

docker run --rm -it \
  --network host \
  ollama-tikz-api