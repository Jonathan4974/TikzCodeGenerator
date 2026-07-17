docker build -t ollama-tikz-api .

docker run --rm -it \
  --network host \
  ollama-tikz-api

or

conda env create -f environment.yml
conda activate ollama-client
uvicorn app:app --host 0.0.0.0 --port 8444