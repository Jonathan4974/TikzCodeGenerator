enroot start \
  --rw \
  --mount "$PWD:/app" \
  --mount /usr/prakt/s0030/projects/data/benchmark_data/images:/images \
  ollama-client.sqsh \
  uvicorn app:app --host 0.0.0.0 --port 8444