conda env create -f environment.yaml
conda activate gemma4-server

uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1