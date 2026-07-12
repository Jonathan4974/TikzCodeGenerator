import os
import io
import time
import json
import signal
import sys
import base64
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image

# ==================== Config ====================
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"
DATASET_NAME = "SJY-1995/GeoTikz-Base"
OUTPUT_PARQUET_PATH = "/usr/prakt/s0042/projects/data/parquet_files/geotikz_augmented.parquet"
CHECKPOINT_PATH = "/usr/prakt/s0042/projects/data/parquet_files/geotikz_checkpoint.json"

TOKEN_LIMIT = 8192
BATCH_SIZE = 32
NUM_WORKERS = 4         

PROMPT = """You are a scientific illustrator describing images for precise redrawing in TikZ.\n
Your task is to describe the image in precise, continuous prose without bullet points,
lists, or line breaks.\n
Start directly with the main object or scene. Avoid introductory phrases like
'Certainly!', 'The image depicts...', 'Here is a precise description.'.\n
Use clear, active language focused on geometry, labels, colors, spatial relationships,
coordinates, and other visible properties.\n
Describe all visible elements such as shapes, lines, arrows, and labels, including
their relative or absolute positions, dimensions, and orientation.\n
Use consistent, minimal naming for objects (e.g., 'circle A', 'line L1') and specify
label positions relative to shapes precisely.\n
Only describe exact, concrete visual elements that enable precise image reconstruction
in TikZ.\n
Avoid vague, interpretive, or inferential language, and exclude summaries,
conclusions, or commentary about the image's meaning, function, or aesthetics.\n
Here are a few examples:\n
A thin black horizontal line centered in the middle, containing nine evenly spaced
black dots, and labeled x2 at the left. Each dot is connected by a thin black line in
an alternating pattern to either x0 (placed at the top middle) or x1 (placed at the
bottom middle).\n
A line chart has different instruction scales of 1/10, 1/4, 1/2, and 1 on the x-axis.
On the y-axis it shows BLEU scores between 20 and 50, with steps of 5. The chart
contains three lines with Zh-En in blue, De-En in red, and Fr-En in brown. All BLEU
scores are initially 20 at the lowest instruction scale. As the instruction scale
increases, BLEU scores improve for all pairs. De-En is the highest, closely followed
by Fr-En and then Zh-En far below. The increase is largest from 1/10 to 1/4 and only
marginally above an instruction scale of 1/4. The legend is placed inside the chart
at the top left.\n
Write a description in this exact style for the given image."""
# ================================================

# Global flags for graceful shutdown
SHUTDOWN_TRIGGERED = False
parquet_writer = None
buffer_chunk = []
current_stream_index = 0  # Tracks absolute index in HF stream

try:
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gemma-tokenizer")
except Exception as e:
    print(f"Tokenizer fallback activated: {e}")
    tokenizer = None

# define PyArrow Schema
schema = pa.schema([
    ('images', pa.binary()),
    ('captions', pa.string()),
    ('references', pa.string())
])


def check_token_length(text):
    """use Gemma Tokenizer to check if tikz_code shorter than LIMIT"""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text)) < TOKEN_LIMIT
        except Exception:
            pass
    return (len(text) // 2) < TOKEN_LIMIT


def query_qwen_vl(image_bytes, prompt="Please describe this geometric image in detail."):
    """ Ollama query method"""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "images": [base64_image]
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"[Error] Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"[Error] Request failed: {str(e)}"


def process_single_sample(sample, idx):
    response_text = sample.get('response', '')
    
    if not check_token_length(response_text):
        return None
        
    img_field = sample.get('image')
    if isinstance(img_field, dict) and 'bytes' in img_field:
        img_bytes = img_field['bytes']
    elif isinstance(img_field, Image.Image):
        img_byte_arr = io.BytesIO()
        img_field.save(img_byte_arr, format=img_field.format or 'PNG')
        img_bytes = img_byte_arr.getvalue()
    elif isinstance(img_field, bytes):
        img_bytes = img_field
    else:
        return None

    caption = query_qwen_vl(img_bytes)
    
    return {
        'images': img_bytes,
        'captions': caption,
        'references': response_text,
        'dataset_row_idx': idx, # Keep track of which row this belonged to
    }


def save_chunk_to_parquet(chunk_data, writer=None):
    columns = {k: [d[k] for d in chunk_data] for k in schema.names}
    table = pa.Table.from_pydict(columns, schema=schema)
    
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, schema, append=os.path.exists(OUTPUT_PARQUET_PATH))
    writer.write_table(table)
    return writer


def save_checkpoint(last_processed_idx):
    """Saves the progress tracking mark into a small JSON file"""
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({"last_processed_idx": last_processed_idx}, f)
    print(f"Progress saved! Next boot will skip up to row index: {last_processed_idx}")


def load_checkpoint():
    """Reads where we left off last time"""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                data = json.load(f)
                return data.get("last_processed_idx", 0)
        except Exception:
            return 0
    return 0

def slurm_termination_handler(signum, frame):
    """Triggered automatically when Slurm sends a timeout warning (SIGTERM)"""
    global SHUTDOWN_TRIGGERED
    print("\nWARNING: Slurm timeout approaching! Initiating emergency safe-save...")
    SHUTDOWN_TRIGGERED = True  # Signal the main loop to freeze and flush immediately

# Register the signal interceptors
signal.signal(signal.SIGTERM, slurm_termination_handler)
signal.signal(signal.SIGINT, slurm_termination_handler)

def main():
    global parquet_writer, buffer_chunk, current_stream_index

    start_row = load_checkpoint()
    if start_row > 0:
        print(f"Resuming from checkpoint. Skipping the first {start_row} samples...")
    else:
        print("No checkpoint found. Starting data generation from scratch.")
    
    print(f"stream dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
    dataset_iter = iter(dataset)

    # Fast-forward the stream up to our checkpoint mark
    for _ in range(start_row):
        try:
            next(dataset_iter)
            current_stream_index += 1
        except StopIteration:
            print("Checkpoint out of bounds. Dataset already fully consumed.")
            return
        
    total_saved = 0
    max_last_idx = start_row
    start_time = time.time()

    # Initialize a thread pool
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        
        # Pre-populate the thread pool queue with a small buffer of tasks
        for _ in range(NUM_WORKERS * 2):
            try:
                sample = next(dataset_iter)
                current_stream_index += 1
                fut = executor.submit(process_single_sample, sample, current_stream_index)
                futures[fut] = current_stream_index
            except StopIteration:
                break
        
        # Process tasks dynamically as they finish running
        while futures and not SHUTDOWN_TRIGGERED:
            for fut in as_completed(futures):
                idx_mark = futures.pop(fut)
                
                # Fetch execution result from completed thread
                result = fut.result()
                if result is not None:
                    max_last_idx = max(max_last_idx, result.pop('dataset_row_idx'))
                    buffer_chunk.append(result)
                    total_saved += 1
                else:
                    max_last_idx = max(max_last_idx, idx_mark)
                
                # Commit micro-batch directly to disk when buffer is full
                if len(buffer_chunk) >= BATCH_SIZE:
                    parquet_writer = save_chunk_to_parquet(buffer_chunk, parquet_writer)
                    buffer_chunk = []
                    save_checkpoint(max_last_idx)
                    print(f"[Streaming active] Total valid added this run: {total_saved}")
                
                # Pull a fresh sample from stream to keep the workers active
                if not SHUTDOWN_TRIGGERED:
                    try:
                        next_sample = next(dataset_iter)
                        current_stream_index += 1
                        new_fut = executor.submit(process_single_sample, next_sample, current_stream_index)
                        futures[new_fut] = current_stream_index
                    except StopIteration:
                        pass
                break 
        
        if SHUTDOWN_TRIGGERED:
            print("Processing loop frozen. Cleaning worker pools...")
            # Cancel anything queued up that hasn't started yet
            executor.shutdown(wait=False, cancel_futures=True)

        # Write any remaining samples left over in the final buffer chunk
        if buffer_chunk:
            parquet_writer = save_chunk_to_parquet(buffer_chunk, parquet_writer)
            save_checkpoint(max_last_idx)

    # Safely close the Parquet file handler and finalize structural metadata
    if parquet_writer:
        parquet_writer.close()
        
    if SHUTDOWN_TRIGGERED:
        print(f"Emergency shutdown complete. Safe to be terminated by Slurm. Resume index: {max_last_idx}")
        sys.exit(0)
    else:
        print(f"\nDataset processing fully completed from end-to-end! Time elapsed: {(time.time() - start_time) / 3600:.2f}h")

if __name__ == "__main__":
    main()