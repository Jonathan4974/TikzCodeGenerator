import os
import time
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def main():
    # 1. Path Configurations
    repo_id = "SJY-1995/GeoTikz-Base"
    output_dir = "data/description_augment"
    output_file = os.path.join(output_dir, "GeoTikz_Base_local.parquet")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing streaming connection to Hugging Face for: {repo_id}")
    
    # 2. Enable Streaming Load
    # Setting streaming=True ensures that it only fetches the indices and metadata,
    # preventing the massive dataset from being entirely loaded into RAM at once.
    ds_stream = load_dataset(repo_id, split="train", streaming=True)
    
    # 3. Chunking & State Configurations
    CHUNK_SIZE = 2000  # Dump data to disk every 2000 rows to balance RAM usage and I/O performance
    chunk = []
    is_first_chunk = True
    
    print(f"Downloading and streaming into local single file: {output_file}")
    print("----------------------------------------------------------------")

    # 4. Progress Management via tqdm
    # Note: Streamed datasets do not expose a total length (len) upfront, so tqdm 
    # cannot display a percentage. However, it will perfectly track processing 
    # speed (rows/s), total rows downloaded, and elapsed time.
    with tqdm(unit=" rows", desc="Streaming GeoTikz Data") as pbar:
        for entry in ds_stream:
            chunk.append(entry)
            pbar.update(1)  # Increment progress bar by 1 row
            
            # When the chunk buffer is full, convert to DataFrame and append to Parquet
            if len(chunk) >= CHUNK_SIZE:
                df = pd.DataFrame(chunk)
                
                # If it's the first chunk, overwrite/create a new file.
                # For subsequent chunks, physically append them to the existing file.
                df.to_parquet(output_file, engine="pyarrow", append=not is_first_chunk)
                
                is_first_chunk = False
                chunk = []  # Flush the buffer to release RAM memory

        # 5. Handle remaining residual rows after the main loop finishes
        if chunk:
            df = pd.DataFrame(chunk)
            df.to_parquet(output_file, engine="pyarrow", append=not is_first_chunk)
            
    print("----------------------------------------------------------------")
    print(f"Success! All parts successfully combined and saved to:")
    print(f"{os.path.abspath(output_file)}")
    print(f"Local File Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()