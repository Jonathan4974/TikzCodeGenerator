# TikZ independent-mode pipeline

The pipeline downloads one Hugging Face split, creates one seeded random sample order, and runs every configured mode independently on that same order. Each mode writes its own Parquet shards. The split cache is deleted after all modes for that split are finished.

## Important change

The pipeline no longer removes LaTeX comments.

- Token counting uses the original `code_with_text`.
- VLM descriptions with text use the original code.
- Deterministic cleaning starts from the original code.
- Full cleaning sends the deterministically cleaned code, including any remaining comments, to Ollama.
- The LLM prompt treats comments as non-visible content and does not request their removal.
- The former `code_with_text_without_comments` column has been removed.

## Configuration

All settings are in `config.py`.

```python
SPLITS = {
    "our_dataset_benchmark": [
        {
            "type": "full_cleaning",
            "num": 2_500,
            "absolute_num": 5_000,
        },
        {
            "type": "simple_vlm_description",
            "num": 2_500,
            "absolute_num": 5_000,
        },
        {
            "type": "deterministic_cleaning",
            "num": 2_500,
            "absolute_num": 5_000,
        },
    ],
}
```

For every mode independently:

- `absolute_num` is the total number of selected rows for that mode output.
- `num` is the number of rows that receive active mode processing.
- The remaining `absolute_num - num` rows contain only `code_with_text`; mode-specific columns are `None`.

All modes use the same seeded random order. With equal `absolute_num` and `num`, they inspect the same samples.

## Output files

```text
/workspace/data/clean_parquets/our_dataset_benchmark/
├── our_dataset_benchmark-full_cleaning_part-00000.parquet
├── our_dataset_benchmark-simple_vlm_description_part-00000.parquet
└── our_dataset_benchmark-deterministic_cleaning_part-00000.parquet
```

Each shard contains at most 50,000 rows.

### `simple_vlm_description`

- `image_with_text`
- `code_with_text`
- `llm_description_with_text`

### `full_cleaning`

- `code_with_text`
- `image_without_text_full`
- `code_without_text_full`
- `llm_description_without_text_full`

### `deterministic_cleaning`

- `code_with_text`
- `image_without_text_deterministic`
- `code_without_text_deterministic`
- `llm_description_without_text_deterministic`

## Deterministic cleaning

The deterministic cleaner removes known text-generator commands such as `lipsum` and empties ordinary TikZ node contents. It does not remove comments.

## Validation

Active deterministic/full-cleaning rows are kept only when:

- LaTeX renders successfully,
- the generated PDF has exactly one page,
- the rendered image is not effectively white.

## Token limit

The tokenizer is loaded from:

```text
unsloth/gemma-4-31B-it-unsloth-bnb-4bit
```

The original LaTeX code may contain at most 8,000 tokens.

## Failure diagnostics

Failures are written per split and mode:

```text
/workspace/data/clean_parquets/failures/<split>-<mode>.jsonl
/workspace/data/clean_parquets/failures/<split>/<mode>/sample-000000123.txt
```

The readable text file contains the original code and the latest generated code available when the error occurred.

## Run

```bash
cd /workspace/tikz_clean_pipeline_v4
pip install -r requirements.txt
python run_pipe.py
```
