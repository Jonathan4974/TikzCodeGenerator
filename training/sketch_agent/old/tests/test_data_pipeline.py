import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from training.sketch_agent.config import build_training_config
from training.sketch_agent.data import (
    SketchAgentDataset,
    generate_synthetic_pairs,
    iter_datikz_renders,
    load_synthetic_dataset,
    sketch_to_canny,
)
from training.sketch_agent.real_data import load_sketchfig_dataset
from training.sketch_agent.ultrasketch_methods import assert_multiple_of, random_displacement_field, resize_to_multiple


def _solid_image(size: int = 64, color=(120, 140, 160)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def _fake_datikz_renders(
    n: int, seed=None, dataset_name=None, split=None, streaming=None, start_index: int = 0, buffer_size=None, size: int = 64
):
    for offset in range(n):
        idx = start_index + offset
        yield f"fake_{idx:03d}", _solid_image(size, color=(idx % 255, 10, 20))


class _FakeUltraSketchPipe:
    """Stand-in with the same call interface as the real diffusers pipeline, used only
    in tests so `run_ultrasketch`'s resize/assert plumbing can be tested without a GPU."""

    device = "cpu"

    def __init__(self):
        self.seen_generator_seeds = []

    def __call__(self, prompt, image, mask_img=None, generator=None, **kwargs):
        self.seen_generator_seeds.append(generator.initial_seed() if generator is not None else None)
        sketch = image.convert("L").point(lambda value: 255 if value > 220 else 0).convert("RGB")
        return SimpleNamespace(images=[sketch])


def test_sketch_to_canny_produces_three_channel_binary_edge_map():
    image = _gradient_image(64)
    canny = sketch_to_canny(image, low_threshold=100, high_threshold=200)
    arr = np.array(canny)

    assert arr.shape == (64, 64, 3)
    assert set(np.unique(arr).tolist()) <= {0, 255}
    assert np.array_equal(arr[..., 0], arr[..., 1])
    assert np.array_equal(arr[..., 1], arr[..., 2])


def test_sketch_agent_dataset_raises_when_no_pairs_available():
    cfg = build_training_config({"use_synthetic_data": False})
    with pytest.raises(ValueError):
        SketchAgentDataset(cfg, extra_pairs=[])


def test_resize_to_multiple_rounds_down_and_is_idempotent():
    image = Image.new("RGB", (70, 50))
    resized = resize_to_multiple(image, multiple=16)
    assert resized.size == (64, 48)
    assert_multiple_of(resized, "resized")
    assert resize_to_multiple(resized, multiple=16).size == resized.size


def _gradient_image(size: int = 64) -> Image.Image:
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    array = np.stack([(x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(array)


def test_random_displacement_field_is_deterministic_per_seed():
    image = _gradient_image(64)
    out_a = random_displacement_field(image, alpha=6, sigma=12, seed=42)
    out_b = random_displacement_field(image, alpha=6, sigma=12, seed=42)
    out_c = random_displacement_field(image, alpha=6, sigma=12, seed=43)

    assert np.array_equal(np.array(out_a), np.array(out_b))
    assert not np.array_equal(np.array(out_a), np.array(out_c))
    assert out_a.size == image.size


def test_generate_synthetic_pairs_assigns_methods_matching_rng_draws():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.iter_datikz_renders", side_effect=_fake_datikz_renders
    ), patch("training.sketch_agent.data.load_ultrasketch_pipeline", return_value=_FakeUltraSketchPipe()):
        seed = 7
        num_samples = 6
        pairs = generate_synthetic_pairs(tmp, num_samples=num_samples, seed=seed, ultrasketch_probability=0.5)

        assert len(pairs) == num_samples

        rng = np.random.default_rng(seed)
        expected_methods = ["ultrasketch" if rng.random() < 0.5 else "displacement" for _ in range(num_samples)]
        assert [pair.method for pair in pairs] == expected_methods

        # Regression check: indices must be contiguous (0..num_samples-1), not skip
        # (a prior bug computed `idx` from the growing `pairs` list mid-loop, producing
        # 0, 2, 4, 6, ... which silently broke prediction/target filename matching).
        stems = [pair.input_path.stem for pair in pairs]
        assert stems == [f"sample_{i:03d}_input" for i in range(num_samples)]

        manifest_path = Path(tmp) / "manifest.jsonl"
        assert manifest_path.exists()
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        assert len(manifest_lines) == num_samples
        assert [json.loads(line)["idx"] for line in manifest_lines] == list(range(num_samples))


def test_generate_synthetic_pairs_ultrasketch_calls_use_a_per_sample_seeded_generator():
    # Regression check: run_ultrasketch used to be called with no generator at all,
    # relying on whatever the ambient global torch RNG state happened to be - not
    # reproducible on its own. Each sample must now get its own deterministic
    # seed=cfg.seed+idx generator, matching random_displacement_field's convention.
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.iter_datikz_renders", side_effect=_fake_datikz_renders
    ):
        pipe = _FakeUltraSketchPipe()
        with patch("training.sketch_agent.data.load_ultrasketch_pipeline", return_value=pipe):
            generate_synthetic_pairs(tmp, num_samples=4, seed=3, ultrasketch_probability=1.0)

    assert pipe.seen_generator_seeds == [3, 4, 5, 6]


def test_generate_synthetic_pairs_probability_zero_is_all_displacement():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.iter_datikz_renders", side_effect=_fake_datikz_renders
    ), patch("training.sketch_agent.data.load_ultrasketch_pipeline", return_value=_FakeUltraSketchPipe()):
        pairs = generate_synthetic_pairs(tmp, num_samples=3, seed=1, ultrasketch_probability=0.0)
        assert all(pair.method == "displacement" for pair in pairs)


def test_generate_synthetic_pairs_probability_one_uses_stub_only():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.iter_datikz_renders", side_effect=_fake_datikz_renders
    ), patch("training.sketch_agent.data.load_ultrasketch_pipeline", return_value=_FakeUltraSketchPipe()):
        pairs = generate_synthetic_pairs(tmp, num_samples=3, seed=1, ultrasketch_probability=1.0)
        assert all(pair.method == "ultrasketch" for pair in pairs)


def test_generate_synthetic_pairs_is_incremental():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.load_ultrasketch_pipeline", return_value=_FakeUltraSketchPipe()
    ):
        calls = []

        def counting_render_source(n, seed=None, dataset_name=None, split=None, streaming=None, start_index: int = 0, buffer_size=None):
            calls.append((n, start_index))
            return _fake_datikz_renders(n, start_index=start_index)

        with patch("training.sketch_agent.data.iter_datikz_renders", side_effect=counting_render_source):
            generate_synthetic_pairs(tmp, num_samples=2, seed=5)
            pairs = generate_synthetic_pairs(tmp, num_samples=5, seed=5)

        assert calls == [(2, 0), (3, 2)]
        assert len(pairs) == 5
        assert [pair.input_path.stem for pair in pairs] == [f"sample_{i:03d}_input" for i in range(5)]
        # Regression check: topped-up renders must use fresh source_names (start_index
        # advanced), not repeat the first batch's names/content from scratch.
        assert [pair.source_name for pair in pairs] == [f"fake_{i:03d}" for i in range(5)]


def test_iter_datikz_renders_shuffle_order_is_independent_of_num_samples_requested():
    # Regression check: buffer_size used to be derived from the per-call `num_samples`
    # (max(num_samples * 10, 1000)), and HF `datasets`' streaming .shuffle(seed, buffer_size)
    # produces a different row sequence for a different buffer_size even with the same seed.
    # That meant "same seed" did not mean "same data" once num_samples changed - e.g. a
    # small incremental top-up call diverged from what the original larger call produced.
    # buffer_size is now fixed (config.datikz_shuffle_buffer_size), so a small request must
    # be a true prefix of a larger request at the same seed.
    from datasets import Dataset

    n_rows = 1600
    fake_rows = {
        "file_id": [f"id_{i:04d}" for i in range(n_rows)],
        "png_image": [_solid_image(2, color=(i % 255, 0, 0)) for i in range(n_rows)],
    }

    def _fake_load_dataset(*args, **kwargs):
        return Dataset.from_dict(fake_rows).to_iterable_dataset()

    with patch("datasets.load_dataset", side_effect=_fake_load_dataset):
        small_request = [file_id for file_id, _ in iter_datikz_renders(20, seed=7)]
        large_request = [file_id for file_id, _ in iter_datikz_renders(1500, seed=7)]

    assert small_request == large_request[:20]


def test_load_synthetic_dataset_reports_real_method_per_pair():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp, patch(
        "training.sketch_agent.data.iter_datikz_renders", side_effect=_fake_datikz_renders
    ), patch("training.sketch_agent.data.load_ultrasketch_pipeline", return_value=_FakeUltraSketchPipe()):
        generate_synthetic_pairs(tmp, num_samples=4, seed=9, ultrasketch_probability=0.5)
        reloaded = load_synthetic_dataset(tmp)
        assert len(reloaded) == 4
        assert all(pair.method in {"ultrasketch", "displacement"} for pair in reloaded)


def test_load_synthetic_dataset_without_manifest_reports_unknown_method():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-data-") as tmp:
        base = Path(tmp)
        (base / "inputs").mkdir(parents=True)
        (base / "targets").mkdir(parents=True)
        _solid_image().save(base / "inputs" / "sample_000_input.png")
        _solid_image().save(base / "targets" / "sample_000_target.png")

        pairs = load_synthetic_dataset(tmp)
        assert len(pairs) == 1
        assert pairs[0].method == "unknown"


def _fake_sketchfig_rows(num_rows: int = 10):
    return [
        {
            "sketch": _solid_image(32, color=(i, 0, 0)),
            "image": _solid_image(32, color=(0, i, 0)),
            "uri": f"https://example.com/q/{i}",
        }
        for i in range(num_rows)
    ]


def test_load_sketchfig_dataset_default_holds_out_everything_for_eval():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-sketchfig-") as tmp:
        rows = _fake_sketchfig_rows(10)
        with patch("datasets.load_dataset", return_value=rows):
            split = load_sketchfig_dataset(tmp)

        assert len(split.train) == 0
        assert len(split.eval) == 10
        for pair in split.eval:
            assert Image.open(pair.input_path).size == (32, 32)
            assert Image.open(pair.target_path).size == (32, 32)


def test_load_sketchfig_dataset_split_is_reproducible_and_disjoint():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-sketchfig-") as tmp:
        rows = _fake_sketchfig_rows(20)
        with patch("datasets.load_dataset", return_value=rows):
            split_a = load_sketchfig_dataset(tmp + "/a", train_fraction=0.3, seed=11)
            split_b = load_sketchfig_dataset(tmp + "/b", train_fraction=0.3, seed=11)

        train_sources_a = {pair.source_name for pair in split_a.train}
        eval_sources_a = {pair.source_name for pair in split_a.eval}
        train_sources_b = {pair.source_name for pair in split_b.train}

        assert len(split_a.train) == 6
        assert len(split_a.eval) == 14
        assert train_sources_a.isdisjoint(eval_sources_a)
        assert train_sources_a == train_sources_b

        split_json = Path(tmp + "/a") / "sketchfig_split.json"
        assert split_json.exists()
