import tempfile
from pathlib import Path

import pytest
from PIL import Image

from training.sketch_agent.eval import (
    dreamsim_similarity,
    evaluate_generated_outputs,
    siglip_similarity,
)


def _make_pair_dir(tmp: str):
    pred_dir = Path(tmp) / "predictions"
    ref_dir = Path(tmp) / "targets"
    pred_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(pred_dir / "sample_000_pred.png")
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(ref_dir / "sample_000_target.png")
    return pred_dir, ref_dir


def test_evaluate_generated_outputs_pixel_cc_is_ok_without_torch():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-eval-") as tmp:
        pred_dir, ref_dir = _make_pair_dir(tmp)
        results = evaluate_generated_outputs(pred_dir, ref_dir, metrics=("pixel_cc",))

        assert results["count"] == 1
        assert results["pixel_cc"]["status"] == "ok"
        assert results["pixel_cc"]["mean"] == pytest.approx(1.0, rel=1e-6)


def test_evaluate_generated_outputs_reports_unavailable_metrics_without_raising():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-eval-") as tmp:
        pred_dir, ref_dir = _make_pair_dir(tmp)
        results = evaluate_generated_outputs(pred_dir, ref_dir, metrics=("pixel_cc", "siglip", "dreamsim"))

        assert results["pixel_cc"]["status"] == "ok"
        assert results["siglip"]["status"] == "unavailable"
        assert results["siglip"]["mean"] is None
        assert results["dreamsim"]["status"] == "unavailable"


def test_evaluate_generated_outputs_rejects_unknown_metric():
    with tempfile.TemporaryDirectory(prefix="sketch-agent-eval-") as tmp:
        pred_dir, ref_dir = _make_pair_dir(tmp)
        with pytest.raises(ValueError):
            evaluate_generated_outputs(pred_dir, ref_dir, metrics=("not_a_metric",))


def test_siglip_similarity_uses_injected_fn():
    calls = []

    def fake_fn(image_a, image_b, model_key=None):
        calls.append((str(image_a), str(image_b), model_key))
        return 0.42

    score = siglip_similarity("a.png", "b.png", similarity_fn=fake_fn)
    assert score == pytest.approx(0.42)
    assert calls == [("a.png", "b.png", "siglip")]


def test_dreamsim_similarity_uses_injected_fn():
    calls = []

    def fake_fn(image_a, image_b):
        calls.append((image_a, image_b))
        return 0.77

    score = dreamsim_similarity("a.png", "b.png", similarity_fn=fake_fn)
    assert score == pytest.approx(0.77)
    assert calls == [("a.png", "b.png")]
