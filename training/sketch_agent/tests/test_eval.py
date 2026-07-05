import pytest
from PIL import Image

from training.sketch_agent.eval import pixel_congruence_coefficient


def test_pixel_congruence_coefficient_is_one_for_identical_images():
    a = Image.new("RGB", (32, 32), color=(10, 20, 30))
    b = Image.new("RGB", (32, 32), color=(10, 20, 30))
    assert pixel_congruence_coefficient(a, b) == pytest.approx(1.0, rel=1e-6)
