"""Tests for the colour-only baseline.

The baseline's whole argument rests on it being genuinely trivial — six numbers,
no spatial information. If it ever accidentally gained access to structure, it
would stop being evidence about what the CNN is doing.
"""

import pathlib
import tempfile

import numpy as np
import pytest
from PIL import Image

from baseline import THUMBNAIL, colour_features


def write_image(tmp_path, name, colour, size=(64, 64)):
    path = tmp_path / name
    Image.new("RGB", size, colour).save(path)
    return path


def test_exactly_six_features():
    """Three channel means and three channel standard deviations. No more."""
    with tempfile.TemporaryDirectory() as directory:
        path = write_image(pathlib.Path(directory), "a.jpg", (10, 20, 30))
        assert colour_features(path).shape == (6,)


def test_a_flat_image_has_zero_variance_and_its_own_colour(tmp_path):
    path = write_image(tmp_path, "flat.png", (10, 20, 30))
    features = colour_features(path)

    assert features[:3] == pytest.approx([10, 20, 30], abs=1.0)
    assert features[3:] == pytest.approx([0, 0, 0], abs=1.0)


def test_spatial_arrangement_is_discarded(tmp_path):
    """The same pixels rearranged must give identical features — this is what
    makes the baseline evidence about staining rather than morphology."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (THUMBNAIL, THUMBNAIL, 3), dtype=np.uint8)

    original = tmp_path / "a.png"
    Image.fromarray(pixels).save(original)

    shuffled = pixels.reshape(-1, 3)
    rng.shuffle(shuffled)
    rotated = tmp_path / "b.png"
    Image.fromarray(shuffled.reshape(THUMBNAIL, THUMBNAIL, 3)).save(rotated)

    assert colour_features(original) == pytest.approx(colour_features(rotated), abs=1.5)


def test_a_greyscale_source_is_read_as_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (32, 32), 128).save(path)
    features = colour_features(path)

    assert features[:3] == pytest.approx([128, 128, 128], abs=1.0)


def test_features_are_deterministic(tmp_path):
    path = write_image(tmp_path, "same.jpg", (77, 88, 99))
    assert colour_features(path) == pytest.approx(colour_features(path))


def test_differently_stained_images_are_separable_by_these_features(tmp_path):
    """The premise of the finding: a global colour shift alone moves the vector."""
    pale = colour_features(write_image(tmp_path, "pale.png", (220, 200, 215)))
    dark = colour_features(write_image(tmp_path, "dark.png", (120, 90, 110)))

    assert np.linalg.norm(pale[:3] - dark[:3]) > 50
