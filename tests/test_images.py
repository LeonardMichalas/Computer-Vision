"""Loading, saving and the two ways of getting back to 8 bits."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from classic_cv import load_grayscale, normalise, save, to_uint8


def test_round_trip_through_disk(tmp_path: Path, noise: NDArray[np.uint8]) -> None:
    written = save(tmp_path / "nested" / "noise.png", noise)
    np.testing.assert_array_equal(load_grayscale(written), noise)


def test_missing_file_raises_instead_of_returning_none(tmp_path: Path) -> None:
    """OpenCV returns None, which fails confusingly much later."""
    with pytest.raises(FileNotFoundError):
        load_grayscale(tmp_path / "not-here.png")


def test_to_uint8_clips_overshoot() -> None:
    values = np.array([[-40.0, 0.4, 127.5, 255.0, 300.0]])
    np.testing.assert_array_equal(to_uint8(values), [[0, 0, 128, 255, 255]])


def test_normalise_stretches_to_the_full_range() -> None:
    """A signed gradient image comes back with its own min at 0 and max at 255."""
    values = np.array([[-100.0, 0.0, 100.0]])
    result = normalise(values)

    assert result[0, 0] == 0
    assert result[0, 2] == 255
    assert result[0, 1] == pytest.approx(128, abs=1), "zero lands in the middle grey"


def test_normalise_handles_a_constant_image() -> None:
    """No range to stretch, and no division by zero."""
    np.testing.assert_array_equal(normalise(np.full((3, 3), 7.0)), np.zeros((3, 3)))
