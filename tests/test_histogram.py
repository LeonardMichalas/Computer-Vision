"""Histograms, equalisation and contrast stretching."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from classic_cv import cumulative_distribution, equalise, histogram, stretch_contrast


def test_histogram_counts_every_pixel(noise: NDArray[np.uint8]) -> None:
    counts = histogram(noise)
    assert counts.shape == (256,)
    assert counts.sum() == noise.size


def test_histogram_matches_a_known_image() -> None:
    image = np.array([[0, 0, 255], [128, 128, 128]], dtype=np.uint8)
    counts = histogram(image)
    assert counts[0] == 2
    assert counts[128] == 3
    assert counts[255] == 1
    assert counts.sum() == 6


def test_cumulative_distribution_ends_at_the_pixel_count(noise: NDArray[np.uint8]) -> None:
    cdf = cumulative_distribution(histogram(noise))
    assert cdf[-1] == noise.size
    assert np.all(np.diff(cdf) >= 0), "a running total can never fall"


def test_equalise_matches_opencv(noise: NDArray[np.uint8]) -> None:
    """The whole point of assignment 1: our lookup table against the library's."""
    np.testing.assert_array_equal(equalise(noise), cv2.equalizeHist(noise))


def test_equalise_matches_opencv_on_a_low_contrast_image(rng: np.random.Generator) -> None:
    """The case equalisation exists for: everything bunched into a narrow band."""
    hazy = rng.integers(100, 140, size=(64, 64), dtype=np.uint8)
    np.testing.assert_array_equal(equalise(hazy), cv2.equalizeHist(hazy))


def test_equalise_uses_the_full_range(rng: np.random.Generator) -> None:
    hazy = rng.integers(100, 140, size=(64, 64), dtype=np.uint8)
    result = equalise(hazy)
    assert result.min() == 0
    assert result.max() == 255


def test_equalise_leaves_a_flat_image_alone() -> None:
    """Nothing to spread out, and no division by zero either."""
    flat = np.full((8, 8), 77, dtype=np.uint8)
    np.testing.assert_array_equal(equalise(flat), flat)


def test_stretch_contrast_maps_the_ends_to_black_and_white() -> None:
    image = np.array([[50, 100, 150]], dtype=np.uint8)
    stretched = stretch_contrast(image)
    assert stretched[0, 0] == 0
    assert stretched[0, 2] == 255
    # The middle stays in the middle: the shape of the histogram is unchanged.
    assert stretched[0, 1] == pytest.approx(128, abs=1)


def test_stretch_contrast_reaches_pure_white() -> None:
    """The 2018 version divided by 256 and stopped one grey level short."""
    image = np.array([[0, 255]], dtype=np.uint8)
    assert stretch_contrast(image)[0, 1] == 255


def test_stretch_contrast_clips_instead_of_wrapping() -> None:
    """Values outside the window used to wrap around, turning shadows white."""
    image = np.array([[10, 100, 240]], dtype=np.uint8)
    stretched = stretch_contrast(image, low=50, high=200)
    assert stretched[0, 0] == 0
    assert stretched[0, 2] == 255


def test_stretch_contrast_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="greater"):
        stretch_contrast(np.zeros((4, 4), dtype=np.uint8), low=100, high=100)
