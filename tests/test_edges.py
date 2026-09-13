"""Sobel edge detection."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from classic_cv import (
    SOBEL_X,
    SOBEL_Y,
    convolve,
    correlate,
    gradient_magnitude,
    sobel_edges,
    sobel_gradients,
)


def test_sobel_x_matches_opencv(noise: NDArray[np.uint8]) -> None:
    """cv2.Sobel uses the same kernel, so the two must agree pixel for pixel.

    BORDER_REFLECT_101 is the mode NumPy calls "reflect"; OpenCV's plain
    BORDER_REFLECT repeats the edge pixel instead and disagrees at the rim.
    """
    ours = correlate(noise, SOBEL_X, padding="reflect")
    theirs = cv2.Sobel(noise, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_sobel_y_matches_opencv(noise: NDArray[np.uint8]) -> None:
    ours = correlate(noise, SOBEL_Y, padding="reflect")
    theirs = cv2.Sobel(noise, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_convolving_the_sobel_kernel_negates_the_gradient(noise: NDArray[np.uint8]) -> None:
    """The convention trap: the kernels are written to be correlated.

    A true convolution flips them, and because they are antisymmetric the
    result comes back with the opposite sign. Same edges, inverted meaning.
    """
    correlated = correlate(noise, SOBEL_X, padding="reflect")
    convolved = convolve(noise, SOBEL_X, padding="reflect")
    np.testing.assert_allclose(convolved, -correlated, atol=1e-9)


def test_vertical_edge_shows_in_x_not_y(vertical_edge: NDArray[np.uint8]) -> None:
    gradient_x, gradient_y = sobel_gradients(vertical_edge)
    assert np.abs(gradient_x).max() > 500
    assert np.abs(gradient_y).max() == pytest.approx(0.0, abs=1e-9)


def test_horizontal_edge_shows_in_y_not_x(horizontal_edge: NDArray[np.uint8]) -> None:
    gradient_x, gradient_y = sobel_gradients(horizontal_edge)
    assert np.abs(gradient_y).max() > 500
    assert np.abs(gradient_x).max() == pytest.approx(0.0, abs=1e-9)


def test_gradient_sign_follows_the_direction_of_the_step() -> None:
    """Dark to light and light to dark are the same edge with opposite signs."""
    rising = np.zeros((10, 10), dtype=np.uint8)
    rising[:, 5:] = 255
    falling = 255 - rising

    gradient_rising, _ = sobel_gradients(rising)
    gradient_falling, _ = sobel_gradients(falling)

    np.testing.assert_allclose(gradient_rising, -gradient_falling)


def test_flat_image_has_no_edges() -> None:
    flat = np.full((20, 20), 120, dtype=np.uint8)
    assert np.count_nonzero(sobel_edges(flat)) == 0


def test_l1_never_understates_l2() -> None:
    """|gx| + |gy| >= hypot(gx, gy), with equality only on the axes."""
    gradient_x = np.array([[3.0, 5.0, 0.0]])
    gradient_y = np.array([[4.0, 0.0, 0.0]])

    l1 = gradient_magnitude(gradient_x, gradient_y, "l1")
    l2 = gradient_magnitude(gradient_x, gradient_y, "l2")

    assert np.all(l1 >= l2 - 1e-9)
    assert l2[0, 0] == pytest.approx(5.0)
    assert l1[0, 0] == pytest.approx(7.0)
    assert l1[0, 1] == pytest.approx(l2[0, 1]), "a pure x gradient agrees"


def test_edges_are_binary(vertical_edge: NDArray[np.uint8]) -> None:
    edges = sobel_edges(vertical_edge, threshold=90)
    assert set(np.unique(edges)) <= {0, 255}
    assert np.count_nonzero(edges) > 0


def test_a_higher_threshold_finds_fewer_edges(noise: NDArray[np.uint8]) -> None:
    strict = np.count_nonzero(sobel_edges(noise, threshold=300))
    loose = np.count_nonzero(sobel_edges(noise, threshold=50))
    assert strict < loose


def test_smoothing_suppresses_noise(rng: np.random.Generator) -> None:
    """The reason step one of the pipeline exists."""
    speckled = rng.integers(0, 256, size=(60, 60), dtype=np.uint8)
    raw = np.count_nonzero(sobel_edges(speckled, threshold=200, smooth=0))
    smoothed = np.count_nonzero(sobel_edges(speckled, threshold=200, smooth=5))
    assert smoothed < raw
