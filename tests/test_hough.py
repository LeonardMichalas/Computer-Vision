"""The Hough transform, checked against lines whose parameters are known."""

import numpy as np
import pytest
from numpy.typing import NDArray

from classic_cv import draw_lines, find_peaks, hough_transform, threshold_mask


def blank(size: int = 101) -> NDArray[np.bool_]:
    return np.zeros((size, size), dtype=bool)


def test_vertical_line_is_found_at_theta_zero() -> None:
    """A vertical line has a horizontal normal: theta = 0, rho = its x."""
    mask = blank()
    mask[:, 30] = True

    lines = find_peaks(hough_transform(mask), count=1)

    assert len(lines) == 1
    assert lines[0].theta == pytest.approx(0.0, abs=0.02)
    assert lines[0].rho == pytest.approx(30.0, abs=1.0)


def test_horizontal_line_is_found_at_theta_ninety_degrees() -> None:
    mask = blank()
    mask[40, :] = True

    lines = find_peaks(hough_transform(mask), count=1)

    assert lines[0].theta == pytest.approx(np.pi / 2, abs=0.02)
    assert lines[0].rho == pytest.approx(40.0, abs=1.0)


def test_diagonal_line_is_found() -> None:
    """y = x runs through the origin, so rho is 0 and theta is 135 degrees."""
    mask = blank()
    np.fill_diagonal(mask, True)

    lines = find_peaks(hough_transform(mask, theta_bins=360), count=1)

    assert lines[0].theta == pytest.approx(3 * np.pi / 4, abs=0.02)
    assert lines[0].rho == pytest.approx(0.0, abs=1.5)


def test_the_peak_collects_one_vote_per_pixel_on_the_line() -> None:
    """The height of a peak is the length of the line that made it."""
    mask = blank()
    mask[:, 30] = True

    lines = find_peaks(hough_transform(mask), count=1)

    assert lines[0].votes == mask.shape[0]


def test_two_lines_are_found_separately() -> None:
    mask = blank()
    mask[:, 20] = True
    mask[70, :] = True

    lines = find_peaks(hough_transform(mask), count=2)
    found = sorted((round(line.rho), round(line.theta, 2)) for line in lines)

    assert found == [(20, 0.0), (70, round(np.pi / 2, 2))]


def test_suppression_stops_one_line_winning_every_slot() -> None:
    """Without it, the cells either side of a peak take the remaining slots."""
    mask = blank()
    mask[:, 30] = True
    space = hough_transform(mask)

    suppressed = find_peaks(space, count=5, suppression_radius=10)
    unsuppressed = find_peaks(space, count=5, suppression_radius=0)

    assert len(suppressed) < len(unsuppressed) or suppressed[1].votes < unsuppressed[1].votes


def test_an_empty_mask_finds_nothing() -> None:
    space = hough_transform(blank())
    assert space.accumulator.sum() == 0
    assert find_peaks(space, count=5) == []


def test_accumulator_shape_follows_the_bin_counts() -> None:
    mask = blank(50)
    space = hough_transform(mask, theta_bins=90, rho_bins=201)

    assert space.accumulator.shape == (201, 90)
    assert space.thetas.shape == (90,)
    assert space.rhos.shape == (201,)
    assert space.thetas[0] == 0.0
    assert space.thetas[-1] < np.pi


def test_threshold_mask_selects_dark_or_bright() -> None:
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(threshold_mask(image, 0, below=True), [[True, False, False]])
    np.testing.assert_array_equal(threshold_mask(image, 255, below=False), [[False, False, True]])


def test_draw_lines_returns_a_colour_image() -> None:
    image = np.full((50, 50), 255, dtype=np.uint8)
    mask = blank(50)
    mask[:, 25] = True

    drawn = draw_lines(image, find_peaks(hough_transform(mask), count=1))

    assert drawn.shape == (50, 50, 3)
    assert drawn.dtype == np.uint8
    # Something red was actually painted where the line is.
    column = drawn[:, 23:28]
    assert np.any((column[:, :, 2] > 150) & (column[:, :, 1] < 100))


def test_rejects_a_three_dimensional_mask() -> None:
    with pytest.raises(ValueError, match="2-D"):
        hough_transform(np.zeros((10, 10, 3), dtype=bool))
