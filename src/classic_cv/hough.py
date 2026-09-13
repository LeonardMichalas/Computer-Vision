"""The Hough transform: finding straight lines by voting.

The idea is a change of coordinates. A line is written not as ``y = mx + c``,
which cannot express a vertical line, but as the pair ``(rho, theta)``: the
angle of the line's normal, and the distance from the origin along it.

    ``rho = x * cos(theta) + y * sin(theta)``

Every marked pixel votes for every line that could pass through it -- one vote
for each angle. Where many pixels lie on the same real line, their votes land
in the same cell of the accumulator and pile up. Finding lines in the image
becomes finding peaks in the accumulator, which is a much easier problem.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from classic_cv.images import ColourImage, GrayImage

Accumulator = NDArray[np.int64]


@dataclass(frozen=True)
class HoughSpace:
    """The accumulator and the axes needed to read it back."""

    accumulator: Accumulator
    """Votes, indexed ``[rho_index, theta_index]``."""

    thetas: NDArray[np.float64]
    """The angle each column stands for, in radians, from 0 to pi."""

    rhos: NDArray[np.float64]
    """The distance each row stands for, in pixels, from -diagonal to +diagonal."""


@dataclass(frozen=True)
class Line:
    """One detected line, in normal form."""

    rho: float
    theta: float
    votes: int


def threshold_mask(image: GrayImage, threshold: int = 0, below: bool = True) -> NDArray[np.bool_]:
    """Pick the pixels that are allowed to vote.

    Args:
        image: greyscale input.
        threshold: the cut-off.
        below: ``True`` selects pixels at or below the threshold, which is what
            you want for dark ink on white paper. ``False`` selects pixels at
            or above it, which is what you want for a white-on-black edge map
            out of :func:`sobel_edges`.
    """
    source = np.asarray(image, dtype=np.int32)
    return source <= threshold if below else source >= threshold


def hough_transform(
    mask: NDArray[np.bool_], theta_bins: int = 180, rho_bins: int | None = None
) -> HoughSpace:
    """Accumulate votes from every marked pixel in ``mask``.

    The votes are counted with :func:`numpy.bincount` over flattened cell
    indices. That is the same tally as incrementing one accumulator cell per
    pixel per angle, done in one pass instead of tens of millions of them.

    Args:
        mask: boolean image, ``True`` where a pixel votes.
        theta_bins: how finely the angle is sampled between 0 and pi. More bins
            locate a line's angle more precisely but spread its votes thinner.
        rho_bins: how finely the distance is sampled. Defaults to roughly one
            bin per pixel of the image diagonal, in each direction.

    Raises:
        ValueError: if the mask is not 2-D, or the bin counts are not positive.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.ndim} dimensions")
    if theta_bins < 1:
        raise ValueError(f"theta_bins must be positive, got {theta_bins}")

    height, width = mask.shape
    diagonal = float(np.hypot(height, width))
    if rho_bins is None:
        rho_bins = 2 * int(np.ceil(diagonal)) + 1
    if rho_bins < 1:
        raise ValueError(f"rho_bins must be positive, got {rho_bins}")

    thetas = np.linspace(0.0, np.pi, theta_bins, endpoint=False)
    rhos = np.linspace(-diagonal, diagonal, rho_bins)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return HoughSpace(np.zeros((rho_bins, theta_bins), dtype=np.int64), thetas, rhos)

    # One rho per (voting pixel, angle) pair: an (N, theta_bins) table.
    rho_values = np.outer(xs, np.cos(thetas)) + np.outer(ys, np.sin(thetas))

    # Which row of the accumulator each rho falls in.
    rho_indices = np.rint((rho_values + diagonal) * ((rho_bins - 1) / (2.0 * diagonal)))
    rho_indices = np.clip(rho_indices, 0, rho_bins - 1).astype(np.int64)

    theta_indices = np.broadcast_to(np.arange(theta_bins, dtype=np.int64), rho_indices.shape)

    flat = (rho_indices * theta_bins + theta_indices).ravel()
    counts = np.bincount(flat, minlength=rho_bins * theta_bins)

    return HoughSpace(counts.reshape(rho_bins, theta_bins).astype(np.int64), thetas, rhos)


def find_peaks(space: HoughSpace, count: int = 9, suppression_radius: int = 10) -> list[Line]:
    """Pull the strongest peaks out of the accumulator, one at a time.

    After each peak is taken its neighbourhood is zeroed. Without that step a
    single strong line wins every slot, because the cells either side of a peak
    are nearly as tall as the peak itself. This is non-maximum suppression, and
    ``suppression_radius`` is the claim that two lines this close are one line.

    Args:
        space: the accumulator to search.
        count: how many lines to return at most.
        suppression_radius: how many cells around each peak to clear.

    Returns:
        Lines in descending vote order. Shorter than ``count`` if the
        accumulator runs out of non-zero cells.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    if suppression_radius < 0:
        raise ValueError(f"suppression_radius must not be negative, got {suppression_radius}")

    remaining = space.accumulator.copy()
    rho_bins, theta_bins = remaining.shape
    lines: list[Line] = []

    for _ in range(count):
        flat_index = int(np.argmax(remaining))
        votes = int(remaining.flat[flat_index])
        if votes <= 0:
            break

        rho_index, theta_index = divmod(flat_index, theta_bins)
        lines.append(
            Line(
                rho=float(space.rhos[rho_index]),
                theta=float(space.thetas[theta_index]),
                votes=votes,
            )
        )

        top = max(0, rho_index - suppression_radius)
        bottom = min(rho_bins, rho_index + suppression_radius + 1)
        left = max(0, theta_index - suppression_radius)
        right = min(theta_bins, theta_index + suppression_radius + 1)
        remaining[top:bottom, left:right] = 0

    return lines


def draw_lines(
    image: GrayImage | ColourImage,
    lines: list[Line],
    colour: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> ColourImage:
    """Draw detected lines back onto a copy of the image, in colour.

    Each ``(rho, theta)`` is turned back into two points far outside the frame
    and joined, so the line is drawn across the whole image rather than only
    where the votes came from.
    """
    source = np.asarray(image, dtype=np.uint8)
    canvas = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) if source.ndim == 2 else source.copy()

    reach = 2 * int(np.hypot(*canvas.shape[:2]))

    for line in lines:
        cos_theta, sin_theta = np.cos(line.theta), np.sin(line.theta)
        # The point on the line closest to the origin.
        x0, y0 = cos_theta * line.rho, sin_theta * line.rho
        # Step away from it along the line, which runs perpendicular to the normal.
        start = (round(x0 - reach * sin_theta), round(y0 + reach * cos_theta))
        end = (round(x0 + reach * sin_theta), round(y0 - reach * cos_theta))
        cv2.line(canvas, start, end, colour, thickness, cv2.LINE_AA)

    return np.asarray(canvas, dtype=np.uint8)
