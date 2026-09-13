"""Histograms, equalisation and contrast stretching.

Both operations here are lookup tables. Work out one value for each of the 256
possible grey levels, then apply the table to the whole image at once -- which
is why neither function needs to loop over pixels.
"""

import numpy as np
from numpy.typing import NDArray

from classic_cv.images import GrayImage

Counts = NDArray[np.int64]


def histogram(image: GrayImage, bins: int = 256) -> Counts:
    """Count how many pixels fall into each intensity bin.

    Written with :func:`numpy.bincount` rather than a Python loop, but it is
    the same tally: walk the pixels, add one to the bin you land in.
    """
    if bins < 1 or 256 % bins != 0:
        raise ValueError(f"bins must divide 256, got {bins}")
    scale = 256 // bins
    flat = np.asarray(image, dtype=np.uint8).ravel() // scale
    return np.bincount(flat, minlength=bins).astype(np.int64)


def cumulative_distribution(counts: Counts) -> Counts:
    """The running total of a histogram: how many pixels are this dark or darker."""
    return np.cumsum(counts).astype(np.int64)


def equalise(image: GrayImage) -> GrayImage:
    """Spread the intensities out so the histogram is as flat as it can be.

    The classic formula. Take the cumulative distribution, ignore the empty
    levels below the darkest pixel present, and stretch what is left across the
    full 0-255 range:

        ``lut[v] = round(255 * (cdf[v] - cdf_min) / (pixels - cdf_min))``

    An image where the intensities are bunched into a narrow band comes out
    using the whole range, which is why a flat, hazy photograph gains contrast.
    ``cv2.equalizeHist`` computes the same table; the test suite checks the two
    against each other.
    """
    source = np.asarray(image, dtype=np.uint8)
    counts = histogram(source)
    cdf = cumulative_distribution(counts)

    total = int(cdf[-1])
    if total == 0:
        return source.copy()

    # The first non-empty level. Subtracting it is what pins the darkest pixel
    # present to 0 instead of leaving a gap at the bottom of the range.
    occupied = cdf[counts > 0]
    cdf_min = int(occupied[0])
    if total == cdf_min:
        # Every pixel has the same value; there is nothing to spread out.
        return source.copy()

    lut = np.rint((cdf - cdf_min) * (255.0 / (total - cdf_min))).astype(np.uint8)
    return lut[source]


def stretch_contrast(
    image: GrayImage, low: int | None = None, high: int | None = None
) -> GrayImage:
    """Linearly rescale the intensity range so ``low`` becomes 0 and ``high`` becomes 255.

    The simpler cousin of :func:`equalise`: it moves the two ends of the range
    without changing the shape of the histogram in between.

    This is the function that was marked ``NOT WORKING AS IT SHOULD`` in the
    2018 version, for three reasons, all fixed here. It divided by 256 rather
    than 255, so pure white came out one short. It wrote float results into a
    ``uint8`` array. And it never clipped, so a pixel outside ``[low, high]``
    wrapped around and a shadow came back bright white.

    Args:
        image: the greyscale image to stretch.
        low: intensity mapped to 0. Defaults to the image's darkest pixel.
        high: intensity mapped to 255. Defaults to the image's brightest pixel.
    """
    source = np.asarray(image, dtype=np.uint8)
    low = int(np.min(source)) if low is None else low
    high = int(np.max(source)) if high is None else high

    if high <= low:
        raise ValueError(f"high must be greater than low, got low={low} high={high}")

    levels = np.arange(256, dtype=np.float64)
    lut = np.clip(np.rint((levels - low) * (255.0 / (high - low))), 0, 255).astype(np.uint8)
    return lut[source]
