"""Convolution, written out by hand.

The loop here runs over the *kernel taps*, not over the pixels. For every one
of the kernel's entries the whole image is shifted and added in one go. That is
the same arithmetic as the textbook four-deep pixel loop and it is still the
convolution written out by hand -- it is simply the version NumPy can execute,
which turns minutes into milliseconds.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from classic_cv.images import FloatImage

Padding = Literal["zero", "edge", "reflect"]
_NumpyPadMode = Literal["constant", "edge", "reflect"]

_PAD_MODES: dict[Padding, _NumpyPadMode] = {
    "zero": "constant",
    "edge": "edge",
    "reflect": "reflect",
}


def _check_kernel(kernel: FloatImage) -> tuple[int, int]:
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2-D, got {kernel.ndim} dimensions")
    height, width = kernel.shape
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError(
            f"kernel must have odd side lengths so it has a centre, got {height}x{width}"
        )
    return height // 2, width // 2


def correlate(
    image: NDArray[np.generic], kernel: FloatImage, padding: Padding = "edge"
) -> FloatImage:
    """Slide the kernel over the image without flipping it first.

    This is what most libraries call filtering. Colour images are handled one
    channel at a time.

    Args:
        image: 2-D greyscale or 3-D colour image, any numeric dtype.
        kernel: 2-D kernel with odd side lengths.
        padding: how to invent the pixels outside the border. ``"zero"``
            reproduces the behaviour of the 2018 version of this code, which
            treated everything outside the image as black and therefore left a
            dark rim. ``"edge"`` repeats the border pixel instead, and
            ``"reflect"`` mirrors the image back on itself. In OpenCV's names
            these are BORDER_CONSTANT, BORDER_REPLICATE and BORDER_REFLECT_101
            -- note the 101: OpenCV's plain BORDER_REFLECT repeats the edge
            pixel as it mirrors, which is a fourth, different thing.

    Returns:
        A float image the same shape as the input. Values are not clipped --
        an edge kernel legitimately produces negatives.
    """
    if image.ndim == 3:
        channels = [correlate(image[:, :, c], kernel, padding) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)

    radius_y, radius_x = _check_kernel(kernel)
    source = np.asarray(image, dtype=np.float64)
    height, width = source.shape

    padded = np.pad(
        source,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode=_PAD_MODES[padding],
    )

    result = np.zeros((height, width), dtype=np.float64)
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            weight = float(kernel[row, column])
            if weight == 0.0:
                continue
            # The window of the padded image that lines up with this tap.
            result += weight * padded[row : row + height, column : column + width]

    return result


def convolve(
    image: NDArray[np.generic], kernel: FloatImage, padding: Padding = "edge"
) -> FloatImage:
    """A true convolution: the kernel is flipped in both directions first.

    For the symmetric kernels in this repository -- the box filter, a Gaussian
    -- this is identical to :func:`correlate`. For the Sobel kernels it is not,
    and the sign of the gradient depends on getting it right.
    """
    return correlate(image, kernel[::-1, ::-1], padding)


def box_kernel(size: int = 5) -> FloatImage:
    """An averaging kernel: every entry is ``1 / size**2``, so the whole sums to 1.

    Summing to 1 is what keeps the image's overall brightness unchanged. The
    2018 version built the same filter out of ones and divided afterwards,
    which is the same thing said twice.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"box kernel size must be odd and positive, got {size}")
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel(size: int = 5, sigma: float | None = None) -> FloatImage:
    """A normalised 2-D Gaussian, for smoothing that weights the centre more.

    Args:
        size: side length, odd.
        sigma: standard deviation in pixels. Defaults to the same rule OpenCV
            uses, which ties the width of the bell to the size of the kernel.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"gaussian kernel size must be odd and positive, got {size}")
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    radius = size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    line = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = np.outer(line, line)
    return kernel / float(kernel.sum())
