"""Sobel edge detection.

An edge is a place where brightness changes fast. "Fast" means a large
derivative, and the Sobel kernels are a three-pixel-wide approximation of the
derivative that smooths along the edge while it differentiates across it.

One convention to be careful about: the kernels below are written to be
*correlated* with the image, not convolved. That is what OpenCV, MATLAB and
every textbook figure do, and it is what makes a dark-to-light step come out
positive. Running a true convolution instead flips the kernel and negates both
gradients -- not wrong, but the signs then mean the opposite of what everyone
else's do. The test suite pins this down against ``cv2.Sobel``.
"""

from typing import Literal

import numpy as np

from classic_cv.convolution import box_kernel, correlate
from classic_cv.images import FloatImage, GrayImage

SOBEL_X: FloatImage = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
)
"""Responds to vertical edges -- brightness changing along the x axis."""

SOBEL_Y: FloatImage = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
)
"""Responds to horizontal edges -- brightness changing along the y axis."""

Norm = Literal["l1", "l2"]


def sobel_gradients(image: GrayImage) -> tuple[FloatImage, FloatImage]:
    """Return the x and y derivatives of the image.

    Both are signed: a dark-to-light edge and a light-to-dark edge have the
    same strength and opposite sign, which is why they are kept as floats
    rather than being squeezed back into ``uint8`` here.

    Positive means brightness increasing to the right (for x) and downward
    (for y), matching ``cv2.Sobel``. See the note in the module docstring.
    """
    return correlate(image, SOBEL_X), correlate(image, SOBEL_Y)


def gradient_magnitude(
    gradient_x: FloatImage, gradient_y: FloatImage, norm: Norm = "l2"
) -> FloatImage:
    """Combine the two derivatives into one edge-strength image.

    Args:
        gradient_x: derivative along x.
        gradient_y: derivative along y.
        norm: ``"l2"`` is the true magnitude, ``sqrt(gx**2 + gy**2)``.
            ``"l1"`` is ``|gx| + |gy|``, the cheap approximation the 2018
            version used. It overstates diagonal edges by up to 41%.
    """
    if norm == "l1":
        return np.abs(gradient_x) + np.abs(gradient_y)
    return np.hypot(gradient_x, gradient_y)


def sobel_edges(
    image: GrayImage,
    threshold: float = 90.0,
    smooth: int = 3,
    norm: Norm = "l2",
) -> GrayImage:
    """The whole pipeline: smooth, differentiate, combine, threshold.

    Smoothing first is not optional in practice. A derivative amplifies noise,
    so without it every sensor speckle becomes an edge.

    Args:
        image: greyscale input.
        threshold: edge strength above which a pixel is called an edge.
        smooth: side length of the box filter applied first. Pass 0 to skip it.
        norm: how to combine the two gradients, see :func:`gradient_magnitude`.

    Returns:
        A binary image: 255 where there is an edge, 0 everywhere else.
    """
    prepared: FloatImage | GrayImage = image
    if smooth:
        prepared = correlate(image, box_kernel(smooth))

    gradient_x, gradient_y = sobel_gradients(np.asarray(prepared))
    magnitude = gradient_magnitude(gradient_x, gradient_y, norm)

    return np.where(magnitude > threshold, 255, 0).astype(np.uint8)
