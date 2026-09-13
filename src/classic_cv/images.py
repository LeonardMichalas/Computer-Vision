"""Image loading, saving and conversion.

This is the only module that talks to OpenCV, and it only ever asks it to move
bytes between disk and a NumPy array.
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]
"""A 2-D 8-bit greyscale image, shape ``(height, width)``."""

ColourImage = NDArray[np.uint8]
"""A 3-D 8-bit BGR image, shape ``(height, width, 3)``."""

FloatImage = NDArray[np.float64]
"""An intermediate result that has not been clipped back to 0-255 yet."""


def load_grayscale(path: Path | str) -> GrayImage:
    """Read an image from disk as 8-bit greyscale.

    Raises:
        FileNotFoundError: if the file is missing or is not an image OpenCV can
            decode. OpenCV signals both by returning ``None``, which is easy to
            miss and produces a confusing error much later.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read an image from {path}")
    return np.asarray(image, dtype=np.uint8)


def load_colour(path: Path | str) -> ColourImage:
    """Read an image from disk as 8-bit BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read an image from {path}")
    return np.asarray(image, dtype=np.uint8)


def save(path: Path | str, image: NDArray[np.uint8]) -> Path:
    """Write an image to disk, creating the parent directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"could not write an image to {destination}")
    return destination


def to_uint8(image: FloatImage) -> GrayImage:
    """Clip a float result into 0-255 and round it to 8-bit.

    Use this when the values are already meant to be in range and anything
    outside it is overshoot -- a convolution result, for example.
    """
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def normalise(image: FloatImage) -> GrayImage:
    """Rescale a float result so its own minimum is 0 and its maximum is 255.

    Use this when the values have no natural range -- a gradient image, which
    is signed, or a Hough accumulator, whose peak depends on the input.
    """
    low = float(np.min(image))
    high = float(np.max(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.rint((image - low) * (255.0 / (high - low))).astype(np.uint8)
