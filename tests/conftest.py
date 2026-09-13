"""Shared fixtures.

The images here are generated, not loaded, so the tests do not depend on the
sample pictures in the assignment folders and a failure points at the algorithm
rather than at a file.
"""

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed, so a failure can be reproduced."""
    return np.random.default_rng(20180626)


@pytest.fixture
def noise(rng: np.random.Generator) -> NDArray[np.uint8]:
    """A 64x48 greyscale image of uniform noise."""
    return rng.integers(0, 256, size=(48, 64), dtype=np.uint8)


@pytest.fixture
def vertical_edge() -> NDArray[np.uint8]:
    """Black on the left, white on the right, with one sharp edge down the middle."""
    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 20:] = 255
    return image


@pytest.fixture
def horizontal_edge() -> NDArray[np.uint8]:
    """Black on top, white below."""
    image = np.zeros((40, 40), dtype=np.uint8)
    image[20:, :] = 255
    return image
