"""The hand-written convolution, checked against OpenCV's."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from classic_cv import box_kernel, convolve, correlate, gaussian_kernel


def test_correlate_matches_opencv(noise: NDArray[np.uint8]) -> None:
    """cv2.filter2D correlates rather than convolves, and replicates the border."""
    kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    ours = correlate(noise, kernel, padding="edge")
    theirs = cv2.filter2D(noise.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REPLICATE)

    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_convolve_flips_the_kernel(noise: NDArray[np.uint8]) -> None:
    """A true convolution equals a correlation with the kernel turned around."""
    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    convolved = convolve(noise, kernel, padding="zero")
    correlated = correlate(noise, kernel[::-1, ::-1], padding="zero")

    np.testing.assert_allclose(convolved, correlated)
    # This kernel shifts the image; flipping decides in which direction.
    np.testing.assert_allclose(convolved[:, :-1], noise[:, 1:].astype(float))


def test_identity_kernel_returns_the_image(noise: NDArray[np.uint8]) -> None:
    identity = np.zeros((3, 3))
    identity[1, 1] = 1.0
    np.testing.assert_allclose(convolve(noise, identity), noise.astype(float))


def test_colour_images_are_filtered_per_channel(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    result = correlate(image, box_kernel(3))

    assert result.shape == image.shape
    for channel in range(3):
        np.testing.assert_allclose(
            result[:, :, channel], correlate(image[:, :, channel], box_kernel(3))
        )


def test_box_kernel_preserves_brightness(noise: NDArray[np.uint8]) -> None:
    """A kernel summing to 1 must not darken or brighten the image overall."""
    smoothed = correlate(noise, box_kernel(5), padding="edge")
    assert smoothed.mean() == pytest.approx(noise.mean(), abs=1.0)


def test_gaussian_kernel_is_normalised_and_centred() -> None:
    kernel = gaussian_kernel(5)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel[2, 2] == kernel.max()
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])


@pytest.mark.parametrize("size", [2, 4, 0, -1])
def test_even_kernels_are_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        box_kernel(size)


def test_zero_padding_darkens_the_border(noise: NDArray[np.uint8]) -> None:
    """The dark rim the 2018 version produced, kept available on purpose."""
    zero = correlate(noise, box_kernel(5), padding="zero")
    edge = correlate(noise, box_kernel(5), padding="edge")
    assert zero[0, 0] < edge[0, 0]
