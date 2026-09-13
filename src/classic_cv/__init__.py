"""Classic computer vision algorithms, implemented from scratch.

Every algorithm in this package is written out by hand with NumPy. OpenCV is
used for reading, writing and displaying images, and in the test suite as an
independent reference to check the hand-written results against -- never as the
implementation itself.
"""

from classic_cv.convolution import box_kernel, convolve, correlate, gaussian_kernel
from classic_cv.edges import SOBEL_X, SOBEL_Y, gradient_magnitude, sobel_edges, sobel_gradients
from classic_cv.histogram import cumulative_distribution, equalise, histogram, stretch_contrast
from classic_cv.hough import draw_lines, find_peaks, hough_transform, threshold_mask
from classic_cv.images import load_grayscale, normalise, save, to_uint8

__version__ = "1.0.0"

__all__ = [
    "SOBEL_X",
    "SOBEL_Y",
    "box_kernel",
    "convolve",
    "correlate",
    "cumulative_distribution",
    "draw_lines",
    "equalise",
    "find_peaks",
    "gaussian_kernel",
    "gradient_magnitude",
    "histogram",
    "hough_transform",
    "load_grayscale",
    "normalise",
    "save",
    "sobel_edges",
    "sobel_gradients",
    "stretch_contrast",
    "threshold_mask",
    "to_uint8",
]
