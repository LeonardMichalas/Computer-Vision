#!/usr/bin/env python3
"""Assignment 1 -- histograms, equalisation and contrast stretching.

Plots the histogram of a greyscale image, equalises it with a hand-written
lookup table, stretches its contrast, and saves a side-by-side comparison
against OpenCV's own ``equalizeHist`` so the two can be judged against each
other.

    python homework1.py
    python homework1.py --image puppy.jpg --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from classic_cv import equalise, histogram, load_grayscale, save, stretch_contrast

HERE = Path(__file__).parent
DEFAULT_IMAGE = HERE / "flower_grey_image.jpg"


def plot_histograms(images: dict[str, np.ndarray], destination: Path, show: bool) -> None:
    """Draw one histogram per image, stacked, sharing an x axis."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(images), 1, figsize=(8, 3 * len(images)), sharex=True)
    for axis, (title, image) in zip(np.atleast_1d(axes), images.items(), strict=True):
        axis.bar(np.arange(256), histogram(image), width=1.0, color="black")
        axis.set_title(title)
        axis.set_ylabel("pixels")
    np.atleast_1d(axes)[-1].set_xlabel("intensity")
    axis.set_xlim(0, 255)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    print(f"wrote {destination}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image", type=Path, default=DEFAULT_IMAGE, help="greyscale image to analyse"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "output", help="where to write results"
    )
    parser.add_argument("--show", action="store_true", help="open the plots in a window as well")
    args = parser.parse_args()

    if not args.show:
        # Nothing to display to: pick the backend that renders straight to file.
        matplotlib.use("Agg")

    import cv2

    original = load_grayscale(args.image)
    print(
        f"{args.image.name}: {original.shape[1]}x{original.shape[0]}, "
        f"intensities {original.min()}-{original.max()}"
    )

    equalised = equalise(original)
    stretched = stretch_contrast(original)

    plot_histograms(
        {
            "original": original,
            "equalised (hand-written)": equalised,
            "contrast stretched": stretched,
        },
        args.output / "histograms.png",
        args.show,
    )

    save(args.output / "equalised.png", equalised)
    save(args.output / "stretched.png", stretched)

    # The point of the comparison: our lookup table against the library's.
    reference = cv2.equalizeHist(original)
    difference = int(np.max(np.abs(equalised.astype(int) - reference.astype(int))))
    print(f"largest disagreement with cv2.equalizeHist: {difference} grey level(s)")

    comparison = np.hstack((original, equalised, reference))
    save(args.output / "comparison.png", comparison)
    print(f"wrote {args.output / 'comparison.png'} (original | ours | OpenCV)")


if __name__ == "__main__":
    main()
