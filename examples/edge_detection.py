#!/usr/bin/env python3
"""Sobel edge detection.

Smooths the image, takes the Sobel derivative in both directions, combines them
into an edge strength and thresholds the result into a binary edge map. Every
stage is written out and saved, so the pipeline can be inspected step by step.

    python examples/edge_detection.py
    python examples/edge_detection.py --image samples/coin.png --threshold 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from classic_cv import (
    box_kernel,
    correlate,
    gradient_magnitude,
    load_grayscale,
    normalise,
    save,
    sobel_gradients,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "samples" / "sudoku.png"
DEFAULT_OUTPUT = ROOT / "output" / "edge_detection"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="image to find edges in")
    parser.add_argument(
        "--threshold", type=float, default=90.0, help="edge strength cut-off (default: 90)"
    )
    parser.add_argument(
        "--smooth", type=int, default=3, help="box filter size applied first, 0 to skip"
    )
    parser.add_argument(
        "--norm",
        choices=["l2", "l1"],
        default="l2",
        help="how to combine the two gradients. 'l1' is the cheaper "
        "|gx|+|gy| approximation (default: l2)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write results"
    )
    parser.add_argument("--show", action="store_true", help="open the stages in windows")
    args = parser.parse_args()

    image = load_grayscale(args.image)
    print(f"{args.image.name}: {image.shape[1]}x{image.shape[0]}")

    # 1. Smooth first, or the derivative turns every speck of noise into an edge.
    prepared = correlate(image, box_kernel(args.smooth)) if args.smooth else image.astype(float)

    # 2. Differentiate in both directions.
    gradient_x, gradient_y = sobel_gradients(np.asarray(prepared))

    # 3. Combine into one edge strength.
    magnitude = gradient_magnitude(gradient_x, gradient_y, args.norm)

    # 4. Threshold into a binary edge map.
    edges = np.where(magnitude > args.threshold, 255, 0).astype(np.uint8)

    stages = {
        "01_smoothed.png": np.clip(prepared, 0, 255).astype(np.uint8),
        # The gradients are signed, so mid-grey is zero and both edge
        # directions stay visible.
        "02_gradient_x.png": normalise(gradient_x),
        "03_gradient_y.png": normalise(gradient_y),
        "04_magnitude.png": normalise(magnitude),
        "05_edges.png": edges,
    }
    for name, picture in stages.items():
        save(args.output / name, picture)
    print(f"wrote {len(stages)} stages to {args.output}")

    edge_share = float(np.count_nonzero(edges)) / edges.size
    print(f"{edge_share:.1%} of pixels are above the threshold of {args.threshold:g}")

    # An independent opinion on the same image, for comparison only.
    reference = cv2.Canny(image, 100, 200)
    save(args.output / "06_canny_reference.png", reference)

    if args.show:
        for name, picture in stages.items():
            cv2.imshow(name, picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
