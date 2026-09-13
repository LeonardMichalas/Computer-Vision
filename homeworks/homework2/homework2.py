#!/usr/bin/env python3
"""Assignment 2 -- smoothing with a box filter.

Applies a hand-written mean filter to an image and saves the result next to the
original. Works on colour and greyscale alike, one channel at a time.

    python homework2.py
    python homework2.py --image kind.png --kernel 9 --padding zero
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from classic_cv import box_kernel, correlate, save, to_uint8

HERE = Path(__file__).parent
DEFAULT_IMAGE = HERE / "ebay.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="image to smooth")
    parser.add_argument("--kernel", type=int, default=5, help="filter size, odd (default: 5)")
    parser.add_argument(
        "--padding",
        choices=["edge", "zero", "reflect"],
        default="edge",
        help="how to treat pixels outside the border. 'zero' is what the 2018 "
        "version did, and it leaves a dark rim (default: edge)",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "output", help="where to write results"
    )
    parser.add_argument("--show", action="store_true", help="open the result in a window")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"could not read an image from {args.image}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    print(
        f"{args.image.name}: {image.shape[1]}x{image.shape[0]}, "
        f"{args.kernel}x{args.kernel} box filter"
    )

    started = time.perf_counter()
    smoothed = to_uint8(correlate(image, box_kernel(args.kernel), padding=args.padding))
    print(f"smoothed in {time.perf_counter() - started:.3f}s")

    destination = save(args.output / f"smoothed_{args.kernel}x{args.kernel}.png", smoothed)
    print(f"wrote {destination}")

    comparison = np.hstack((image, smoothed))
    print(f"wrote {save(args.output / 'comparison.png', comparison)} (original | smoothed)")

    if args.show:
        cv2.imshow("original | smoothed", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
