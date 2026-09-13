#!/usr/bin/env python3
"""Line detection with the Hough transform.

Finds straight lines by voting. Every marked pixel votes for each line that
could pass through it; the lines that really exist collect the most votes. The
accumulator is saved as an image too, because the voting pattern is the part
worth looking at.

    python examples/line_detection.py
    python examples/line_detection.py --image samples/lines_noise_high.pgm --peaks 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from classic_cv import draw_lines, find_peaks, hough_transform, load_grayscale, save, threshold_mask
from classic_cv.images import normalise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "samples" / "lines_clean.pgm"
DEFAULT_OUTPUT = ROOT / "output" / "line_detection"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="image to find lines in")
    parser.add_argument("--peaks", type=int, default=9, help="how many lines to look for")
    parser.add_argument(
        "--threshold", type=int, default=0, help="pixels this dark or darker vote (default: 0)"
    )
    parser.add_argument(
        "--bright", action="store_true", help="vote on bright pixels instead, for an edge map"
    )
    parser.add_argument("--theta-bins", type=int, default=180, help="angle resolution")
    parser.add_argument(
        "--suppression", type=int, default=10, help="cells cleared around each peak"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write results"
    )
    parser.add_argument("--show", action="store_true", help="open the results in windows")
    args = parser.parse_args()

    image = load_grayscale(args.image)
    mask = threshold_mask(image, args.threshold, below=not args.bright)
    voters = int(mask.sum())
    print(f"{args.image.name}: {image.shape[1]}x{image.shape[0]}, {voters} pixels voting")
    if voters == 0:
        raise SystemExit("no pixel passed the threshold, so there is nothing to vote")

    space = hough_transform(mask, theta_bins=args.theta_bins)
    lines = find_peaks(space, count=args.peaks, suppression_radius=args.suppression)

    print(f"{len(lines)} line(s) found:")
    for index, line in enumerate(lines, start=1):
        print(
            f"  {index:2d}. rho={line.rho:8.2f}px  theta={line.theta:6.2f}rad  {line.votes} votes"
        )

    save(args.output / "houghspace.png", normalise(space.accumulator.astype(float)))

    detected = draw_lines(image, lines)
    # Put the voting pixels back on top in black. A detected line covers the
    # very pixels that voted for it, so without this the evidence is hidden
    # under the answer.
    detected[mask] = (0, 0, 0)
    save(args.output / "lines.png", detected)
    print(f"wrote {args.output / 'houghspace.png'} and {args.output / 'lines.png'}")

    if args.show:
        cv2.imshow("hough space", normalise(space.accumulator.astype(float)))
        cv2.imshow("detected lines", detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
