# Classic Computer Vision

[![CI](https://github.com/LeonardMichalas/Computer-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/LeonardMichalas/Computer-Vision/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Classic computer vision algorithms, implemented from scratch.** Histogram equalisation, convolution, Sobel edge detection and the Hough transform — written out by hand with NumPy, rather than by calling the one-line OpenCV function that already does it.

That is the point of this repository. OpenCV is here to move images between disk and memory, and in the test suite as an independent second opinion to check the hand-written results against. It never provides the algorithm.

---

## Quick start

```bash
git clone https://github.com/LeonardMichalas/Computer-Vision.git
cd Computer-Vision

uv sync                                          # or: pip install -e .
uv run python examples/line_detection.py         # find the lines in a noisy drawing
```

Every example runs from anywhere, finds its own sample images, and writes results to `output/`. Add `--show` to open windows as well, and `--help` to any script for its options.

Requires **Python 3.12 or newer**.

---

## The library

```python
from classic_cv import equalise, load_grayscale, sobel_edges

image = load_grayscale("samples/sudoku.png")
edges = sobel_edges(equalise(image), threshold=90)
```

| Module | What it holds |
| --- | --- |
| `classic_cv.images` | Loading, saving, and the two ways back to 8 bits: `to_uint8` clips, `normalise` rescales |
| `classic_cv.convolution` | `correlate`, `convolve`, `box_kernel`, `gaussian_kernel` |
| `classic_cv.histogram` | `histogram`, `cumulative_distribution`, `equalise`, `stretch_contrast` |
| `classic_cv.edges` | `SOBEL_X`, `SOBEL_Y`, `sobel_gradients`, `gradient_magnitude`, `sobel_edges` |
| `classic_cv.hough` | `threshold_mask`, `hough_transform`, `find_peaks`, `draw_lines` |

---

## Examples

### Histogram equalisation

Equalisation the long way: tally the histogram, take its cumulative distribution, ignore the empty levels below the darkest pixel present, and stretch what is left across the full 0–255 range. That gives a lookup table, and the table is applied to the whole image at once.

The script then runs `cv2.equalizeHist` on the same image and reports the largest disagreement between the two. **It is currently zero grey levels** — the hand-written table is not merely close to OpenCV's, it is identical.

Contrast stretching is the simpler cousin: it moves the two ends of the intensity range without changing the shape of the histogram in between.

```bash
uv run python examples/histogram_equalization.py --image samples/puppy.jpg --show
```

### Image smoothing

A mean filter of any odd size, with a choice of what to do at the border: repeat the edge pixel, mirror the image, or treat everything outside as black. The last one is the simplest rule and the one that leaves a dark rim — available as `--padding zero`, because seeing the artefact explains the choice better than a paragraph does.

```bash
uv run python examples/image_smoothing.py --kernel 9 --padding zero
```

### Edge detection

The pipeline, with every stage saved separately so it can be inspected:

1. **Smooth** first — a derivative amplifies noise, so without this every sensor speckle becomes an edge.
2. **Differentiate** with the two Sobel kernels, giving a signed gradient in x and in y.
3. **Combine** into one edge strength: `hypot(gx, gy)`, or `|gx| + |gy|` with `--norm l1`, the cheaper approximation, which overstates diagonal edges by up to 41%.
4. **Threshold** into a binary edge map.

```bash
uv run python examples/edge_detection.py --image samples/coin.png --threshold 60
```

### Line detection

Finding straight lines by voting, and the most complete piece here. A line is written as `rho = x·cos θ + y·sin θ` rather than `y = mx + c`, because that form can express a vertical line. Every marked pixel then votes for every line that could pass through it. Where many pixels lie on the same real line their votes land in the same cell and pile up, so finding lines in the image becomes finding peaks in the accumulator.

Two details worth the attention they take:

- **The accumulator is saved as an image** (`houghspace.png`). The voting pattern is the part worth looking at — the butterfly shapes are what a line looks like in Hough space.
- **Peaks are taken one at a time, and each one's neighbourhood is then zeroed.** Without that, a single strong line wins every slot, because the cells either side of a peak are nearly as tall as the peak itself.

The `samples/lines_*.pgm` files are the same drawing under rising amounts of noise, from `lines_clean` through `lines_noise_high`, plus `lines_broken` where the strokes are dotted rather than continuous. They make it easy to see where the transform stops coping.

```bash
uv run python examples/line_detection.py --image samples/lines_noise_high.pgm --peaks 4
```

---

## Two conventions that bite

Both are pinned down by tests, because both are easy to get wrong and hard to notice.

- **Correlation is not convolution.** The Sobel kernels here are written to be *correlated* with the image, which is what OpenCV, MATLAB and every textbook figure do, and what makes a dark-to-light step come out positive. A true convolution flips the kernel, and because the Sobel kernels are antisymmetric the gradients come back negated — same edges, opposite meaning.
- **`reflect` is not `BORDER_REFLECT`.** NumPy's `reflect` is OpenCV's `BORDER_REFLECT_101`. OpenCV's plain `BORDER_REFLECT` repeats the edge pixel as it mirrors, which is a fourth, different thing. The interiors agree; only the rim gives it away.

## On the loops

The hand-written convolution loops over the *kernel taps*, not over the pixels: for each of the kernel's entries the whole image is shifted and added in one go. That is the same arithmetic as the textbook four-deep pixel loop, and it is still the convolution written out by hand — it is the version NumPy can actually execute.

The difference is not academic. Smoothing the 420×336 sample image with a 5×5 filter:

| | Time |
| --- | --- |
| Looping over pixels | 6.26 s |
| Looping over kernel taps | 0.013 s |

Same result to within one grey level of rounding, about 500 times faster.

---

## Development

```bash
uv run pytest          # 49 tests
uv run ruff check .    # lint
uv run mypy src        # types, strict mode
```

The interesting tests are the ones that check the hand-written code against an independent implementation of the same thing:

- `correlate` against `cv2.filter2D`
- `sobel_gradients` against `cv2.Sobel`
- `equalise` against `cv2.equalizeHist` — exact match, every pixel
- the Hough transform against lines whose `rho` and `theta` are known by construction

The rest cover the cases that are easy to get wrong: a flat image with nothing to equalise, an empty accumulator with nothing to find, kernels with no centre, thresholds that select nothing. CI runs all of it on Python 3.12 and 3.13, and then runs all four examples end to end, because a green library with a broken script is not much use.

### Layout

```
src/classic_cv/   the algorithms
examples/         one runnable script per technique
samples/          test images
tests/            49 tests, including the cross-checks against OpenCV
```

---

## License

MIT — see [LICENSE](LICENSE).

<sub>A personal project, built in my own time.</sub>
