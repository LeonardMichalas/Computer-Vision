# Computer Vision

**Classic computer vision algorithms, implemented from scratch.** Histogram equalisation, convolution, Sobel edge detection and the Hough transform — written out by hand with NumPy, rather than by calling the one-line OpenCV function that already does it.

That is the whole point of this repository. OpenCV is here to move images between disk and memory, and in the test suite as an independent second opinion to check the hand-written results against. It never provides the algorithm.

> Originally coursework for *Introduction to Computer Vision* at DHBW Stuttgart in 2018, written for Python 2.7. Rebuilt in 2026 for current Python, with a test suite and CI. The algorithms are the same ones; what changed is everything around them — see [What changed](#what-changed-in-the-rewrite).

---

## Quick start

```bash
git clone https://github.com/LeonardMichalas/Computer-Vision.git
cd Computer-Vision

uv sync                                        # or: pip install -e .
uv run python exercises/ex4/homework4.py       # find lines in a hand-drawn image
```

Every script runs from anywhere, finds its own sample images, and writes results to an `output/` folder next to itself. Add `--show` to open windows as well, and `--help` to any script to see its options.

Requires **Python 3.12 or newer**.

---

## What is inside

| # | Topic | Written by hand | Script |
| --- | --- | --- | --- |
| 1 | Histograms and contrast | Histogram, CDF equalisation, contrast stretching | [`homeworks/homework1`](homeworks/homework1/homework1.py) |
| 2 | Smoothing | Box filter, three border policies | [`homeworks/homework2`](homeworks/homework2/homework2.py) |
| 3 | Edge detection | 2D convolution, Sobel, gradient magnitude, thresholding | [`homeworks/homework3`](homeworks/homework3/homework3.py) |
| 4 | Line detection | Hough transform, accumulator, non-maximum suppression | [`exercises/ex4`](exercises/ex4/homework4.py) |

### 1 — Histograms, equalisation and contrast

Equalisation the long way: tally the histogram, take its cumulative distribution, ignore the empty levels below the darkest pixel present, and stretch what is left across the full 0–255 range. That gives a lookup table, and the table is applied to the whole image at once.

The script then runs `cv2.equalizeHist` on the same image and reports the largest disagreement between the two. **It is currently zero grey levels** — the hand-written table is not merely close to OpenCV's, it is identical.

Contrast stretching is the simpler cousin, and it is the function that was marked `NOT WORKING AS IT SHOULD` in 2018. It is fixed here; the three reasons it was broken are written up in the docstring.

```bash
uv run python homeworks/homework1/homework1.py --image puppy.jpg --show
```

### 2 — Smoothing with a box filter

A mean filter of any odd size, with a choice of what to do at the border: repeat the edge pixel, mirror the image, or treat everything outside as black. That last one is what the 2018 version did, and it is why its output had a dark rim. It is still available as `--padding zero`, because seeing the artefact explains the choice better than a paragraph does.

```bash
uv run python homeworks/homework2/homework2.py --kernel 9 --padding zero
```

### 3 — Sobel edge detection

The pipeline, with every stage saved separately so it can be inspected:

1. **Smooth** first — a derivative amplifies noise, so without this every sensor speckle becomes an edge.
2. **Differentiate** with the two Sobel kernels, giving a signed gradient in x and in y.
3. **Combine** into one edge strength: `hypot(gx, gy)`, or `|gx| + |gy|` with `--norm l1`, which is the cheap approximation the 2018 version used and which overstates diagonal edges by up to 41%.
4. **Threshold** into a binary edge map.

```bash
uv run python homeworks/homework3/homework3.py --image coin.png --threshold 60
```

### 4 — The Hough transform

Finding straight lines by voting, and the most complete piece here. A line is written as `rho = x·cos θ + y·sin θ` rather than `y = mx + c`, because that form can express a vertical line. Every marked pixel then votes for every line that could pass through it. Where many pixels lie on the same real line their votes land in the same cell and pile up, so finding lines in the image becomes finding peaks in the accumulator.

Two details worth the attention they take:

- **The accumulator is saved as an image** (`houghspace.png`). The voting pattern is the part worth looking at — the butterfly shapes are what a line looks like in Hough space.
- **Peaks are taken one at a time, and each one's neighbourhood is then zeroed.** Without that, a single strong line wins every slot, because the cells either side of a peak are nearly as tall as the peak itself.

```bash
uv run python exercises/ex4/homework4.py --image img3.pgm --peaks 4 --theta-bins 360
```

---

## The library

The algorithms live in `src/classic_cv/`, so they can be imported instead of copied between assignments.

```python
from classic_cv import equalise, load_grayscale, sobel_edges

image = load_grayscale("photo.png")
edges = sobel_edges(equalise(image), threshold=90)
```

| Module | What it holds |
| --- | --- |
| `classic_cv.images` | Loading, saving, and the two ways back to 8 bits: `to_uint8` clips, `normalise` rescales |
| `classic_cv.convolution` | `correlate`, `convolve`, `box_kernel`, `gaussian_kernel` |
| `classic_cv.histogram` | `histogram`, `cumulative_distribution`, `equalise`, `stretch_contrast` |
| `classic_cv.edges` | `SOBEL_X`, `SOBEL_Y`, `sobel_gradients`, `gradient_magnitude`, `sobel_edges` |
| `classic_cv.hough` | `threshold_mask`, `hough_transform`, `find_peaks`, `draw_lines` |

### Two conventions that bite

Both are pinned down by tests, because both are easy to get wrong and hard to notice.

- **Correlation is not convolution.** The Sobel kernels here are written to be *correlated* with the image, which is what OpenCV, MATLAB and every textbook figure do, and what makes a dark-to-light step come out positive. A true convolution flips the kernel, and because the Sobel kernels are antisymmetric the gradients come back negated — same edges, opposite meaning.
- **`reflect` is not `BORDER_REFLECT`.** NumPy's `reflect` is OpenCV's `BORDER_REFLECT_101`. OpenCV's plain `BORDER_REFLECT` repeats the edge pixel as it mirrors, which is a fourth, different thing. The interiors agree; only the rim gives it away.

### On the loops

The hand-written convolution loops over the *kernel taps*, not over the pixels: for each of the kernel's entries the whole image is shifted and added in one go. That is the same arithmetic as the textbook four-deep pixel loop, and it is still the convolution written out by hand — it is the version NumPy can actually execute.

The difference is not academic. Smoothing the 420×336 sample image with a 5×5 filter:

| | Time |
| --- | --- |
| 2018 version, pixel-by-pixel loops | 6.26 s |
| This version | 0.013 s |

Same result to within one grey level of rounding, about 500 times faster.

---

## How it is verified

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

The rest cover the edges that are easy to get wrong: a flat image with nothing to equalise, an empty accumulator with nothing to find, kernels with no centre, thresholds that select nothing. CI runs all of it on Python 3.12 and 3.13, and then runs all four assignment scripts end to end, because a green library with a broken script is not much use.

---

## What changed in the rewrite

The algorithms are unchanged. Everything around them is different.

**Fixed**

- Python 3.12+ throughout. The original blocker was integer division: `(kernel-1)/2` returns `2.0` on Python 3, and `range()` will not take a float.
- Contrast stretching now works. It divided by 256 instead of 255, so white came out one level short; it wrote floats into a `uint8` array; and it never clipped, so a pixel outside the window wrapped around and a shadow came back bright white.
- Negative kernel indices no longer wrap around the array. That was harmless for a symmetric all-ones kernel and wrong for anything else.
- Border handling is a choice rather than an accident.
- `filter` no longer shadows the Python builtin.
- `.vscode/settings.json` no longer points at a hardcoded Anaconda path on a MacBook from 2018.

**Added**

- A real package, so the four assignments share one convolution instead of three copies.
- 49 tests, ruff, mypy in strict mode, and GitHub Actions across two Python versions.
- A command line on every script: `--image`, `--threshold`, `--output`, `--show` and so on, instead of editing a filename at the top of the file.
- Every intermediate stage saved, not just the final result.

**Removed**

- `homeworks/homework4/homework4.py`, a stub containing one comment line. The finished assignment 4 is in `exercises/ex4/`.
- `project/empty.py`, a placeholder for a project that was never started.

The 2018 code is all still in the git history, including the first attempt at smoothing that unrolled 25 neighbours into 25 named variables.

---

## Seminar paper

`SeminarPaper_CV.pdf` — *"Amazon Go as a Use Case to Enable Image Recognition Using Convolutional Neural Networks"*, submitted 7 June 2018 at DHBW Stuttgart, Faculty of Economics, Business Information Systems. A group paper written with three coursemates.

`Paintings/` holds the figures drawn by hand for that work — feature maps, pooling, normalisation, and the SVM and random forest comparisons — as editable `.paint` sources next to exported PNGs.

---

<sub>Built in my own time. Coursework from 2018, rebuilt in 2026 because the algorithms are worth reading and the Python was not.</sub>
