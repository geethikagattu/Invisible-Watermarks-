# Detecting AI-Edited Parts of an Image Using Invisible Watermarks

This project trains a PyTorch pipeline to embed an invisible message in an
image and detect/localize the watermark after common image transformations.
The main entry point is [`maskmark_full-3.py`](./maskmark_full-3.py).

## Requirements

- Python 3.9 or newer
- The packages listed in [`requirements.txt`](./requirements.txt)
- A CUDA-capable PyTorch installation is optional; the script uses CUDA when
  available and otherwise runs on CPU.

## Installation

From the project directory, create and activate a virtual environment:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For CUDA acceleration, install the PyTorch build that matches your system from
the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Dataset

The dataset used for the reported experiments is CASIA 2.0, containing
authentic images, tampered images, and ground-truth masks. The dataset is not
included in this repository; obtain it separately, respecting its license,
and place the files as described below.

The script does not download datasets. It scans image directories recursively
and supports `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.bmp` files.

Pass the image directory with `--images`. For meaningful localization results,
also pass binary ground-truth masks with `--masks`. Each mask should have the
same filename stem as its image, or use the image stem followed by `_gt`
(for example, `image01.jpg` and `image01_gt.png`). The script indexes mask
files by stem and resizes them to the selected image size.

Example layout:

```text
data/
├── images/
│   ├── image01.jpg
│   └── image02.jpg
└── masks/
    ├── image01_gt.png
    └── image02_gt.png
```

For validation and testing, a usable matching ground-truth mask is used when
available. If a mask is missing or has fewer than 16 foreground pixels, the
script generates a random rectangle or ellipse instead. Random masks are
deterministic for validation and test samples. During training, the script
sometimes substitutes a random mask even when a ground-truth mask exists.

Images are resized to 128x128 by default (the reported setup uses this size).
The image list is shuffled with a fixed seed and split by image into 80%
training, 10% validation, and 10% test sets. With the complete CASIA 2.0 image
set used in the reported experiments, the test split contains 1,262 images.

## Run

### Quick synthetic run

This generates synthetic images and masks; it requires no dataset and is
intended to check that installation and the full pipeline work:

```bash
python maskmark_full-3.py --synthetic --epochs 2 --out results
```

### Train and test with your dataset

```bash
python maskmark_full-3.py \
  --images ./data/images \
  --masks ./data/masks \
  --epochs 15 \
  --out results
```

The script trains using the training split, validates during training, then
evaluates the held-out test split against clean images and several simulated
attacks. Training and testing are part of this single command; the script does
not currently provide a separate test-only command. Omit `--masks` if masks
are unavailable, bearing in mind the random-mask fallback described above.

Useful options include `--size` (image side length, default `128`), `--bs`
(batch size, default `16`), `--max_images` (limit the number of images; `0`
means no limit), and `--warmup` (invisibility-loss warmup epochs, default `3`).
Run `python maskmark_full-3.py --help` to see all options.

## Outputs

The directory passed to `--out` is created if needed. A completed run writes:

- `maskmark.pt` — trained encoder and extractor weights
- `results.json` — metrics for each evaluated attack
- `roc_pr.png`, `confusion.png`, `robustness.png` — evaluation plots
- `examples.png` — sample images, watermark maps, and saliency maps

## Project structure

```text
.
├── maskmark_full-3.py   # Main entry point: training, evaluation, and plots
├── requirements.txt     # Python package dependencies
├── README.md            # Setup, dataset, and usage instructions
├── data/                # Optional: place downloaded dataset here
└── results/             # Created when the pipeline runs
```
