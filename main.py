"""
main.py
-------
Iterates over every image in the `images/` folder, counts cells in each one
using the selected cell-counting method, and prints a grouped summary table.
Annotated copies are saved to a settings-specific folder under `images/`.
"""

import sys
from pathlib import Path

from log_blob_cell_counter import LoGBlobCellCounter
from summary_report import SummaryReport
from threshold_cell_counter import ThresholdCellCounter
from watershed_cell_counter import WatershedCellCounter

# Configuration

IMAGES_DIR = Path("images")                  # folder that contains the source images
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Counting settings (tweak these for your microscopy images)
COUNTING_METHOD = "log"  # "threshold" | "watershed" | "log"
MIN_CELL_AREA = 200          # pixels^2 -- blobs smaller than this are ignored
MAX_CELL_AREA = 4_000       # pixels^2 -- blobs larger than this are ignored

# Thresholding settings
THRESHOLD_METHOD = "adaptive"   # "otsu" | "adaptive" | "simple"

# Watershed settings
WATERSHED_FOREGROUND_THRESHOLD = 0.1
WATERSHED_OPENING_KERNEL_SIZE = 1 # [1,3,7,9...]
WATERSHED_OPENING_ITERATIONS = 1 # int
WATERSHED_DEBUG = False

# LoG blob detection settings
LOG_MIN_SIGMA = 5.0
LOG_MAX_SIGMA = 10.0
LOG_NUM_SIGMA = 10
LOG_RESPONSE_THRESHOLD = 0.06
LOG_OVERLAP_THRESHOLD = 0.6
LOG_DEBUG = False

# Summary settings
PLOT_COUNTS = True         # set to True to save a cell-count-over-time plot


# Helpers

def build_output_dir() -> Path:
    """Return the annotated-image folder for the current run settings."""
    folder_name = f"annotated_{COUNTING_METHOD}_min{MIN_CELL_AREA}_max{MAX_CELL_AREA}"

    if COUNTING_METHOD == "threshold":
        folder_name += f"_{THRESHOLD_METHOD}"
    elif COUNTING_METHOD == "watershed":
        folder_name += (
            f"_fg{WATERSHED_FOREGROUND_THRESHOLD:g}"
            f"_openK{WATERSHED_OPENING_KERNEL_SIZE}"
            f"_openI{WATERSHED_OPENING_ITERATIONS}"
        )
    elif COUNTING_METHOD == "log":
        folder_name += (
            f"_sigma{LOG_MIN_SIGMA:g}-{LOG_MAX_SIGMA:g}"
            f"_n{LOG_NUM_SIGMA}"
            f"_resp{LOG_RESPONSE_THRESHOLD:g}"
            f"_overlap{LOG_OVERLAP_THRESHOLD:g}"
        )

    return IMAGES_DIR / folder_name


def create_counter(
    image_path: Path,
) -> ThresholdCellCounter | WatershedCellCounter | LoGBlobCellCounter:
    """Build the configured cell counter for one image."""
    if COUNTING_METHOD == "threshold":
        return ThresholdCellCounter(
            image_path=str(image_path),
            threshold_method=THRESHOLD_METHOD,
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
        )

    if COUNTING_METHOD == "watershed":
        return WatershedCellCounter(
            image_path=str(image_path),
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
            foreground_threshold=WATERSHED_FOREGROUND_THRESHOLD,
            opening_kernel_size=WATERSHED_OPENING_KERNEL_SIZE,
            opening_iterations=WATERSHED_OPENING_ITERATIONS,
            debug=WATERSHED_DEBUG,
        )

    if COUNTING_METHOD == "log":
        return LoGBlobCellCounter(
            image_path=str(image_path),
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
            min_sigma=LOG_MIN_SIGMA,
            max_sigma=LOG_MAX_SIGMA,
            num_sigma=LOG_NUM_SIGMA,
            response_threshold=LOG_RESPONSE_THRESHOLD,
            overlap_threshold=LOG_OVERLAP_THRESHOLD,
            debug=LOG_DEBUG,
        )

    raise ValueError(
        f"Unknown COUNTING_METHOD '{COUNTING_METHOD}'. "
        "Choose 'threshold', 'watershed', or 'log'."
    )


def collect_images(directory: Path) -> list[Path]:
    """Return a sorted list of supported image paths inside *directory*."""
    if not directory.exists():
        print(f"[ERROR] Images directory not found: {directory.resolve()}")
        sys.exit(1)

    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not paths:
        print(f"[WARNING] No supported images found in: {directory.resolve()}")
    return paths


def process_image(image_path: Path, output_dir: Path) -> dict:
    """
    Run the full cell-counting pipeline for one image.
    Returns a dict with the image name and cell count (or an error message).
    """
    counter = create_counter(image_path)

    try:
        cell_count = counter.run()

        annotated_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        counter.save_annotated(str(annotated_path))

        return {"file": image_path.name, "cells": cell_count, "error": None}

    except Exception as exc:
        return {"file": image_path.name, "cells": None, "error": str(exc)}


# Entry point

def main() -> None:
    output_dir = build_output_dir()
    if output_dir.exists():
        print(
            "[INFO] Annotated folder for these settings already exists; "
            "skipping this run."
        )
        print(f"Folder: {output_dir.resolve()}")
        return

    output_dir.mkdir(parents=True, exist_ok=False)

    image_paths = collect_images(IMAGES_DIR)
    print(f"Found {len(image_paths)} image(s) in '{IMAGES_DIR}'. Processing...\n")

    results = []
    for path in image_paths:
        print(f"  Processing: {path.name} ...", end=" ", flush=True)
        result = process_image(path, output_dir)
        results.append(result)

        if result["error"]:
            print(f"FAILED ({result['error']})")
        else:
            print(f"{result['cells']} cell(s) found.")

    report = SummaryReport(results)
    report.print_summary()
    report_path = output_dir / "summary_report.txt"
    report.save_text(report_path)
    print(f"Summary report saved to: {report_path.resolve()}")

    if PLOT_COUNTS:
        plot_path = output_dir / "cell_counts_over_time.png"
        report.plot_counts_over_time(plot_path)
        print(f"Cell-count plot saved to: {plot_path.resolve()}")

    print(f"Annotated images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
