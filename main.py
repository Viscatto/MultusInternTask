"""
main.py
-------
Iterates over every image in the `images/` folder, counts cells in each one
using the selected cell-counting method, and prints a grouped summary table.
Annotated copies are saved to a settings-specific folder under `images/`.
"""

import sys
from pathlib import Path

from counters.cellpose_cell_counter import CellposeCellCounter
from counters.dino_cell_counter import DinoCellCounter
from counters.dino_cell_counter2 import DinoCellCounter2
from counters.log_blob_cell_counter import LoGBlobCellCounter
from preprocess_images import (
    build_output_path as build_preprocess_output_path,
)
from preprocess_images import process_image as preprocess_only_image
from preprocess_images import save_image as save_preprocess_image
from summary_report import SummaryReport
from counters.threshold_cell_counter import ThresholdCellCounter
from counters.watershed_cell_counter import WatershedCellCounter


# Configuration

IMAGES_DIR = Path("output")                  # folder that contains the source images
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Counting settings (tweak these for your microscopy images)
COUNTING_METHOD = "watershed"  # "threshold" | "watershed" | "log" | "cellpose" | "dino" | "dino2" | "preprocess"
MIN_CELL_AREA = 300          # pixels^2 -- blobs smaller than this are ignored
MAX_CELL_AREA = 2_000       # pixels^2 -- blobs larger than this are ignored

# Preprocessing-only settings
PREPROCESS_OUTPUT_DIR = Path("output")

# Thresholding settings
THRESHOLD_METHOD = "adaptive"   # "otsu" | "adaptive" | "simple"

# Watershed settings
WATERSHED_FOREGROUND_THRESHOLD = 0.1
WATERSHED_OPENING_KERNEL_SIZE = 3 # [1,3,5,7,9...]
WATERSHED_OPENING_ITERATIONS = 2 # int
WATERSHED_DEBUG = False

# LoG blob detection settings
LOG_MIN_SIGMA = 5.0
LOG_MAX_SIGMA = 10.0
LOG_NUM_SIGMA = 10
LOG_RESPONSE_THRESHOLD = 0.06
LOG_OVERLAP_THRESHOLD = 0.6
LOG_DEBUG = False

# Cellpose settings
CELLPOSE_MODEL_TYPE = "cyto3"
CELLPOSE_TRAIN_MODEL = True
CELLPOSE_CUSTOM_MODELS_DIR = Path("cellpose_models")
CELLPOSE_CUSTOM_MODEL_NAME = None
CELLPOSE_PRETRAINED_MODEL_PATH = None
CELLPOSE_DIAMETER = 100
CELLPOSE_FLOW_THRESHOLD = 0.8
CELLPOSE_CELLPROB_THRESHOLD = -4.0
CELLPOSE_CHANNELS = [0, 0]
CELLPOSE_USE_GPU = False
CELLPOSE_NORMALIZE = True
CELLPOSE_USE_PREPROCESSED_IMAGE = True
CELLPOSE_MAX_IMAGE_DIMENSION = 512
CELLPOSE_BATCH_SIZE = 1
CELLPOSE_TILE_OVERLAP = 0.05
CELLPOSE_BSIZE = 256

# DINO attention settings
DINO_MODEL_NAME = "vit_small_patch16_224.dino"
DINO_THRESHOLD = 0.45
MIN_CELL_DISTANCE = 10
MIN_CELL_SIZE = 40
DINO_MAX_IMAGE_DIMENSION = 896
DINO_OPENING_KERNEL_SIZE = 3
DINO_OPENING_ITERATIONS = 1
DINO_CLOSING_KERNEL_SIZE = 5
DINO_CLOSING_ITERATIONS = 1
DINO_PEAK_THRESHOLD = 0.2
DINO_DEBUG = False

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
    elif COUNTING_METHOD == "dino":
        folder_name += (
            f"_thr{DINO_THRESHOLD:g}"
            f"_dist{MIN_CELL_DISTANCE}"
            f"_size{MIN_CELL_SIZE}"
            f"_maxdim{DINO_MAX_IMAGE_DIMENSION}"
            f"_openK{DINO_OPENING_KERNEL_SIZE}"
            f"_openI{DINO_OPENING_ITERATIONS}"
            f"_closeK{DINO_CLOSING_KERNEL_SIZE}"
            f"_closeI{DINO_CLOSING_ITERATIONS}"
            f"_peak{DINO_PEAK_THRESHOLD:g}"
        )
    elif COUNTING_METHOD == "dino2":
        folder_name += (
            f"_thr{DINO_THRESHOLD:g}"
            f"_dist{MIN_CELL_DISTANCE}"
            f"_size{MIN_CELL_SIZE}"
            f"_maxdim{DINO_MAX_IMAGE_DIMENSION}"
            f"_openK{DINO_OPENING_KERNEL_SIZE}"
            f"_openI{DINO_OPENING_ITERATIONS}"
            f"_closeK{DINO_CLOSING_KERNEL_SIZE}"
            f"_closeI{DINO_CLOSING_ITERATIONS}"
            f"_peak{DINO_PEAK_THRESHOLD:g}"
        )
    elif COUNTING_METHOD == "preprocess":
        folder_name = PREPROCESS_OUTPUT_DIR.name
    elif COUNTING_METHOD == "cellpose":
        diameter = "auto" if CELLPOSE_DIAMETER is None else f"{CELLPOSE_DIAMETER:g}"
        model_label = CELLPOSE_CUSTOM_MODEL_NAME or CELLPOSE_MODEL_TYPE
        folder_name += (
            f"_{model_label}"
            f"_diam{diameter}"
            f"_flow{CELLPOSE_FLOW_THRESHOLD:g}"
            f"_prob{CELLPOSE_CELLPROB_THRESHOLD:g}"
            f"_ch{CELLPOSE_CHANNELS[0]}-{CELLPOSE_CHANNELS[1]}"
            f"_pre{int(CELLPOSE_USE_PREPROCESSED_IMAGE)}"
            f"_maxdim{CELLPOSE_MAX_IMAGE_DIMENSION}"
            f"_train{int(CELLPOSE_TRAIN_MODEL)}"
        )

    return IMAGES_DIR / folder_name


def create_counter(
    image_path: Path,
) -> (
    ThresholdCellCounter
    | WatershedCellCounter
    | LoGBlobCellCounter
    | DinoCellCounter
    | DinoCellCounter2
    | CellposeCellCounter
):
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

    if COUNTING_METHOD == "dino":
        return DinoCellCounter(
            image_path=str(image_path),
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
            attention_threshold=DINO_THRESHOLD,
            min_cell_distance=MIN_CELL_DISTANCE,
            min_cell_size=MIN_CELL_SIZE,
            max_image_dimension=DINO_MAX_IMAGE_DIMENSION,
            opening_kernel_size=DINO_OPENING_KERNEL_SIZE,
            opening_iterations=DINO_OPENING_ITERATIONS,
            closing_kernel_size=DINO_CLOSING_KERNEL_SIZE,
            closing_iterations=DINO_CLOSING_ITERATIONS,
            peak_threshold=DINO_PEAK_THRESHOLD,
            debug=DINO_DEBUG,
            model_name=DINO_MODEL_NAME,
        )

    if COUNTING_METHOD == "dino2":
        return DinoCellCounter2(
            image_path=str(image_path),
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
            attention_threshold=DINO_THRESHOLD,
            min_cell_distance=MIN_CELL_DISTANCE,
            min_cell_size=MIN_CELL_SIZE,
            max_image_dimension=DINO_MAX_IMAGE_DIMENSION,
            opening_kernel_size=DINO_OPENING_KERNEL_SIZE,
            opening_iterations=DINO_OPENING_ITERATIONS,
            closing_kernel_size=DINO_CLOSING_KERNEL_SIZE,
            closing_iterations=DINO_CLOSING_ITERATIONS,
            peak_threshold=DINO_PEAK_THRESHOLD,
            debug=DINO_DEBUG,
            model_name=DINO_MODEL_NAME,
        )

    if COUNTING_METHOD == "cellpose":
        return CellposeCellCounter(
            image_path=str(image_path),
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA,
            model_type=CELLPOSE_MODEL_TYPE,
            train_model=CELLPOSE_TRAIN_MODEL,
            custom_models_dir=str(CELLPOSE_CUSTOM_MODELS_DIR),
            custom_model_name=CELLPOSE_CUSTOM_MODEL_NAME,
            pretrained_model_path=(
                str(CELLPOSE_PRETRAINED_MODEL_PATH)
                if CELLPOSE_PRETRAINED_MODEL_PATH is not None
                else None
            ),
            diameter=CELLPOSE_DIAMETER,
            flow_threshold=CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
            channels=CELLPOSE_CHANNELS,
            use_gpu=CELLPOSE_USE_GPU,
            normalize=CELLPOSE_NORMALIZE,
            use_preprocessed_image=CELLPOSE_USE_PREPROCESSED_IMAGE,
            max_image_dimension=CELLPOSE_MAX_IMAGE_DIMENSION,
            batch_size=CELLPOSE_BATCH_SIZE,
            tile_overlap=CELLPOSE_TILE_OVERLAP,
            bsize=CELLPOSE_BSIZE,
        )

    raise ValueError(
        f"Unknown COUNTING_METHOD '{COUNTING_METHOD}'. "
        "Choose 'threshold', 'watershed', 'log', 'dino', 'dino2', or 'cellpose'."
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
    if COUNTING_METHOD == "preprocess":
        output_dir = IMAGES_DIR.parent / PREPROCESS_OUTPUT_DIR
        image_paths = collect_images(IMAGES_DIR)
        print(f"Found {len(image_paths)} image(s) in '{IMAGES_DIR}'. Preprocessing...\n")

        for path in image_paths:
            print(f"  Processing: {path.name} ...", end=" ", flush=True)
            try:
                preprocessed = preprocess_only_image(path)
                output_path = build_preprocess_output_path(output_dir, path)
                save_preprocess_image(preprocessed, output_path)
                print(f"saved to {output_path.name}")
            except Exception as exc:
                print(f"FAILED ({exc})")

        print(f"Processed masks saved to: {output_dir.resolve()}")
        return

    if COUNTING_METHOD == "cellpose" and CELLPOSE_TRAIN_MODEL:
        counter = create_counter(IMAGES_DIR / "training_placeholder.tif")
        try:
            counter.run()
        except RuntimeError as exc:
            print(exc)
            saved_model_path = counter.resolve_saved_model_path()
            if saved_model_path is not None:
                print(
                    "After training in the Cellpose GUI, save the model here: "
                    f"{saved_model_path.resolve()}"
                )
            else:
                print(
                    "After training in the Cellpose GUI, set "
                    "CELLPOSE_CUSTOM_MODEL_NAME to the saved model name so it "
                    "can be reused in future experiments."
                )
        return

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
