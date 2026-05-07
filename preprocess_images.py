"""
Preprocess brightfield MSC microscopy images.

Pipeline:
1. Load 8-bit grayscale images from an input directory
2. Min-max normalize intensities
3. Correct background with large-Gaussian subtraction
4. Enhance contrast with CLAHE
5. Reduce noise with median + Gaussian filtering
6. Save preprocessed images to the output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_images(input_dir: Path) -> list[Path]:
    """Return supported images from the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load one image as 8-bit grayscale."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Rescale intensities to the full 8-bit range."""
    image_float = image.astype(np.float32)
    min_value = float(image_float.min())
    max_value = float(image_float.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image_float - min_value) / (max_value - min_value)
    return np.uint8(np.clip(normalized * 255.0, 0, 255))


def correct_background(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Approximate rolling-ball correction with a large Gaussian background model."""
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = cv2.subtract(image, background)
    return min_max_normalize(corrected)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance local contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    return clahe.apply(image)


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Smooth fine texture while preserving cell boundaries."""
    median = cv2.medianBlur(image, 3)
    return cv2.GaussianBlur(median, (0, 0), sigmaX=1.0, sigmaY=1.0)

def process_image(image_path: Path) -> np.ndarray:
    """Run the full preprocessing pipeline and return the processed image."""
    original = load_grayscale_image(image_path)
    normalized = min_max_normalize(original)
    corrected = correct_background(normalized)
    enhanced = enhance_contrast(corrected)
    denoised = reduce_noise(enhanced)
    return denoised


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Write a preprocessed image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def build_output_path(output_dir: Path, image_path: Path) -> Path:
    """Create the output path for one preprocessed image."""
    return output_dir / f"{image_path.stem}.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess brightfield MSC microscopy images.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing grayscale microscopy images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where preprocessed images will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing over all images in the input directory."""
    args = parse_args()
    image_paths = collect_images(args.input_dir)

    if not image_paths:
        print(f"No supported images found in: {args.input_dir.resolve()}")
        return

    for image_path in image_paths:
        try:
            preprocessed = process_image(image_path)
            output_path = build_output_path(args.output_dir, image_path)
            save_image(preprocessed, output_path)
            print(f"{image_path.name}: saved to {output_path}")
        except Exception as exc:
            print(f"{image_path.name}: FAILED ({exc})")


if __name__ == "__main__":
    main()