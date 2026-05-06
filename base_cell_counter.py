import math
from pathlib import Path

import cv2
import numpy as np


class BaseCellCounter:
    """Shared image loading, annotation, and saving behavior for cell counters."""

    method_name = "base"

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
    ):
        self.image_path = Path(image_path)
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area

        self.original: np.ndarray | None = None
        self.gray: np.ndarray | None = None
        self.binary: np.ndarray | None = None
        self.cell_count: int = 0
        self.cell_contours: list = []

    def load_image(self) -> np.ndarray:
        """Load image from disk and return a BGR array."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        img = cv2.imread(str(self.image_path))
        if img is None:
            raise ValueError(f"Could not decode image: {self.image_path}")

        self.original = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def preprocess(self) -> np.ndarray:
        """
        Denoise and enhance contrast before counting.
        Returns the preprocessed grayscale image.
        """
        if self.gray is None:
            raise RuntimeError("Call load_image() first.")

        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        self.gray = enhanced
        return enhanced

    def save_annotated(self, output_path: str | None = None) -> Path:
        """
        Draw detected cell outlines on the original image, write to disk,
        and return the path.
        """
        if self.original is None:
            raise RuntimeError("Run the full pipeline before saving.")

        annotated = self.original.copy()
        if self.cell_contours:
            cv2.drawContours(annotated, self.cell_contours, -1, (0, 255, 0), 2)

        self._draw_annotation_legend(annotated)

        if output_path is None:
            stem = self.image_path.stem
            suffix = self.image_path.suffix
            output_path = self.image_path.with_name(f"{stem}_annotated{suffix}")

        out = Path(output_path)
        cv2.imwrite(str(out), annotated)
        return out

    def _draw_annotation_legend(self, image: np.ndarray) -> None:
        """Draw cell count plus min/max area reference circles."""
        green = (0, 255, 0)
        white = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        cv2.putText(
            image,
            f"Cells: {self.cell_count}",
            (10, 30),
            font,
            1.0,
            green,
            2,
        )

        min_radius = max(2, int(round(math.sqrt(self.min_cell_area / math.pi))))
        max_radius = max(2, int(round(math.sqrt(self.max_cell_area / math.pi))))

        y = 55 + max_radius
        min_center = (10 + max_radius, y)
        max_center = (min_center[0] + max_radius + min_radius + 95, y)

        cv2.circle(image, min_center, min_radius, green, 2)
        cv2.circle(image, max_center, max_radius, green, 2)

        cv2.putText(
            image,
            f"min {self.min_cell_area}",
            (min_center[0] - max_radius, y + max_radius + 22),
            font,
            font_scale,
            white,
            thickness,
        )
        cv2.putText(
            image,
            f"max {self.max_cell_area}",
            (max_center[0] - max_radius, y + max_radius + 22),
            font,
            font_scale,
            white,
            thickness,
        )
