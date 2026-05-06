import cv2
import numpy as np

from base_cell_counter import BaseCellCounter


class ThresholdCellCounter(BaseCellCounter):
    """
    Counts cells by thresholding the image and filtering connected contours by
    area.
    """

    method_name = "threshold"

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
        threshold_method: str = "otsu",       # "otsu" | "adaptive" | "simple"
        threshold_value: int = 127,            # used only when method="simple"
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def apply_threshold(self) -> np.ndarray:
        """
        Convert the grayscale image to a binary mask using the chosen method.
        Returns the binary image.
        """
        if self.gray is None:
            raise RuntimeError("Call load_image() and preprocess() first.")

        if self.threshold_method == "otsu":
            _, binary = cv2.threshold(
                self.gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        elif self.threshold_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                self.gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=11,
                C=2,
            )

        elif self.threshold_method == "simple":
            _, binary = cv2.threshold(
                self.gray, self.threshold_value, 255,
                cv2.THRESH_BINARY_INV
            )

        else:
            raise ValueError(
                f"Unknown threshold_method '{self.threshold_method}'. "
                "Choose 'otsu', 'adaptive', or 'simple'."
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        self.binary = binary
        return binary

    def count_cells(self) -> int:
        """
        Find contours in the binary mask, filter by area, and return the cell
        count.
        """
        if self.binary is None:
            raise RuntimeError("Call apply_threshold() first.")

        contours, _ = cv2.findContours(
            self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.cell_contours = [
            c for c in contours
            if self.min_cell_area <= cv2.contourArea(c) <= self.max_cell_area
        ]
        self.cell_count = len(self.cell_contours)
        return self.cell_count

    def run(self) -> int:
        """Load -> preprocess -> threshold -> count. Returns the cell count."""
        self.load_image()
        self.preprocess()
        self.apply_threshold()
        return self.count_cells()
