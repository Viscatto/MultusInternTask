import cv2
import numpy as np

from base_cell_counter import BaseCellCounter


class WatershedCellCounter(BaseCellCounter):
    """
    Counts cells using watershed segmentation to split touching objects before
    filtering segments by area.
    """

    method_name = "watershed"

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
        foreground_threshold: float = 0.35,
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.foreground_threshold = foreground_threshold
        self.markers: np.ndarray | None = None

    def segment_cells(self) -> np.ndarray:
        """Create watershed markers and return the labeled marker image."""
        if self.gray is None:
            raise RuntimeError("Call load_image() and preprocess() first.")
        if self.original is None:
            raise RuntimeError("Call load_image() first.")

        _, binary = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_background = cv2.dilate(opening, kernel, iterations=3)
        distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_foreground = cv2.threshold(
            distance,
            self.foreground_threshold * distance.max(),
            255,
            0,
        )
        sure_foreground = np.uint8(sure_foreground)
        unknown = cv2.subtract(sure_background, sure_foreground)

        _, markers = cv2.connectedComponents(sure_foreground)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(self.original.copy(), markers)

        self.binary = opening
        self.markers = markers
        return markers

    def count_cells(self) -> int:
        """Filter watershed segments by area and return the cell count."""
        if self.markers is None:
            raise RuntimeError("Call segment_cells() first.")

        contours = []
        for label in np.unique(self.markers):
            if label <= 1:
                continue

            mask = np.zeros(self.markers.shape, dtype=np.uint8)
            mask[self.markers == label] = 255
            area = cv2.countNonZero(mask)
            if not self.min_cell_area <= area <= self.max_cell_area:
                continue

            label_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if label_contours:
                contours.append(max(label_contours, key=cv2.contourArea))

        self.cell_contours = contours
        self.cell_count = len(self.cell_contours)
        return self.cell_count

    def run(self) -> int:
        """Load -> preprocess -> watershed segment -> count."""
        self.load_image()
        self.preprocess()
        self.segment_cells()
        return self.count_cells()
