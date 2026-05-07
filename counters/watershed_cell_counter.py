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
        opening_kernel_size: int = 3,
        opening_iterations: int = 1,
        debug: bool = False,
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.foreground_threshold = foreground_threshold
        self.opening_kernel_size = opening_kernel_size
        self.opening_iterations = opening_iterations
        self.debug = debug
        self.markers: np.ndarray | None = None

    def segment_cells(self) -> np.ndarray:
        """Create watershed markers and return the labeled marker image."""
        if self.gray is None:
            raise RuntimeError("Call load_image() and preprocess() first.")
        if self.original is None:
            raise RuntimeError("Call load_image() first.")

        _, binary = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_size = self._validated_odd_kernel_size(self.opening_kernel_size)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        opening = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=max(0, self.opening_iterations),
        )

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

        if self.debug:
            self.show_debug_images(
                distance=distance,
                binary=binary,
                opening=opening,
                sure_foreground=sure_foreground,
                markers=markers,
            )

        self.binary = opening
        self.markers = markers
        return markers

    def show_debug_images(
        self,
        distance: np.ndarray,
        binary: np.ndarray,
        opening: np.ndarray,
        sure_foreground: np.ndarray,
        markers: np.ndarray,
    ) -> None:
        """Display intermediate watershed images for parameter tuning."""
        distance_max = distance.max()
        markers_max = markers.max()

        cv2.imshow("distance", distance / distance_max if distance_max else distance)
        cv2.imshow("binary", binary)
        cv2.imshow("opening", opening)
        cv2.imshow("sure_fg", sure_foreground)
        cv2.imshow(
            "markers",
            (markers.astype(np.float32) / markers_max)
            if markers_max
            else markers.astype(np.float32),
        )
        cv2.waitKey(0)

    def _validated_odd_kernel_size(self, kernel_size: int) -> int:
        """OpenCV morphology kernels should be positive odd dimensions."""
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

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
