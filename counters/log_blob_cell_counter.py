import math

import cv2
import numpy as np

from base_cell_counter import BaseCellCounter


class LoGBlobCellCounter(BaseCellCounter):
    """
    Counts cells with Laplacian of Gaussian blob detection.

    This works best when cells appear as bright, roughly round blobs on a darker
    background. Each detected blob is annotated as a circle.
    """

    method_name = "log"

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
        min_sigma: float = 2.0,
        max_sigma: float = 12.0,
        num_sigma: int = 10,
        response_threshold: float = 0.03,
        overlap_threshold: float = 0.5,
        debug: bool = False,
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.response_threshold = response_threshold
        self.overlap_threshold = overlap_threshold
        self.debug = debug
        self.blobs: list[tuple[float, float, float, float]] = []

    def detect_blobs(self) -> list[tuple[float, float, float, float]]:
        """
        Detect bright LoG blobs.
        Returns tuples of (x, y, radius, response).
        """
        if self.gray is None:
            raise RuntimeError("Call load_image() and preprocess() first.")

        gray_float = self.gray.astype(np.float32) / 255.0
        sigmas = np.linspace(self.min_sigma, self.max_sigma, self.num_sigma)
        responses = []

        for sigma in sigmas:
            blurred = cv2.GaussianBlur(gray_float, (0,0), sigmaX=sigma, sigmaY=sigma)
            laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
            response = -(sigma ** 2) * laplacian
            responses.append(response)

        scale_space = np.stack(responses, axis=0)
        candidates = self._find_scale_space_maxima(scale_space, sigmas)
        candidates = [
            blob for blob in candidates
            if self.min_cell_area <= math.pi * (blob[2] ** 2) <= self.max_cell_area
        ]
        self.blobs = self._suppress_overlapping_blobs(candidates)
        self.cell_contours = [self._circle_contour(x, y, radius) for x, y, radius, _ in self.blobs]
        self.cell_count = len(self.blobs)

        if self.debug:
            self.show_debug_image(scale_space.max(axis=0))

        return self.blobs

    def count_cells(self) -> int:
        """Return the number of LoG blobs detected."""
        if not self.blobs:
            self.detect_blobs()
        return self.cell_count

    def run(self) -> int:
        """Load -> preprocess -> LoG blob detection -> count."""
        self.load_image()
        self.preprocess()
        self.detect_blobs()
        return self.cell_count

    def show_debug_image(self, max_response: np.ndarray) -> None:
        """Display the maximum LoG response image for parameter tuning."""
        response_max = max_response.max()
        normalized = max_response / response_max if response_max else max_response
        cv2.imshow("log_response", normalized)
        cv2.waitKey(0)

    def _find_scale_space_maxima(
        self,
        scale_space: np.ndarray,
        sigmas: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        """Find thresholded local maxima across x, y, and sigma."""
        candidates = []
        scale_count = scale_space.shape[0]

        for scale_index, response in enumerate(scale_space):
            dilated = cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
            local_max = (response == dilated) & (response >= self.response_threshold)

            if scale_index > 0:
                local_max &= response >= scale_space[scale_index - 1]
            if scale_index < scale_count - 1:
                local_max &= response >= scale_space[scale_index + 1]

            ys, xs = np.where(local_max)
            radius = math.sqrt(2.0) * sigmas[scale_index]
            for x, y in zip(xs, ys):
                candidates.append((float(x), float(y), float(radius), float(response[y, x])))

        return sorted(candidates, key=lambda blob: blob[3], reverse=True)

    def _suppress_overlapping_blobs(
        self,
        blobs: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        """Keep stronger blobs when detections overlap too much."""
        kept = []
        for blob in blobs:
            if all(
                self._circle_overlap_fraction(blob, selected) <= self.overlap_threshold
                for selected in kept
            ):
                kept.append(blob)
        return kept

    def _circle_overlap_fraction(
        self,
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        """Approximate overlap as intersection-over-smaller-area."""
        x1, y1, r1, _ = first
        x2, y2, r2, _ = second
        distance = math.hypot(x1 - x2, y1 - y2)

        if distance >= r1 + r2:
            return 0.0
        if distance <= abs(r1 - r2):
            return 1.0

        area1 = r1 ** 2 * math.acos((distance ** 2 + r1 ** 2 - r2 ** 2) / (2 * distance * r1))
        area2 = r2 ** 2 * math.acos((distance ** 2 + r2 ** 2 - r1 ** 2) / (2 * distance * r2))
        area3 = 0.5 * math.sqrt(
            max(
                0.0,
                (-distance + r1 + r2)
                * (distance + r1 - r2)
                * (distance - r1 + r2)
                * (distance + r1 + r2),
            )
        )
        intersection = area1 + area2 - area3
        smaller_area = math.pi * min(r1, r2) ** 2
        return intersection / smaller_area if smaller_area else 0.0

    def _circle_contour(self, x: float, y: float, radius: float) -> np.ndarray:
        """Return a contour approximating one detected circular blob."""
        points = cv2.ellipse2Poly(
            center=(int(round(x)), int(round(y))),
            axes=(int(round(radius)), int(round(radius))),
            angle=0,
            arcStart=0,
            arcEnd=360,
            delta=10,
        )
        return points.reshape((-1, 1, 2))
