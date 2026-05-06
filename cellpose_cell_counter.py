import os
from pathlib import Path

import cv2
import numpy as np

from base_cell_counter import BaseCellCounter

class CellposeCellCounter(BaseCellCounter):
    """
    Counts cells using Cellpose segmentation.

    Cellpose is imported lazily so the other counters can still be used when
    Cellpose is not installed.
    """

    method_name = "cellpose"

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
        model_type: str = "cyto3",
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        channels: list[int] | tuple[int, int] = (0, 0),
        use_gpu: bool = False,
        normalize: bool = True,
        use_preprocessed_image: bool = False,
        max_image_dimension: int | None = 1024,
        batch_size: int = 1,
        tile_overlap: float = 0.1,
        bsize: int = 256,
        train_model: bool = False,
        custom_models_dir: str | None = None,
        custom_model_name: str | None = None,
        pretrained_model_path: str | None = None,
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.model_type = model_type
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.channels = list(channels)
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.use_preprocessed_image = use_preprocessed_image
        self.max_image_dimension = max_image_dimension
        self.batch_size = batch_size
        self.tile_overlap = tile_overlap
        self.bsize = bsize
        self.train_model = train_model
        self.custom_models_dir = (
            Path(custom_models_dir) if custom_models_dir is not None else None
        )
        self.custom_model_name = custom_model_name
        self.pretrained_model_path = (
            Path(pretrained_model_path) if pretrained_model_path else None
        )
        self.masks: np.ndarray | None = None

    def segment_cells(self) -> np.ndarray:
        """Run Cellpose and return the labeled mask image."""
        if self.gray is None:
            raise RuntimeError("Call load_image() first.")
        if self.train_model:
            raise RuntimeError(
                "CELLPOSE_TRAIN_MODEL is True. Launch the Cellpose GUI training "
                "flow instead of running batch counting."
            )

        model = self._create_model()
        image, scale = self._prepare_cellpose_image(self.gray)
        effective_bsize = self._effective_bsize()

        eval_kwargs = {
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "tile_overlap": self.tile_overlap,
            "bsize": effective_bsize,
        }

        try:
            masks, _, _ = model.eval(image, **eval_kwargs)
        except TypeError:
            eval_kwargs["channels"] = self.channels
            masks, _, _ = model.eval(image, **eval_kwargs)
        except ValueError:
            masks, _, _, _ = model.eval(image, **eval_kwargs)

        masks = masks.astype(np.int32)
        if scale != 1.0:
            masks = cv2.resize(
                masks,
                (self.gray.shape[1], self.gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        self.masks = masks
        self.binary = np.uint8(self.masks > 0) * 255
        return self.masks

    def count_cells(self) -> int:
        """Filter Cellpose masks by area and return the cell count."""
        if self.masks is None:
            raise RuntimeError("Call segment_cells() first.")

        contours = []
        for label in np.unique(self.masks):
            if label == 0:
                continue

            mask = np.zeros(self.masks.shape, dtype=np.uint8)
            mask[self.masks == label] = 255
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
        """Load -> optional preprocess -> Cellpose segment -> count."""
        if self.train_model:
            self.launch_training_gui()
            raise RuntimeError(
                "Cellpose training GUI launched. Finish training there, then set "
                "CELLPOSE_TRAIN_MODEL = False and point to the saved model for "
                "future experiments."
            )

        self.load_image()
        if self.use_preprocessed_image:
            self.preprocess()
        self.segment_cells()
        return self.count_cells()

    def _create_model(self):
        """Create a Cellpose model, with a clear error if Cellpose is missing."""
        try:
            from cellpose import models
        except ImportError as exc:
            raise ImportError(
                "Cellpose is not installed. Install it in your Python environment "
                "before using COUNTING_METHOD = 'cellpose'."
            ) from exc

        if self.pretrained_model_path is not None:
            if not self.pretrained_model_path.exists():
                raise FileNotFoundError(
                    f"Cellpose custom model not found: {self.pretrained_model_path}"
                )
            return models.CellposeModel(
                pretrained_model=os.fspath(self.pretrained_model_path),
                gpu=self.use_gpu,
            )

        try:
            return models.CellposeModel(
                pretrained_model=self.model_type,
                gpu=self.use_gpu,
            )
        except TypeError:
            return models.CellposeModel(
                model_type=self.model_type,
                gpu=self.use_gpu,
            )

    def _prepare_cellpose_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Optionally downscale large images before Cellpose inference."""
        if not self.max_image_dimension:
            return image, 1.0

        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.max_image_dimension:
            return image, 1.0

        scale = self.max_image_dimension / longest_side
        resized = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _effective_bsize(self) -> int:
        """
        Cellpose v4 `cpsam` expects the standard 256 tile size.
        Smaller custom tile sizes can trigger tensor-shape mismatches.
        """
        return 256 if self.bsize != 256 else self.bsize

    def launch_training_gui(self) -> None:
        """
        Launch the Cellpose GUI so a custom model can be trained and saved into
        a project-local models directory for reuse in future experiments.
        """
        models_dir = self._resolve_custom_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(models_dir)

        try:
            from cellpose.gui import gui
        except ImportError as exc:
            raise RuntimeError(
                "Cellpose GUI is not available. Install Cellpose with GUI support "
                "in your Python environment."
            ) from exc

        try:
            gui.run()
        except Exception as exc:
            raise RuntimeError(
                "Cellpose GUI could not be launched."
            ) from exc

    def resolve_saved_model_path(self) -> Path | None:
        """Return the expected project-local path for a trained custom model."""
        if not self.custom_model_name:
            return None
        return self._resolve_custom_models_dir() / self.custom_model_name

    def _resolve_custom_models_dir(self) -> Path:
        """Return the directory where custom Cellpose models should live."""
        if self.custom_models_dir is not None:
            return self.custom_models_dir
        return self.image_path.parent.parent / "cellpose_models"
