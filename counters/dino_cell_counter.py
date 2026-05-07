import math

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import frangi
from sklearn.decomposition import PCA

from base_cell_counter import BaseCellCounter


class DinoCellCounter(BaseCellCounter):
    """
    Count cells from DINO ViT-S/16 self-supervised attention maps.

    The primary detection path uses the last-block CLS-to-patch attention,
    followed by thresholding, morphological cleanup, distance transform, and
    non-maximum suppression over candidate peaks.
    """

    method_name = "dino"
    _model_cache: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        image_path: str,
        min_cell_area: int = 50,
        max_cell_area: int = 5000,
        attention_threshold: float = 0.45,
        min_cell_distance: int = 18,
        min_cell_size: int = 40,
        max_image_dimension: int = 896,
        opening_kernel_size: int = 3,
        opening_iterations: int = 1,
        closing_kernel_size: int = 5,
        closing_iterations: int = 1,
        peak_threshold: float = 0.2,
        debug: bool = False,
        model_name: str = "vit_small_patch16_224.dino",
    ):
        super().__init__(
            image_path=image_path,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        self.attention_threshold = attention_threshold
        self.min_cell_distance = min_cell_distance
        self.min_cell_size = min_cell_size
        self.max_image_dimension = max_image_dimension
        self.opening_kernel_size = opening_kernel_size
        self.opening_iterations = opening_iterations
        self.closing_kernel_size = closing_kernel_size
        self.closing_iterations = closing_iterations
        self.peak_threshold = peak_threshold
        self.debug = debug
        self.model_name = model_name
        self.coordinates: list[tuple[int, int]] = []
        self.debug_images: dict[str, np.ndarray] = {}

        self._torch = self._import_torch()
        self.device = self._select_device()
        self.model = self._get_model()

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Return last-layer patch tokens for the given image."""
        tensor, _, patch_grid = self._prepare_dino_input(image)
        with self._torch.no_grad():
            patch_tokens = self._forward_patch_tokens(tensor, patch_grid)

        patch_tokens = patch_tokens[0].detach().cpu().numpy()
        return patch_tokens.reshape(patch_grid[0], patch_grid[1], -1)

    def extract_attention(self, image: np.ndarray) -> np.ndarray:
        """Return the normalized CLS attention map in original image space."""
        tensor, prepared_shape, patch_grid = self._prepare_dino_input(image)

        with self._torch.no_grad():
            x = self._forward_token_sequence(tensor, patch_grid, stop_before_last=True)

            last_block = self.model.blocks[-1]
            x_norm = last_block.norm1(x)
            batch, tokens, channels = x_norm.shape
            heads = last_block.attn.num_heads
            head_dim = channels // heads

            qkv = last_block.attn.qkv(x_norm)
            qkv = qkv.reshape(batch, tokens, 3, heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)

            if hasattr(last_block.attn, "q_norm"):
                q = last_block.attn.q_norm(q)
            if hasattr(last_block.attn, "k_norm"):
                k = last_block.attn.k_norm(k)

            attn = (q * last_block.attn.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

        cls_attention = attn[0, :, 0, 1:].mean(dim=0).detach().cpu().numpy()
        attention_map = cls_attention.reshape(patch_grid)
        attention_map = cv2.resize(
            attention_map,
            (prepared_shape[1], prepared_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        attention_map = self._resize_to_original_space(attention_map, image.shape[:2])
        attention_map -= attention_map.min()
        max_value = attention_map.max()
        if max_value > 0:
            attention_map /= max_value
        return attention_map

    def count_cells(self, image: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        """
        Count cells and return `(count, coordinates)` in original image space.
        """
        all_coordinates: list[tuple[int, int]] = []
        debug_payload = None

        for scale in (0.75, 1.0, 1.25):
            scaled_image = self._resize_by_scale(image, scale)
            feature_map = self._compute_feature_map(scaled_image)
            ridge_map = self._compute_ridge_map(scaled_image)
            combined_map = self._combine_detection_maps(feature_map, ridge_map)

            threshold = np.percentile(combined_map, 75)
            binary_mask = np.uint8(combined_map >= threshold) * 255
            binary_mask = self._cleanup_mask(binary_mask)
            binary_mask = self._filter_components(binary_mask)

            distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            coordinates = self._detect_peaks(distance, binary_mask)
            all_coordinates.extend(
                self._rescale_coordinates_to_original(coordinates, scale)
            )

            if debug_payload is None or abs(scale - 1.0) < 1e-6:
                debug_payload = {
                    "feature_map": self._normalize_to_uint8(feature_map),
                    "ridge_map": self._normalize_to_uint8(ridge_map),
                    "combined_map": self._normalize_to_uint8(combined_map),
                    "binary_mask": binary_mask,
                    "distance_map": self._normalize_to_uint8(distance),
                }

        coordinates = self._suppress_coordinate_overlaps(all_coordinates)

        self.debug_images = debug_payload or {}
        if self.debug:
            self.show_debug_images()

        return len(coordinates), coordinates

    def run(self) -> int:
        """Load -> preprocess -> DINO attention detection -> count."""
        self.load_image()
        self.preprocess()

        if self.original is None:
            raise RuntimeError("Call load_image() first.")

        count, coordinates = self.count_cells(self.original)
        self.coordinates = coordinates
        self.cell_count = count
        self.cell_contours = [self._point_contour(x, y) for x, y in coordinates]
        return count

    def show_debug_images(self) -> None:
        """Display attention and mask intermediates for parameter tuning."""
        for name, image in self.debug_images.items():
            cv2.imshow(f"dino_{name}", image)
        cv2.waitKey(0)

    def _select_device(self):
        """Use CUDA if available, otherwise fall back to CPU."""
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def _get_model(self):
        """Load the pretrained DINO model once per process and device."""
        cache_key = (str(self.device), self.model_name)
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model

        model = self._load_timm_model()
        self._model_cache[cache_key] = model
        return model

    def _load_timm_model(self):
        """Load ViT-S/16 DINO through timm, with torch.hub fallback."""
        try:
            import timm

            model = timm.create_model(self.model_name, pretrained=True)
        except ImportError:
            model = self._torch_hub_fallback()
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not load the DINO model. Check that pretrained weights are "
                "available for timm or torch.hub."
            ) from exc

        model.eval()
        model.to(self.device)
        return model

    def _torch_hub_fallback(self):
        """Fallback when timm is unavailable."""
        try:
            model = self._torch.hub.load(
                "facebookresearch/dino:main",
                "dino_vits16",
                pretrained=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINO via timm or torch.hub. Install `timm` or make "
                "sure torch.hub can access cached DINO weights."
            ) from exc

        model.eval()
        model.to(self.device)
        return model

    def _prepare_dino_input(
        self,
        image: np.ndarray,
    ) -> tuple[object, tuple[int, int], tuple[int, int]]:
        """
        Convert the image into a normalized tensor suitable for ViT-S/16.
        """
        processed = self.gray if self.gray is not None else cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )
        rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        resized = self._resize_for_model(rgb)
        height, width = resized.shape[:2]
        patch_size = self._patch_size()
        patch_height = max(1, height // patch_size)
        patch_width = max(1, width // patch_size)

        tensor = self._torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        mean = self._torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        std = self._torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, (height, width), (patch_height, patch_width)

    def _resize_for_model(self, image: np.ndarray) -> np.ndarray:
        """Resize to a manageable size while keeping ViT patch alignment."""
        height, width = image.shape[:2]
        longest_side = max(height, width)

        if self.max_image_dimension and longest_side > self.max_image_dimension:
            scale = self.max_image_dimension / longest_side
            new_width = max(16, int(round(width * scale)))
            new_height = max(16, int(round(height * scale)))
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
            height, width = image.shape[:2]

        patch_size = self._patch_size()
        aligned_height = max(patch_size, height - (height % patch_size))
        aligned_width = max(patch_size, width - (width % patch_size))

        if aligned_height != height or aligned_width != width:
            image = cv2.resize(
                image,
                (aligned_width, aligned_height),
                interpolation=cv2.INTER_AREA,
            )

        return image

    def _resize_to_original_space(
        self,
        image: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        """Map a resized map back to the original image size."""
        return cv2.resize(
            image,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    def _embed_tokens(self, patch_tokens, patch_grid: tuple[int, int]):
        """Apply class token and positional embedding across timm variants."""
        if hasattr(self.model, "_pos_embed"):
            x = self._embed_tokens_with_interpolated_positional_encoding(
                patch_tokens,
                patch_grid,
            )
        else:
            x = patch_tokens
            cls_token = getattr(self.model, "cls_token", None)
            if cls_token is not None:
                cls_token = cls_token.expand(x.shape[0], -1, -1)
                x = self._torch.cat((cls_token, x), dim=1)

            pos_embed = getattr(self.model, "pos_embed", None)
            if pos_embed is not None:
                pos_embed = self._interpolate_pos_embed(pos_embed, patch_grid)
                x = x + pos_embed

        if hasattr(self.model, "patch_drop"):
            x = self.model.patch_drop(x)
        if hasattr(self.model, "norm_pre"):
            x = self.model.norm_pre(x)
        return x

    def _embed_tokens_with_interpolated_positional_encoding(
        self,
        patch_tokens,
        patch_grid: tuple[int, int],
    ):
        """Use timm internals when available, but with resized position embeddings."""
        x = patch_tokens
        cls_token = getattr(self.model, "cls_token", None)
        if cls_token is not None:
            cls_token = cls_token.expand(x.shape[0], -1, -1)
            x = self._torch.cat((cls_token, x), dim=1)

        pos_embed = getattr(self.model, "pos_embed", None)
        if pos_embed is not None:
            pos_embed = self._interpolate_pos_embed(pos_embed, patch_grid)
            x = x + pos_embed
        return x

    def _interpolate_pos_embed(self, pos_embed, patch_grid: tuple[int, int]):
        """Resize the ViT positional embeddings to the current patch grid."""
        patch_height, patch_width = patch_grid
        num_prefix_tokens = 1 if getattr(self.model, "cls_token", None) is not None else 0

        if pos_embed.shape[1] == patch_height * patch_width + num_prefix_tokens:
            return pos_embed

        prefix = pos_embed[:, :num_prefix_tokens, :]
        patch_pos = pos_embed[:, num_prefix_tokens:, :]
        embed_dim = patch_pos.shape[-1]
        source_grid_size = int(round(math.sqrt(patch_pos.shape[1])))

        if source_grid_size * source_grid_size != patch_pos.shape[1]:
            raise RuntimeError(
                "Could not infer the source positional embedding grid for the DINO model."
            )

        patch_pos = patch_pos.reshape(1, source_grid_size, source_grid_size, embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = self._torch.nn.functional.interpolate(
            patch_pos,
            size=(patch_height, patch_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1,
            patch_height * patch_width,
            embed_dim,
        )

        if num_prefix_tokens:
            return self._torch.cat((prefix, patch_pos), dim=1)
        return patch_pos

    def _threshold_attention(self, attention_map: np.ndarray) -> np.ndarray:
        """Threshold the normalized attention map."""
        binary = np.zeros_like(attention_map, dtype=np.uint8)
        binary[attention_map >= self.attention_threshold] = 255
        return binary

    def _compute_feature_map(self, image: np.ndarray) -> np.ndarray:
        """Project DINO patch features to a saliency map with PCA."""
        features = self.extract_features(image)
        patch_height, patch_width, channels = features.shape
        flat_features = features.reshape(-1, channels)

        component_count = max(1, min(3, flat_features.shape[0], flat_features.shape[1]))
        reduced = PCA(n_components=component_count).fit_transform(flat_features)
        reduced = reduced.reshape(patch_height, patch_width, component_count)

        l2_map = np.linalg.norm(reduced, axis=-1)
        variance_map = np.var(reduced, axis=-1)
        feature_map = 0.5 * self._normalize_map(l2_map) + 0.5 * self._normalize_map(variance_map)

        return self._resize_to_original_space(feature_map, image.shape[:2])

    def _compute_ridge_map(self, image: np.ndarray) -> np.ndarray:
        """Enhance elongated cell-like structures with Frangi filtering."""
        gray = self._preprocess_grayscale(image)
        ridge = frangi(gray.astype(np.float32) / 255.0)
        return self._normalize_map(ridge)

    def _combine_detection_maps(
        self,
        feature_map: np.ndarray,
        ridge_map: np.ndarray,
    ) -> np.ndarray:
        """Blend feature and ridge evidence into the final detection map."""
        combined = 0.6 * self._normalize_map(feature_map) + 0.4 * self._normalize_map(ridge_map)
        return self._normalize_map(combined)

    def _cleanup_mask(self, binary: np.ndarray) -> np.ndarray:
        """Apply configurable opening and closing to the thresholded mask."""
        cleaned = binary.copy()

        if self.opening_iterations > 0:
            open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                self._odd_kernel(self.opening_kernel_size),
            )
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=self.opening_iterations,
            )

        if self.closing_iterations > 0:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                self._odd_kernel(self.closing_kernel_size),
            )
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=self.closing_iterations,
            )

        return cleaned

    def _filter_components(self, binary: np.ndarray) -> np.ndarray:
        """Remove small artifacts before distance-transform peak finding."""
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        filtered = np.zeros_like(binary)

        for label in range(1, component_count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_cell_size:
                continue
            filtered[labels == label] = 255

        return filtered

    def _detect_peaks(
        self,
        distance: np.ndarray,
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Find local maxima on the distance transform."""
        if distance.max() <= 0:
            return []

        peak_coordinates = peak_local_max(
            distance,
            min_distance=self.min_cell_distance,
            threshold_abs=self.peak_threshold,
            labels=(mask > 0),
        )
        return [(int(x), int(y)) for y, x in peak_coordinates]

    def _suppress_coordinate_overlaps(
        self,
        coordinates: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Merge multi-scale detections with a final distance-based NMS."""
        kept: list[tuple[int, int]] = []
        min_distance_sq = self.min_cell_distance ** 2

        for x, y in coordinates:
            if 0 <= x and 0 <= y:
                if all((x - kx) ** 2 + (y - ky) ** 2 >= min_distance_sq for kx, ky in kept):
                    kept.append((x, y))

        return kept

    def _forward_patch_tokens(self, tensor, patch_grid: tuple[int, int]):
        """Return final patch tokens across DINO/timm model variants."""
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(tensor)
            if isinstance(features, dict):
                patch_tokens = features.get("x_norm_patchtokens")
                if patch_tokens is None:
                    prenorm = features.get("x_prenorm")
                    if prenorm is not None:
                        patch_tokens = prenorm[:, 1:, :]
            else:
                patch_tokens = features[:, 1:, :]

            if patch_tokens is not None:
                return patch_tokens

        x = self._forward_token_sequence(tensor, patch_grid, stop_before_last=False)
        return x[:, 1:, :]

    def _forward_token_sequence(
        self,
        tensor,
        patch_grid: tuple[int, int],
        stop_before_last: bool,
    ):
        """Run the transformer token path with broad model-version compatibility."""
        x = self.model.patch_embed(tensor)
        x = self._embed_tokens(x, patch_grid)

        blocks = self.model.blocks[:-1] if stop_before_last else self.model.blocks
        for block in blocks:
            x = block(x)

        norm = getattr(self.model, "norm", None)
        if norm is not None and not stop_before_last:
            x = norm(x)
        return x

    def _point_contour(self, x: int, y: int) -> np.ndarray:
        """Create a small circle contour around a detected center."""
        radius = max(
            3,
            int(round(math.sqrt(max(self.min_cell_area, 1) / math.pi))),
        )
        points = cv2.ellipse2Poly(
            center=(int(x), int(y)),
            axes=(radius, radius),
            angle=0,
            arcStart=0,
            arcEnd=360,
            delta=15,
        )
        return points.reshape((-1, 1, 2))

    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Normalize a float map into 8-bit image space for debugging."""
        max_value = image.max()
        if max_value <= 0:
            return np.zeros_like(image, dtype=np.uint8)
        return np.uint8(np.clip((image / max_value) * 255.0, 0, 255))

    def _normalize_map(self, image: np.ndarray) -> np.ndarray:
        """Normalize a float map to [0, 1]."""
        image = image.astype(np.float32)
        min_value = image.min()
        max_value = image.max()
        if max_value <= min_value:
            return np.zeros_like(image, dtype=np.float32)
        return (image - min_value) / (max_value - min_value)

    def _patch_size(self) -> int:
        """Return the ViT patch size."""
        return 16

    def _odd_kernel(self, size: int) -> tuple[int, int]:
        """Return a positive odd morphology kernel size."""
        size = max(1, int(size))
        if size % 2 == 0:
            size += 1
        return (size, size)

    def _import_torch(self):
        """Import torch with a clear project-specific error message."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for COUNTING_METHOD = 'dino'. Install "
                "`torch` and either `timm` or make cached DINO hub weights "
                "available."
            ) from exc
        return torch

    def _preprocess_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Apply the same grayscale preprocessing path to arbitrary image scales."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _resize_by_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Resize an image for multi-scale detection."""
        if abs(scale - 1.0) < 1e-6:
            return image.copy()

        height, width = image.shape[:2]
        resized_width = max(16, int(round(width * scale)))
        resized_height = max(16, int(round(height * scale)))
        return cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _rescale_coordinates_to_original(
        self,
        coordinates: list[tuple[int, int]],
        scale: float,
    ) -> list[tuple[int, int]]:
        """Map multi-scale detections back to original image coordinates."""
        if abs(scale - 1.0) < 1e-6:
            return coordinates

        return [
            (int(round(x / scale)), int(round(y / scale)))
            for x, y in coordinates
        ]
