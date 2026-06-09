"""Lightweight image-processing lens locator.

This backend is intentionally dependency-light. It gives the project a
deterministic fallback when the trained YOLOv5 segmentation model is not
available on the deployment target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .geometry import mask_to_detection
from .result import LensDetection


@dataclass
class ClassicalConfig:
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.85
    threshold_percentile: float = 85.0
    min_threshold: float = 6.0
    blur_kernel: int = 5


class ClassicalLensDetector:
    """Detect circular lens templates from edge/background contrast."""

    def __init__(self, config: Optional[ClassicalConfig] = None):
        self.config = config or ClassicalConfig()

    def predict(self, image_path: str | Path) -> List[LensDetection]:
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32)
        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        gray = _box_blur(gray, self.config.blur_kernel)
        score = np.abs(gray - _border_median(gray))
        threshold = max(
            self.config.min_threshold,
            float(np.percentile(score, self.config.threshold_percentile)),
        )
        candidate_mask = score >= threshold
        component = _largest_component(
            candidate_mask,
            min_area=int(candidate_mask.size * self.config.min_area_ratio),
            max_area=int(candidate_mask.size * self.config.max_area_ratio),
        )
        if component is None:
            return []
        confidence = _confidence(component)
        return [
            mask_to_detection(
                component,
                image_size=image.size,
                confidence=confidence,
                source="classical",
            )
        ]


def _border_median(gray: np.ndarray, border: int = 8) -> float:
    border = max(1, min(border, gray.shape[0] // 4, gray.shape[1] // 4))
    samples = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    return float(np.median(samples))


def _box_blur(gray: np.ndarray, kernel: int) -> np.ndarray:
    kernel = max(1, int(kernel))
    if kernel == 1:
        return gray
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = np.pad(gray, pad, mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    for dy in range(kernel):
        for dx in range(kernel):
            out += padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return out / float(kernel * kernel)


def _largest_component(mask: np.ndarray, min_area: int, max_area: int) -> Optional[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best_points: List[Tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            points = _collect_component(mask, visited, x, y)
            area = len(points)
            if area < min_area or area > max_area:
                continue
            if area > len(best_points):
                best_points = points
    if not best_points:
        return None
    component = np.zeros(mask.shape, dtype=bool)
    ys = [point[1] for point in best_points]
    xs = [point[0] for point in best_points]
    component[ys, xs] = True
    return component


def _collect_component(mask: np.ndarray, visited: np.ndarray, x0: int, y0: int) -> List[Tuple[int, int]]:
    queue: deque[Tuple[int, int]] = deque([(x0, y0)])
    visited[y0, x0] = True
    points: List[Tuple[int, int]] = []
    height, width = mask.shape
    while queue:
        x, y = queue.popleft()
        points.append((x, y))
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))
    return points


def _confidence(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    aspect = min(width, height) / max(width, height)
    coverage = mask.sum() / max(1.0, width * height)
    return float(np.clip(0.35 + 0.45 * aspect + 0.20 * min(coverage / 0.18, 1.0), 0.0, 0.99))

