"""Geometry utilities shared by classical and neural detectors."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .result import BBox, LensDetection, Point


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(points: Sequence[Point]) -> Point:
    if len(points) < 3:
        return mean_point(points)
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return mean_point(points)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (3.0 * area2)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (3.0 * area2)
    return float(cx), float(cy)


def mean_point(points: Sequence[Point]) -> Point:
    if not points:
        return 0.0, 0.0
    pts = np.asarray(points, dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def bbox_from_points(points: Sequence[Point]) -> BBox:
    pts = np.asarray(points, dtype=float)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def pca_angle_deg(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    return float(math.degrees(math.atan2(axis[1], axis[0])))


def yolo_segment_to_points(values: Sequence[float], width: int, height: int) -> List[Point]:
    """Convert a YOLO segmentation row after class id into pixel coordinates."""

    if len(values) % 2:
        raise ValueError("YOLO segmentation coordinates must contain x/y pairs.")
    points: List[Point] = []
    for x_norm, y_norm in zip(values[0::2], values[1::2]):
        points.append((float(x_norm) * width, float(y_norm) * height))
    return points


def detection_from_polygon(
    points: Sequence[Point],
    image_size: Tuple[int, int],
    confidence: float = 1.0,
    source: str = "annotation",
    class_name: str = "glass",
) -> LensDetection:
    bbox = bbox_from_points(points)
    cx, cy = polygon_centroid(points)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    radius = (width + height) / 4.0
    return LensDetection(
        bbox_xyxy=bbox,
        center_xy=(cx, cy),
        radius_px=float(radius),
        area_px=polygon_area(points),
        confidence=float(confidence),
        angle_deg=pca_angle_deg(points),
        class_name=class_name,
        source=source,
        image_size=image_size,
        contour=[(float(x), float(y)) for x, y in points],
    )


def mask_to_detection(
    mask: np.ndarray,
    image_size: Tuple[int, int],
    confidence: float,
    source: str,
    class_name: str = "glass",
) -> LensDetection:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot create detection from an empty mask.")
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    width = max(1.0, x2 - x1 + 1.0)
    height = max(1.0, y2 - y1 + 1.0)
    bbox_center = (x1 + width / 2.0, y1 + height / 2.0)
    radius = (width + height) / 4.0
    points = list(zip(xs.astype(float), ys.astype(float)))
    return LensDetection(
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=(float(bbox_center[0]), float(bbox_center[1])),
        radius_px=float(radius),
        area_px=float(mask.sum()),
        confidence=float(confidence),
        angle_deg=pca_angle_deg(points),
        class_name=class_name,
        source=source,
        image_size=image_size,
        contour=_ellipse_contour((x1, y1, x2, y2)),
        metadata={"mask_area_ratio": float(mask.mean())},
    )


def _ellipse_contour(bbox: BBox, samples: int = 72) -> List[Point]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    return [
        (float(cx + math.cos(theta) * rx), float(cy + math.sin(theta) * ry))
        for theta in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    ]
