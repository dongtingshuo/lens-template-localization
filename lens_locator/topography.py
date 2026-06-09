"""Refractive power topography from Hartmann-style spot displacement fields."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .result import LensDetection, Point


@dataclass(frozen=True)
class TopographyConfig:
    """Physical and image-processing parameters for refractive measurement."""

    pixel_size_mm: float = 0.05
    sensor_focal_length_mm: float = 25.0
    lens_diameter_mm: float = 70.0
    spot_pitch_px: float = 32.0
    spot_threshold_percentile: float = 92.0
    min_spot_area_px: int = 4
    max_spot_area_px: int = 800
    max_pair_distance_px: float = 48.0


@dataclass(frozen=True)
class SpotCentroid:
    x: float
    y: float
    area_px: int
    intensity: float

    @property
    def xy(self) -> Point:
        return self.x, self.y

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PowerSample:
    image_xy: Point
    reference_xy: Point
    field_xy_mm: Point
    displacement_xy_px: Point
    local_power_d: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RefractivePowerMap:
    """Power topography summary and per-sample measurements."""

    samples: Sequence[PowerSample]
    sphere_equivalent_d: float
    cylinder_d: float
    axis_deg: float
    principal_powers_d: Tuple[float, float]
    mean_power_d: float
    min_power_d: float
    max_power_d: float
    rms_fit_error_d: float
    matrix_d: Tuple[Tuple[float, float], Tuple[float, float]]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["samples"] = [sample.to_dict() for sample in self.samples]
        return data


class RefractiveTopographyEstimator:
    """Estimate lens power map from reference and measured spot positions."""

    def __init__(self, config: Optional[TopographyConfig] = None):
        self.config = config or TopographyConfig()

    def measure(
        self,
        image_path: str | Path,
        reference_path: str | Path | None = None,
        lens_detection: LensDetection | None = None,
    ) -> RefractivePowerMap:
        measured_spots = detect_spots(image_path, lens_detection, self.config)
        if reference_path:
            reference_spots = [spot.xy for spot in detect_spots(reference_path, lens_detection, self.config)]
        else:
            reference_spots = generate_reference_grid(measured_spots, lens_detection, self.config)
        pairs = pair_spots(reference_spots, [spot.xy for spot in measured_spots], self.config.max_pair_distance_px)
        return estimate_power_map(pairs, lens_detection, self.config)


def detect_spots(
    image_path: str | Path,
    lens_detection: LensDetection | None,
    config: TopographyConfig,
) -> List[SpotCentroid]:
    image = Image.open(image_path).convert("L")
    gray = np.asarray(image, dtype=np.float32)
    roi = _roi_mask(gray.shape, lens_detection)
    values = gray[roi]
    if values.size == 0:
        return []
    threshold = max(float(np.percentile(values, config.spot_threshold_percentile)), float(values.mean() + values.std()))
    mask = (gray >= threshold) & roi
    return _components_to_spots(gray, mask, config)


def generate_reference_grid(
    measured_spots: Sequence[SpotCentroid],
    lens_detection: LensDetection | None,
    config: TopographyConfig,
) -> List[Point]:
    if lens_detection:
        x1, y1, x2, y2 = lens_detection.bbox_xyxy
        cx, cy = lens_detection.center_xy
        rx = max(1.0, (x2 - x1) / 2.0)
        ry = max(1.0, (y2 - y1) / 2.0)
    elif measured_spots:
        xs = [spot.x for spot in measured_spots]
        ys = [spot.y for spot in measured_spots]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx, ry = max(1.0, (x2 - x1) / 2.0), max(1.0, (y2 - y1) / 2.0)
    else:
        return []

    pitch = max(4.0, float(config.spot_pitch_px))
    points: List[Point] = []
    nx = int(math.floor(rx / pitch))
    ny = int(math.floor(ry / pitch))
    for iy in range(-ny, ny + 1):
        for ix in range(-nx, nx + 1):
            x = cx + ix * pitch
            y = cy + iy * pitch
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 0.92:
                points.append((float(x), float(y)))
    return points


def pair_spots(
    reference_points: Sequence[Point],
    measured_points: Sequence[Point],
    max_distance_px: float,
) -> List[Tuple[Point, Point]]:
    unused = set(range(len(measured_points)))
    pairs: List[Tuple[Point, Point]] = []
    for ref in sorted(reference_points, key=lambda point: (point[1], point[0])):
        best_index = None
        best_distance = float(max_distance_px)
        for index in list(unused):
            distance = _distance(ref, measured_points[index])
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None:
            unused.remove(best_index)
            pairs.append((ref, measured_points[best_index]))
    return pairs


def estimate_power_map(
    spot_pairs: Sequence[Tuple[Point, Point]],
    lens_detection: LensDetection | None,
    config: TopographyConfig,
) -> RefractivePowerMap:
    if len(spot_pairs) < 3:
        raise ValueError("At least three matched Hartmann spots are required for topography estimation.")

    center = _measurement_center(spot_pairs, lens_detection)
    scale_mm_per_px = _mm_per_px(spot_pairs, lens_detection, config)
    rows = []
    theta = []
    samples: List[PowerSample] = []
    for reference_xy, measured_xy in spot_pairs:
        field_px = (reference_xy[0] - center[0], reference_xy[1] - center[1])
        field_m = np.asarray(field_px, dtype=float) * scale_mm_per_px / 1000.0
        if float(np.linalg.norm(field_m)) < 1e-9:
            continue
        displacement_px = (measured_xy[0] - reference_xy[0], measured_xy[1] - reference_xy[1])
        angle = np.asarray(displacement_px, dtype=float) * config.pixel_size_mm / config.sensor_focal_length_mm
        rows.append([field_m[0], field_m[1], 0.0, 0.0])
        rows.append([0.0, 0.0, field_m[0], field_m[1]])
        theta.extend([angle[0], angle[1]])
        local_power = float(np.dot(angle, field_m) / np.dot(field_m, field_m))
        samples.append(
            PowerSample(
                image_xy=(float(measured_xy[0]), float(measured_xy[1])),
                reference_xy=(float(reference_xy[0]), float(reference_xy[1])),
                field_xy_mm=(float(field_m[0] * 1000.0), float(field_m[1] * 1000.0)),
                displacement_xy_px=(float(displacement_px[0]), float(displacement_px[1])),
                local_power_d=local_power,
            )
        )

    if len(samples) < 3:
        raise ValueError("Matched spots are too close to the optical center for stable power estimation.")

    coeffs, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(theta, dtype=float), rcond=None)
    raw_matrix = np.asarray([[coeffs[0], coeffs[1]], [coeffs[2], coeffs[3]]], dtype=float)
    power_matrix = (raw_matrix + raw_matrix.T) / 2.0
    values, vectors = np.linalg.eigh(power_matrix)
    order = np.argsort(values)
    p_min = float(values[order[0]])
    p_max = float(values[order[1]])
    axis_vector = vectors[:, order[0]]
    axis_deg = float((math.degrees(math.atan2(axis_vector[1], axis_vector[0])) + 180.0) % 180.0)
    residuals = []
    for sample in samples:
        field_m = np.asarray(sample.field_xy_mm, dtype=float) / 1000.0
        measured_angle = np.asarray(sample.displacement_xy_px, dtype=float) * config.pixel_size_mm / config.sensor_focal_length_mm
        residual_angle = measured_angle - power_matrix @ field_m
        residuals.append(float(np.linalg.norm(residual_angle) / max(np.linalg.norm(field_m), 1e-9)))
    powers = [sample.local_power_d for sample in samples]
    return RefractivePowerMap(
        samples=samples,
        sphere_equivalent_d=float((p_max + p_min) / 2.0),
        cylinder_d=float(p_max - p_min),
        axis_deg=axis_deg,
        principal_powers_d=(p_min, p_max),
        mean_power_d=float(np.mean(powers)),
        min_power_d=float(np.min(powers)),
        max_power_d=float(np.max(powers)),
        rms_fit_error_d=float(math.sqrt(np.mean(np.square(residuals)))),
        matrix_d=((float(power_matrix[0, 0]), float(power_matrix[0, 1])), (float(power_matrix[1, 0]), float(power_matrix[1, 1]))),
        metadata={
            "matched_spot_count": len(samples),
            "scale_mm_per_px": scale_mm_per_px,
            "pixel_size_mm": config.pixel_size_mm,
            "sensor_focal_length_mm": config.sensor_focal_length_mm,
        },
    )


def _roi_mask(shape: Tuple[int, int], lens_detection: LensDetection | None) -> np.ndarray:
    height, width = shape
    if lens_detection is None:
        return np.ones((height, width), dtype=bool)
    x1, y1, x2, y2 = lens_detection.bbox_xyxy
    cx, cy = lens_detection.center_xy
    rx = max(1.0, (x2 - x1) / 2.0) * 1.15
    ry = max(1.0, (y2 - y1) / 2.0) * 1.15
    y_grid, x_grid = np.ogrid[:height, :width]
    return ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 <= 1.0


def _components_to_spots(gray: np.ndarray, mask: np.ndarray, config: TopographyConfig) -> List[SpotCentroid]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    spots: List[SpotCentroid] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            points = _collect_component(mask, visited, x, y)
            area = len(points)
            if area < config.min_spot_area_px or area > config.max_spot_area_px:
                continue
            ys = np.asarray([point[1] for point in points], dtype=int)
            xs = np.asarray([point[0] for point in points], dtype=int)
            weights = gray[ys, xs].astype(float)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            spots.append(
                SpotCentroid(
                    x=float(np.dot(xs, weights) / weight_sum),
                    y=float(np.dot(ys, weights) / weight_sum),
                    area_px=area,
                    intensity=float(weights.mean()),
                )
            )
    return sorted(spots, key=lambda spot: (spot.y, spot.x))


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


def _measurement_center(
    spot_pairs: Sequence[Tuple[Point, Point]],
    lens_detection: LensDetection | None,
) -> Point:
    if lens_detection:
        return lens_detection.center_xy
    refs = np.asarray([pair[0] for pair in spot_pairs], dtype=float)
    return float(refs[:, 0].mean()), float(refs[:, 1].mean())


def _mm_per_px(
    spot_pairs: Sequence[Tuple[Point, Point]],
    lens_detection: LensDetection | None,
    config: TopographyConfig,
) -> float:
    if lens_detection and lens_detection.diameter_px > 0:
        return config.lens_diameter_mm / lens_detection.diameter_px
    refs = np.asarray([pair[0] for pair in spot_pairs], dtype=float)
    span = max(float(np.ptp(refs[:, 0])), float(np.ptp(refs[:, 1])), 1.0)
    return config.lens_diameter_mm / span


def _distance(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))
