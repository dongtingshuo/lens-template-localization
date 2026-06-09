"""Synthetic demo input generation for the full pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw

from .topography import TopographyConfig


def create_demo_lens_image(path: str | Path, size: Tuple[int, int] = (360, 320)) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (210, 212, 216))
    draw = ImageDraw.Draw(image)
    width, height = size
    bbox = (60, 40, width - 60, height - 40)
    draw.ellipse(bbox, fill=(196, 199, 208), outline=(86, 91, 112), width=7)
    draw.ellipse((bbox[0] + 34, bbox[1] + 24, bbox[2] - 34, bbox[3] - 24), outline=(170, 175, 190), width=2)
    image.save(path)
    return path


def create_demo_topography_pair(
    directory: str | Path,
    sphere_d: float = 0.4,
    cylinder_d: float = 0.2,
    axis_deg: float = 35.0,
    config: TopographyConfig | None = None,
) -> Tuple[Path, Path]:
    config = config or TopographyConfig()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference_path = directory / "hartmann-reference.png"
    measured_path = directory / "hartmann-measured.png"
    _draw_spot_field(reference_path, sphere_d=0.0, cylinder_d=0.0, axis_deg=0.0, config=config)
    _draw_spot_field(measured_path, sphere_d=sphere_d, cylinder_d=cylinder_d, axis_deg=axis_deg, config=config)
    return reference_path, measured_path


def _draw_spot_field(
    path: Path,
    sphere_d: float,
    cylinder_d: float,
    axis_deg: float,
    config: TopographyConfig,
    size: Tuple[int, int] = (360, 320),
) -> None:
    width, height = size
    cx, cy = width / 2.0, height / 2.0
    radius_px = 116.0
    image = Image.new("RGB", size, (26, 30, 38))
    draw = ImageDraw.Draw(image)
    draw.ellipse((cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px), outline=(88, 99, 122), width=3)
    matrix = _power_matrix(sphere_d, cylinder_d, axis_deg)
    mm_per_px = config.lens_diameter_mm / (2.0 * radius_px)
    pitch = config.spot_pitch_px
    n = int(radius_px // pitch)
    for iy in range(-n, n + 1):
        for ix in range(-n, n + 1):
            x0 = cx + ix * pitch
            y0 = cy + iy * pitch
            if (x0 - cx) ** 2 + (y0 - cy) ** 2 > (radius_px * 0.9) ** 2:
                continue
            field_m = ((x0 - cx) * mm_per_px / 1000.0, (y0 - cy) * mm_per_px / 1000.0)
            angle_x = matrix[0][0] * field_m[0] + matrix[0][1] * field_m[1]
            angle_y = matrix[1][0] * field_m[0] + matrix[1][1] * field_m[1]
            dx = angle_x * config.sensor_focal_length_mm / config.pixel_size_mm
            dy = angle_y * config.sensor_focal_length_mm / config.pixel_size_mm
            x = x0 + dx
            y = y0 + dy
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(238, 242, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _power_matrix(sphere_d: float, cylinder_d: float, axis_deg: float):
    axis = math.radians(axis_deg)
    c = math.cos(axis)
    s = math.sin(axis)
    rotation = ((c, -s), (s, c))
    p_axis = sphere_d - cylinder_d / 2.0
    p_cross = sphere_d + cylinder_d / 2.0
    return (
        (
            rotation[0][0] * p_axis * rotation[0][0] + rotation[0][1] * p_cross * rotation[0][1],
            rotation[0][0] * p_axis * rotation[1][0] + rotation[0][1] * p_cross * rotation[1][1],
        ),
        (
            rotation[1][0] * p_axis * rotation[0][0] + rotation[1][1] * p_cross * rotation[0][1],
            rotation[1][0] * p_axis * rotation[1][0] + rotation[1][1] * p_cross * rotation[1][1],
        ),
    )
