"""Visualization helpers for CLI outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .result import LensLocalizationResult
from .topography import RefractivePowerMap


def save_overlay(image_path: str | Path, result: LensLocalizationResult, output_path: str | Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx, cy = detection.center_xy
        draw.ellipse((x1, y1, x2, y2), outline=(20, 184, 166, 255), width=3)
        draw.rectangle((x1, y1, x2, y2), outline=(37, 99, 235, 220), width=2)
        draw.line((cx - 8, cy, cx + 8, cy), fill=(239, 68, 68, 255), width=2)
        draw.line((cx, cy - 8, cx, cy + 8), fill=(239, 68, 68, 255), width=2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(15, 23, 42, 255))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_topography_overlay(
    image_path: str | Path,
    power_map: RefractivePowerMap,
    output_path: str | Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    span = max(abs(power_map.max_power_d - power_map.min_power_d), 1e-6)
    for sample in power_map.samples:
        x, y = sample.image_xy
        ratio = (sample.local_power_d - power_map.min_power_d) / span
        color = _power_color(ratio)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(15, 23, 42, 220))
    label = (
        f"SE {power_map.sphere_equivalent_d:+.2f}D  "
        f"CYL {power_map.cylinder_d:+.2f}D  AX {power_map.axis_deg:.0f}"
    )
    draw.rectangle((8, 8, 280, 34), fill=(248, 250, 252, 230))
    draw.text((14, 14), label, fill=(15, 23, 42, 255))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _power_color(ratio: float):
    ratio = max(0.0, min(1.0, float(ratio)))
    red = int(37 + 218 * ratio)
    blue = int(235 - 198 * ratio)
    green = int(99 + 65 * (1.0 - abs(ratio - 0.5) * 2.0))
    return red, green, blue, 210
