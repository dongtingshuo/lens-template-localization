"""Visualization helpers for CLI outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .result import LensLocalizationResult


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

