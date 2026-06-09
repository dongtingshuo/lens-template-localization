"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .classical import ClassicalConfig
from .pipeline import LensLocatorConfig
from .topography import TopographyConfig
from .yolo import YoloConfig


def load_config(path: str | Path) -> LensLocatorConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    yolo_payload = payload.get("yolo", {})
    classical_payload = payload.get("classical", {})
    topography_payload = payload.get("topography", {})
    return LensLocatorConfig(
        backend=payload.get("backend", "auto"),
        yolo=_build_yolo_config(yolo_payload),
        classical=ClassicalConfig(**classical_payload),
        topography=TopographyConfig(**topography_payload),
    )


def _build_yolo_config(payload: Dict[str, Any]) -> YoloConfig:
    values = dict(payload)
    for key in ("weights", "yolo_root", "data"):
        if key in values:
            values[key] = Path(values[key])
    if "image_size" in values:
        values["image_size"] = tuple(values["image_size"])
    return YoloConfig(**values)
