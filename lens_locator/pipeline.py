"""Backend selection and inference orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .classical import ClassicalConfig, ClassicalLensDetector
from .result import LensLocalizationResult
from .topography import TopographyConfig
from .yolo import YoloConfig, YoloV5SegmentLensDetector

BackendName = Literal["auto", "yolo", "classical"]


@dataclass
class LensLocatorConfig:
    backend: BackendName = "yolo"
    yolo: YoloConfig = field(default_factory=YoloConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    topography: TopographyConfig = field(default_factory=TopographyConfig)


class LensLocator:
    """Unified lens localization pipeline."""

    def __init__(self, config: Optional[LensLocatorConfig] = None):
        self.config = config or LensLocatorConfig()

    def locate(self, image_path: str | Path) -> LensLocalizationResult:
        start = time.perf_counter()
        backend_used = self.config.backend
        error = None
        try:
            detector = self._build_detector(self.config.backend)
            detections = detector.predict(image_path)
        except Exception as exc:
            if self.config.backend != "auto":
                raise
            error = f"{type(exc).__name__}: {exc}"
            backend_used = "classical"
            detections = ClassicalLensDetector(self.config.classical).predict(image_path)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return LensLocalizationResult(
            image_path=str(image_path),
            detections=detections,
            backend=backend_used,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    def _build_detector(self, backend: BackendName):
        if backend in ("auto", "yolo"):
            return YoloV5SegmentLensDetector(self.config.yolo)
        if backend == "classical":
            return ClassicalLensDetector(self.config.classical)
        raise ValueError(f"Unsupported backend: {backend}")
