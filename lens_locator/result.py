"""Result models for lens localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LensDetection:
    """Single lens detection with geometry ready for downstream calibration."""

    bbox_xyxy: BBox
    center_xy: Point
    radius_px: float
    area_px: float
    confidence: float
    angle_deg: float = 0.0
    class_name: str = "glass"
    source: str = "classical"
    image_size: Tuple[int, int] = (0, 0)
    contour: List[Point] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def diameter_px(self) -> float:
        return self.radius_px * 2.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["diameter_px"] = self.diameter_px
        return data


@dataclass(frozen=True)
class LensLocalizationResult:
    """Container returned by the pipeline."""

    image_path: str
    detections: Sequence[LensDetection]
    backend: str
    elapsed_ms: float
    error: Optional[str] = None

    @property
    def best(self) -> Optional[LensDetection]:
        if not self.detections:
            return None
        return max(self.detections, key=lambda item: item.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "backend": self.backend,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "error": self.error,
            "detections": [item.to_dict() for item in self.detections],
        }

