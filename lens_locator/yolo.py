"""Optional YOLOv5 segmentation backend."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import mask_to_detection
from .result import LensDetection


@dataclass
class YoloConfig:
    weights: Path = Path("artifacts/weights/lens-yolov5-seg.pt")
    yolo_root: Path = Path("third_party/yolov5")
    data: Path = Path("artifacts/datasets/lens.yaml")
    image_size: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "0"


class YoloV5SegmentLensDetector:
    """Adapter around the local YOLOv5 segmentation workspace."""

    def __init__(self, config: Optional[YoloConfig] = None):
        self.config = config or YoloConfig()
        if not self.config.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.config.weights}")
        if not self.config.yolo_root.exists():
            raise FileNotFoundError(f"YOLOv5 root not found: {self.config.yolo_root}")
        self._load_runtime()

    def predict(self, image_path: str | Path) -> List[LensDetection]:
        image_path = Path(image_path)
        dataset = self.LoadImages(
            str(image_path),
            img_size=self.image_size,
            stride=self.stride,
            auto=self.pt,
        )
        detections: List[LensDetection] = []
        for _, im, im0s, _, _ in dataset:
            im_tensor = self.torch.from_numpy(im).to(self.model.device)
            im_tensor = im_tensor.half() if self.model.fp16 else im_tensor.float()
            im_tensor /= 255.0
            if len(im_tensor.shape) == 3:
                im_tensor = im_tensor[None]
            pred, proto = self.model(im_tensor, augment=False, visualize=False)[:2]
            pred = self.non_max_suppression(
                pred,
                self.config.conf_threshold,
                self.config.iou_threshold,
                None,
                False,
                max_det=5,
                nm=32,
            )
            for index, det in enumerate(pred):
                im0 = im0s.copy() if not isinstance(im0s, list) else im0s[index].copy()
                if len(det) == 0:
                    continue
                masks = self.process_mask(proto[index], det[:, 6:], det[:, :4], im_tensor.shape[2:], upsample=True)
                det[:, :4] = self.scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                masks_hwc = masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                masks_hwc = self.scale_image(im_tensor.shape[2:], masks_hwc, im0.shape)
                if masks_hwc.ndim == 2:
                    masks_hwc = masks_hwc[:, :, None]
                masks_nhw = np.moveaxis(masks_hwc.astype(bool), -1, 0)
                for mask, row in zip(masks_nhw, det.cpu().numpy()):
                    conf = float(row[4])
                    cls_id = int(row[5])
                    class_name = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
                    detections.append(
                        mask_to_detection(
                            mask,
                            image_size=(im0.shape[1], im0.shape[0]),
                            confidence=conf,
                            source="yolov5-seg",
                            class_name=class_name,
                        )
                    )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _load_runtime(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for the YOLOv5 backend.") from exc

        root = self.config.yolo_root.resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _prepare_runtime_cache()

        from models.common import DetectMultiBackend
        from utils.dataloaders import LoadImages
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.segment.general import process_mask, scale_image
        from utils.torch_utils import select_device

        self.torch = torch
        self.LoadImages = LoadImages
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.process_mask = process_mask
        self.scale_image = scale_image
        self.device = select_device(self.config.device)
        original_torch_load = torch.load

        def torch_load_checkpoint_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_checkpoint_compat
        try:
            self.model = DetectMultiBackend(
                self.config.weights,
                device=self.device,
                data=self.config.data,
                fp16=False,
            )
        finally:
            torch.load = original_torch_load
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.image_size = check_img_size(self.config.image_size, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.image_size))


def _prepare_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "lens_locator_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
