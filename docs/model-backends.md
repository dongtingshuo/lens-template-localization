# Model Backends

本工程提供两类镜片模板识别后端：默认可运行的传统视觉后端，以及可选的 YOLOv5 segmentation 后端。两个后端都归一化为同一个 `LensDetection` 输出结构，便于上层测量流程复用。

This project provides two lens-template recognition backends: a default runnable classical-vision backend and an optional YOLOv5 segmentation backend. Both normalize their outputs into the same `LensDetection` structure for downstream measurement.

## Classical Backend

传统视觉后端依赖 `Pillow` 和 `numpy`，适用于快速验证、边缘部署和无 GPU 环境。它通过背景估计、对比度阈值、连通区域筛选和几何拟合输出镜片中心与尺寸。

The classical backend depends only on `Pillow` and `numpy`. It is designed for quick validation, edge deployment, and GPU-free environments. It estimates background intensity, thresholds contrast, filters connected components, and converts the dominant component into lens geometry.

```bash
python -m lens_locator.cli path/to/image.png --backend classical --pretty
```

## YOLOv5 Segmentation Backend

YOLOv5 segmentation 后端适合复杂背景、弱边界或批量生产场景。启用该后端时，通过 YAML 指定权重、YOLOv5 工作区、类别数据文件、置信度阈值、IoU 阈值和推理设备。

The YOLOv5 segmentation backend is intended for complex backgrounds, weak boundaries, or production batches. It is configured through YAML with weight path, YOLOv5 workspace, class metadata, confidence threshold, IoU threshold, and device.

```yaml
backend: yolo
yolo:
  weights: artifacts/weights/lens-yolov5-seg.pt
  yolo_root: third_party/yolov5
  data: artifacts/datasets/lens.yaml
  image_size: [640, 640]
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: ""
```

## Automatic Mode

`backend: auto` 会先尝试 YOLOv5 segmentation。如果模型运行环境不可用，会回退到传统视觉后端，并在 JSON 结果的 `error` 字段中保留回退原因。

`backend: auto` tries YOLOv5 segmentation first. If the model runtime is unavailable, it falls back to the classical backend and preserves the fallback reason in the JSON `error` field.

## ONNX Export

```bash
YOLO_ROOT=third_party/yolov5 \
WEIGHTS=artifacts/weights/lens-yolov5-seg.pt \
IMAGE_SIZE=640 \
OPSET=12 \
./scripts/export_yolov5_onnx.sh
```

The export script keeps deployment paths configurable through environment variables, making it suitable for local workstations and CI jobs.
