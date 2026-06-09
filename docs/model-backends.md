# Model Backends / 模型后端

本工程提供两类镜片模板识别后端：默认的 GPU + YOLOv5 segmentation 后端，以及备选的轻量 Python/classical 后端。两个后端都归一化为 `LensDetection`，因此上层焦度地形图测量流程无需关心识别来源。

This project provides two lens-template recognition backends: the default GPU-backed YOLOv5 segmentation backend and the lightweight Python/classical fallback backend. Both normalize their outputs into `LensDetection`, so the topography pipeline does not need to know which backend produced the localization result.

## Default YOLO/GPU Backend / 默认 YOLO/GPU 后端

YOLOv5 segmentation 后端适合复杂背景、弱边界和批量生产场景。默认配置使用 `backend: yolo` 和 `device: "0"`，表示优先使用第一块 GPU。

The YOLOv5 segmentation backend is intended for complex backgrounds, weak boundaries, and production batches. The default configuration uses `backend: yolo` and `device: "0"`, which targets the first GPU.

```yaml
backend: yolo
yolo:
  weights: artifacts/weights/lens-yolov5-seg.pt
  yolo_root: third_party/yolov5
  data: artifacts/datasets/lens.yaml
  image_size: [640, 640]
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: "0"
```

## Fallback Python Backend / 备选 Python 后端

轻量 Python/classical 后端依赖 `Pillow` 和 `numpy`，适用于快速演示、CI 测试、无 GPU 环境和基础集成验证。它通过背景估计、对比度阈值、连通区域筛选和几何拟合输出镜片中心与尺寸。

The lightweight Python/classical backend depends on `Pillow` and `numpy`. It is suitable for quick demos, CI tests, GPU-free environments, and basic integration checks. It estimates background intensity, thresholds contrast, filters connected components, and converts the dominant component into lens geometry.

```bash
python -m lens_locator.cli path/to/image.png --backend classical --pretty
```

## Automatic Mode / 自动模式

`backend: auto` 会先尝试 YOLOv5 segmentation。如果模型运行环境不可用，会回退到 classical 后端，并在 JSON 结果的 `error` 字段中保留回退原因。该模式适合开发调试，不建议作为严格生产默认值。

`backend: auto` tries YOLOv5 segmentation first. If the model runtime is unavailable, it falls back to the classical backend and preserves the fallback reason in the JSON `error` field. This mode is useful during development, but it is not recommended as the strict production default.

## Training and Export / 训练与导出

GPU 训练用于生成生产权重；运行时工程只要求推理权重和 YOLOv5 推理工作区。ONNX 导出脚本支持通过环境变量指定路径。

GPU training is used to produce production weights; the runtime project only requires inference weights and a YOLOv5 inference workspace. The ONNX export script accepts paths through environment variables.

```bash
YOLO_ROOT=third_party/yolov5 \
WEIGHTS=artifacts/weights/lens-yolov5-seg.pt \
IMAGE_SIZE=640 \
OPSET=12 \
./scripts/export_yolov5_onnx.sh
```
