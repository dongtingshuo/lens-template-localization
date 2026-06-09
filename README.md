# Lens Template Recognition and Localization

镜片模板识别与定位算法工程，面向自动化光学检测、镜片加工定位和屈光度地形图采集前处理场景。项目提供统一的 Python API 与 CLI，可输出镜片区域、中心点、半径/直径、方向角、置信度和可视化叠加结果。

This project packages a lens-template recognition and localization pipeline for automated optical inspection, lens processing alignment, and preprocessing before refractive topography measurement. It exposes a unified Python API and CLI that return lens region geometry, center point, radius/diameter, orientation, confidence, and optional overlay visualization.

## Highlights

- 双后端架构：优先使用训练完成的 YOLOv5 segmentation 模型；在模型或深度学习运行时不可用时自动回退到轻量传统视觉后端。
- 工程化输出：统一返回 `LensLocalizationResult` / `LensDetection`，字段稳定，便于接入上位机、检测服务或标定流程。
- 可配置交付：模型权重、YOLOv5 工作区、类别数据、阈值和图像尺寸均通过 YAML 配置管理。
- 发布边界清晰：仓库只保留运行时源码、配置、脚本、测试和正式文档；参考文献、会议纪要、训练数据、权重和第三方开发工作区不进入版本库。
- Verifiable core: geometry and fallback detection are covered by unit tests, so downstream integration has a deterministic smoke test even without GPU artifacts.

## Repository Layout

```text
.
├── configs/
│   └── lens_locator.yaml        # default runtime configuration
├── docs/
│   ├── architecture.md          # pipeline and module design
│   └── model-artifacts.md       # model, dataset, and vendor workspace policy
├── lens_locator/
│   ├── classical.py             # lightweight fallback detector
│   ├── cli.py                   # command-line entry point
│   ├── config.py                # YAML loader
│   ├── geometry.py              # geometry conversion utilities
│   ├── pipeline.py              # backend orchestration
│   ├── result.py                # structured result models
│   ├── visualize.py             # overlay rendering
│   └── yolo.py                  # optional YOLOv5 segmentation adapter
├── scripts/
│   └── export_yolov5_onnx.sh    # ONNX export wrapper
├── tests/
│   ├── test_classical.py
│   └── test_geometry.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

Local-only paths such as `artifacts/`, `third_party/`, `data/`, `models/`, `docs/reference/`, and the original Chinese development workspace are ignored intentionally.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For the YOLOv5 segmentation backend, install the optional dependencies and prepare the local model artifacts described in [docs/model-artifacts.md](docs/model-artifacts.md).

```bash
python -m pip install -e ".[yolo]"
```

## Quick Start

Run the automatic pipeline. If the configured YOLO model is available, the neural backend is used; otherwise the classical backend runs as a deterministic fallback.

```bash
python -m lens_locator.cli path/to/lens-image.jpg \
  --config configs/lens_locator.yaml \
  --overlay-dir outputs/overlays \
  --json outputs/result.json \
  --pretty
```

Run the dependency-light backend only:

```bash
python -m lens_locator.cli path/to/lens-image.jpg \
  --backend classical \
  --overlay-dir outputs/overlays \
  --pretty
```

Use the package directly:

```python
from lens_locator import LensLocator

locator = LensLocator()
result = locator.locate("path/to/lens-image.jpg")
best = result.best

if best is not None:
    print(best.center_xy, best.diameter_px, best.confidence)
```

## Output Contract

Each detection is serialized with stable fields:

| Field | Meaning |
| --- | --- |
| `bbox_xyxy` | Bounding box in pixel coordinates: `x1, y1, x2, y2`. |
| `center_xy` | Lens center in pixel coordinates. |
| `radius_px` / `diameter_px` | Estimated lens size in pixels. |
| `area_px` | Pixel area of the detected mask or component. |
| `confidence` | Backend confidence score. |
| `angle_deg` | Principal-axis orientation in degrees. |
| `class_name` | Detection class, defaulting to `glass`. |
| `source` | Backend that produced the detection, such as `yolov5-seg` or `classical`. |
| `contour` | Display-ready contour approximation. |
| `metadata` | Backend-specific diagnostic values. |

## Configuration

Default runtime configuration:

```yaml
backend: auto
yolo:
  weights: artifacts/weights/lens-yolov5-seg.pt
  yolo_root: third_party/yolov5
  data: artifacts/datasets/lens.yaml
  image_size: [640, 640]
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: ""
classical:
  min_area_ratio: 0.005
  max_area_ratio: 0.85
  threshold_percentile: 85.0
  min_threshold: 6.0
  blur_kernel: 5
```

Set `backend: yolo` when model artifacts are mandatory. Keep `backend: auto` for deployments where a classical fallback is acceptable.

## Model Artifacts

The trained lens segmentation model is treated as a deployment artifact rather than source code. Place local files in this layout:

```text
artifacts/
├── datasets/
│   └── lens.yaml
└── weights/
    └── lens-yolov5-seg.pt

third_party/
└── yolov5/
    ├── export.py
    ├── models/
    └── utils/
```

These paths are ignored by Git to keep the public repository clean and to avoid publishing reference documents, raw datasets, large weights, or copied third-party training workspaces.

## Export

Export the configured PyTorch segmentation weight to ONNX:

```bash
YOLO_ROOT=third_party/yolov5 \
WEIGHTS=artifacts/weights/lens-yolov5-seg.pt \
IMAGE_SIZE=640 \
OPSET=12 \
./scripts/export_yolov5_onnx.sh
```

## Tests

```bash
pytest -q
```

The current tests validate geometry conversion and the classical fallback detector with synthetic input. YOLO inference should be verified in the deployment environment after placing the local artifacts.

## Repository Metadata

About:

```text
镜片模板识别与中心定位算法工程 / Lens template recognition and center localization toolkit.
```

Topics:

```text
computer-vision, lens-detection, lens-localization, image-segmentation, yolo, optical-metrology, python, machine-vision
```

## License

This project is released under `GPL-3.0-or-later`. If a deployment bundles YOLOv5-derived runtime code or weights, keep the corresponding third-party license notices with that deployment package.
