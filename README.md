# Lens Template Recognition and Refractive Power Topography

镜片模板识别与焦度地形图测量工程，提供镜片区域识别、中心定位、Hartmann 点阵位移测量、屈光度矩阵拟合和焦度地形图输出能力。项目默认使用纯 Python 运行链路，可在没有 GPU 和训练权重的环境中完成端到端演示、测试和集成。

This project implements lens-template recognition, center localization, Hartmann spot-displacement measurement, refractive power matrix fitting, and power-topography reporting. The default pipeline is fully runnable with the included Python implementation, so it can be tested and integrated without a GPU or trained model weights.

## Capabilities

- 镜片模板识别与定位：输出镜片外接框、中心点、半径/直径、方向角、轮廓和置信度。
- 焦度地形图测量：基于参考/测量点阵的光斑位移，拟合局部焦度、球镜等效、柱镜和轴位。
- 双后端识别架构：内置传统视觉后端可直接运行；YOLOv5 segmentation 后端可作为高精度生产后端接入。
- 工程化接口：提供 Python API、`lens-locate` / `lens-topography` CLI、YAML 配置、JSON 输出和 overlay 可视化。
- 可验证交付：包含合成 demo 生成脚本和自动化测试，覆盖定位、几何转换、地形图拟合与 CLI 输出。

- Lens template recognition and localization: returns bounding box, center, radius/diameter, orientation, contour, and confidence.
- Refractive power topography: estimates local power, sphere equivalent, cylinder, and axis from reference/measured spot displacement.
- Dual recognition backend: the built-in classical backend runs out of the box, while the YOLOv5 segmentation adapter can be enabled for production accuracy.
- Engineering interface: Python API, `lens-locate` / `lens-topography` CLI, YAML configuration, JSON output, and overlay visualization.
- Verifiable delivery: synthetic demo generation and tests cover localization, geometry conversion, topography fitting, and CLI output.

## Repository Layout

```text
.
├── configs/
│   └── lens_locator.yaml          # runtime configuration
├── docs/
│   ├── architecture.md            # pipeline and module design
│   └── model-backends.md          # classical and neural backend configuration
├── lens_locator/
│   ├── classical.py               # dependency-light lens detector
│   ├── cli.py                     # command-line entry point
│   ├── config.py                  # YAML loader
│   ├── demo.py                    # synthetic demo image generation
│   ├── geometry.py                # geometry conversion utilities
│   ├── pipeline.py                # localization backend orchestration
│   ├── result.py                  # structured localization result models
│   ├── topography.py              # refractive power topography estimator
│   ├── visualize.py               # localization and topography overlays
│   └── yolo.py                    # optional YOLOv5 segmentation adapter
├── scripts/
│   ├── export_yolov5_onnx.sh
│   └── generate_demo_inputs.py
├── tests/
│   ├── test_classical.py
│   ├── test_geometry.py
│   └── test_topography.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

The YOLO backend is optional:

```bash
python -m pip install -e ".[yolo]"
```

## Demo Workflow

Generate synthetic lens and Hartmann spot-field images:

```bash
python scripts/generate_demo_inputs.py --output-dir outputs/demo
```

Run lens localization:

```bash
python -m lens_locator.cli outputs/demo/lens.png \
  --backend classical \
  --overlay-dir outputs/overlays \
  --json outputs/lens-result.json \
  --pretty
```

Run refractive power topography:

```bash
python -m lens_locator.cli outputs/demo/hartmann-measured.png \
  --backend classical \
  --measure-topography \
  --reference outputs/demo/hartmann-reference.png \
  --topography-overlay-dir outputs/topography-overlays \
  --json outputs/topography-result.json \
  --pretty
```

## Python API

```python
from lens_locator import LensLocator, RefractiveTopographyEstimator

locator = LensLocator()
localization = locator.locate("outputs/demo/lens.png")

topography = RefractiveTopographyEstimator().measure(
    "outputs/demo/hartmann-measured.png",
    reference_path="outputs/demo/hartmann-reference.png",
)

print(localization.best.center_xy)
print(topography.sphere_equivalent_d, topography.cylinder_d, topography.axis_deg)
```

## Output Contract

Localization result:

| Field | Meaning |
| --- | --- |
| `bbox_xyxy` | Lens bounding box in pixel coordinates. |
| `center_xy` | Lens center in pixel coordinates. |
| `radius_px` / `diameter_px` | Estimated lens size in pixels. |
| `area_px` | Pixel area of the detected mask or component. |
| `confidence` | Backend confidence score. |
| `angle_deg` | Principal-axis orientation in degrees. |
| `source` | Backend that produced the detection. |
| `contour` | Display-ready contour approximation. |

Topography result:

| Field | Meaning |
| --- | --- |
| `sphere_equivalent_d` | Mean spherical equivalent in diopters. |
| `cylinder_d` | Principal-power difference in diopters. |
| `axis_deg` | Cylinder axis in degrees. |
| `principal_powers_d` | Minimum and maximum principal powers. |
| `mean_power_d` / `min_power_d` / `max_power_d` | Local power statistics. |
| `rms_fit_error_d` | Residual fitting error in diopters. |
| `samples` | Per-spot image position, reference position, displacement, and local power. |

## Configuration

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
topography:
  pixel_size_mm: 0.05
  sensor_focal_length_mm: 25.0
  lens_diameter_mm: 70.0
  spot_pitch_px: 32.0
  spot_threshold_percentile: 92.0
  min_spot_area_px: 4
  max_spot_area_px: 800
  max_pair_distance_px: 48.0
```

`backend: auto` tries the YOLO adapter first and falls back to the classical backend. `backend: classical` gives a fully local deterministic pipeline.

## Algorithm Overview

1. Lens localization normalizes the image and detects the dominant lens-like region.
2. Hartmann spot detection extracts bright connected components inside the measurement field.
3. Reference and measured spots are paired by nearest-neighbor matching.
4. Spot displacement is converted into angular deflection using pixel size and sensor focal length.
5. A 2D refractive power matrix is fitted by least squares.
6. Principal powers are decomposed into sphere equivalent, cylinder, and axis.

## Tests

```bash
pytest -q
```

Expected result:

```text
5 passed
```

## Repository Metadata

About:

```text
镜片模板识别与焦度地形图测量工程 / Lens template recognition and refractive power topography toolkit.
```

Topics:

```text
computer-vision, lens-detection, lens-localization, refractive-topography, optical-metrology, machine-vision, python
```

## License

This project is released under `GPL-3.0-or-later`.
