# Lens Template Recognition and Refractive Power Topography / 镜片模板识别与焦度地形图测量

镜片模板识别与焦度地形图测量工程，提供镜片区域识别、中心定位、Hartmann 点阵位移测量、屈光度矩阵拟合和焦度地形图输出能力。默认生产链路采用 GPU + YOLOv5 segmentation 完成镜片模板识别，轻量 Python/classical 链路作为无 GPU、无权重或快速演示时的备选。

This project implements lens-template recognition, center localization, Hartmann spot-displacement measurement, refractive power matrix fitting, and power-topography reporting. The default production workflow uses GPU-backed YOLOv5 segmentation for lens-template recognition, while the lightweight Python/classical workflow is kept as a fallback for GPU-free, weight-free, or demo environments.

## Capabilities / 工程能力

- 镜片模板识别与定位：输出镜片外接框、中心点、半径/直径、方向角、轮廓和置信度。  
  Lens template recognition and localization: returns bounding box, center, radius/diameter, orientation, contour, and confidence.
- 焦度地形图测量：基于参考/测量点阵的光斑位移，拟合局部焦度、球镜等效、柱镜和轴位。  
  Refractive power topography: estimates local power, sphere equivalent, cylinder, and axis from reference/measured spot displacement.
- 默认生产后端：YOLOv5 segmentation 使用 GPU 推理完成复杂背景下的镜片模板识别。  
  Default production backend: YOLOv5 segmentation uses GPU inference for lens-template recognition in complex scenes.
- 备选 Python 链路：`classical` 后端可用于快速演示、无 GPU 环境和基础集成验证。  
  Fallback Python workflow: the `classical` backend supports quick demos, GPU-free environments, and basic integration checks.
- 工程化交付：提供 Python API、`lens-locate` / `lens-topography` CLI、YAML 配置、JSON 输出和 overlay 可视化。  
  Engineering delivery: Python API, `lens-locate` / `lens-topography` CLI, YAML configuration, JSON output, and overlay visualization.

## Default Runtime / 默认运行链路

默认运行链路是 GPU + YOLOv5 segmentation。它更符合项目的正式目标：在复杂背景、真实镜片边界和批量检测场景中获得更稳定的模板识别结果。

The default runtime is GPU-backed YOLOv5 segmentation. This matches the project's production goal: robust template recognition for complex backgrounds, real lens boundaries, and batch inspection.

```yaml
backend: yolo
yolo:
  device: "0"
```

没有 GPU、权重或 YOLO 运行环境时，可以显式切换到 Python 备选链路：

When GPU, model weights, or the YOLO runtime are unavailable, explicitly switch to the Python fallback workflow:

```yaml
backend: classical
```

## Repository Layout / 仓库结构

```text
.
├── configs/
│   └── lens_locator.yaml          # 运行配置 / runtime configuration
├── docs/
│   ├── architecture.md            # 架构说明 / architecture
│   └── model-backends.md          # 后端说明 / backend configuration
├── lens_locator/
│   ├── classical.py               # 轻量镜片检测 / lightweight lens detector
│   ├── cli.py                     # 命令行入口 / CLI entry point
│   ├── config.py                  # 配置加载 / YAML loader
│   ├── demo.py                    # demo 数据生成 / synthetic demo generation
│   ├── geometry.py                # 几何计算 / geometry utilities
│   ├── pipeline.py                # 定位流程 / localization orchestration
│   ├── result.py                  # 结果模型 / result models
│   ├── topography.py              # 焦度地形图 / refractive topography
│   ├── visualize.py               # 可视化 / overlays
│   └── yolo.py                    # 默认 YOLO 后端 / default YOLO backend
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

## Installation / 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

默认 YOLO/GPU 后端依赖：

Default YOLO/GPU backend dependencies:

```bash
python -m pip install -e ".[yolo]"
```

## Demo Workflow / 演示流程

生成合成镜片图与 Hartmann 点阵图：

Generate synthetic lens and Hartmann spot-field images:

```bash
python scripts/generate_demo_inputs.py --output-dir outputs/demo
```

运行镜片定位：

Run lens localization:

```bash
python -m lens_locator.cli outputs/demo/lens.png \
  --backend classical \
  --overlay-dir outputs/overlays \
  --json outputs/lens-result.json \
  --pretty
```

运行焦度地形图测量：

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

## Python API / Python 接口

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

## Output Contract / 输出结构

定位结果 / Localization result:

| 字段 / Field | 含义 / Meaning |
| --- | --- |
| `bbox_xyxy` | 镜片外接框像素坐标 / Lens bounding box in pixel coordinates. |
| `center_xy` | 镜片中心点像素坐标 / Lens center in pixel coordinates. |
| `radius_px` / `diameter_px` | 镜片半径和直径估计 / Estimated lens radius and diameter. |
| `area_px` | 检测区域面积 / Pixel area of the detected component. |
| `confidence` | 后端置信度 / Backend confidence score. |
| `angle_deg` | 主轴方向角 / Principal-axis orientation. |
| `source` | 输出来源后端 / Backend that produced the detection. |
| `contour` | 可视化轮廓 / Display-ready contour approximation. |

地形图结果 / Topography result:

| 字段 / Field | 含义 / Meaning |
| --- | --- |
| `sphere_equivalent_d` | 球镜等效值 / Mean spherical equivalent in diopters. |
| `cylinder_d` | 柱镜值 / Principal-power difference in diopters. |
| `axis_deg` | 柱镜轴位 / Cylinder axis in degrees. |
| `principal_powers_d` | 主焦度 / Minimum and maximum principal powers. |
| `mean_power_d` / `min_power_d` / `max_power_d` | 局部焦度统计 / Local power statistics. |
| `rms_fit_error_d` | 拟合残差 / Residual fitting error in diopters. |
| `samples` | 单点光斑位移与局部焦度 / Per-spot displacement and local power. |

## Configuration / 配置

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

## Algorithm Overview / 算法流程

1. 镜片定位模块读取图像并检测主要镜片区域。  
   The localization module reads the image and detects the dominant lens region.
2. Hartmann 光斑检测模块提取测量区域内的亮斑连通域。  
   The Hartmann spot detector extracts bright connected components in the measurement field.
3. 参考光斑与测量光斑通过最近邻方式配对。  
   Reference and measured spots are paired by nearest-neighbor matching.
4. 光斑位移通过像元尺寸和传感器焦距转换为角偏转。  
   Spot displacement is converted into angular deflection using pixel size and sensor focal length.
5. 使用最小二乘拟合二维屈光度矩阵。  
   A 2D refractive power matrix is fitted by least squares.
6. 对主焦度做分解，得到球镜等效、柱镜和轴位。  
   Principal powers are decomposed into sphere equivalent, cylinder, and axis.

## Tests / 测试

```bash
pytest -q
```

期望结果 / Expected result:

```text
5 passed
```

## Repository Metadata / 仓库元信息

About / 简介:

```text
镜片模板识别与焦度地形图测量工程 / Lens template recognition and refractive power topography toolkit.
```

Topics / 主题:

```text
computer-vision, lens-detection, lens-localization, refractive-topography, optical-metrology, machine-vision, python
```

## License / 许可证

This project is released under `GPL-3.0-or-later`.

本项目采用 `GPL-3.0-or-later` 许可证发布。
