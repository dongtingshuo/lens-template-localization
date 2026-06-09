# Architecture / 架构设计

本文档说明镜片模板识别与焦度地形图测量工程的运行时结构。项目以 GPU + YOLOv5 segmentation 作为默认模板识别后端，以轻量 Python/classical 后端作为备选，并在同一套结果结构上完成焦度地形图测量。

This document describes the runtime architecture of the lens-template recognition and refractive power topography project. The default recognition backend is GPU-backed YOLOv5 segmentation, while the lightweight Python/classical backend is available as a fallback; both feed the same topography measurement workflow.

## Pipeline / 流程

```text
input image / 输入图像
  -> lens localization / 镜片定位
  -> optional overlay / 可选叠加图
  -> Hartmann spot detection / Hartmann 光斑检测
  -> reference/measured pairing / 参考与测量光斑配对
  -> displacement-to-angle conversion / 位移到角偏转转换
  -> refractive matrix fitting / 屈光度矩阵拟合
  -> power topography JSON and overlay / 焦度地形图 JSON 与叠加图
```

## Modules / 模块

| 模块 / Module | 职责 / Responsibility |
| --- | --- |
| `lens_locator.pipeline` | 选择定位后端并返回 `LensLocalizationResult`。 / Selects the localization backend and returns `LensLocalizationResult`. |
| `lens_locator.yolo` | 默认 YOLOv5 segmentation 后端适配器。 / Default YOLOv5 segmentation backend adapter. |
| `lens_locator.classical` | 轻量 Python 备选检测器，基于对比度和连通域。 / Lightweight Python fallback detector based on contrast and connected components. |
| `lens_locator.topography` | 检测 Hartmann 光斑、配对参考/测量点并拟合焦度地形图。 / Detects Hartmann spots, pairs reference/measured points, and fits refractive power maps. |
| `lens_locator.geometry` | 将 mask 或 polygon 转换为中心、外接框、半径、角度和轮廓。 / Converts masks or polygons into center, bounding box, radius, angle, and contour. |
| `lens_locator.result` | 定义定位结果数据结构。 / Defines localization result dataclasses. |
| `lens_locator.visualize` | 绘制定位叠加图和焦度地形图叠加图。 / Draws localization overlays and power-map overlays. |
| `lens_locator.demo` | 生成稳定的合成 demo 输入。 / Generates deterministic synthetic demo inputs. |
| `lens_locator.config` | 加载 YAML 并构造运行时配置对象。 / Loads YAML and builds runtime configuration objects. |

## Data Flow / 数据流

1. `LensLocator.locate()` 接收图像路径并返回一个或多个 `LensDetection`。  
   `LensLocator.locate()` receives an image path and returns one or more `LensDetection` objects.
2. 默认配置使用 YOLOv5/GPU 后端；备选配置可使用 `classical` 后端。  
   The default configuration uses the YOLOv5/GPU backend; fallback configuration can use the `classical` backend.
3. `RefractiveTopographyEstimator.measure()` 在测量图中检测亮斑质心。  
   `RefractiveTopographyEstimator.measure()` detects bright spot centroids in the measured image.
4. 提供参考图时，参考图光斑作为参考场；否则可由镜片几何和 pitch 生成规则参考网格。  
   When a reference image is provided, its spots define the reference field; otherwise a regular reference grid can be generated from lens geometry and pitch.
5. 配对光斑位移通过 `pixel_size_mm / sensor_focal_length_mm` 转换为角偏转。  
   Matched spot displacement is converted to angular deflection using `pixel_size_mm / sensor_focal_length_mm`.
6. 通过最小二乘拟合 2x2 屈光度矩阵。  
   A 2x2 refractive power matrix is fitted by least squares.
7. 特征值分解得到主焦度、球镜等效、柱镜和轴位。  
   Eigen decomposition produces principal powers, sphere equivalent, cylinder, and axis.

## Backend Contract / 后端接口约定

每个定位后端都暴露同一接口：

Every localization backend exposes the same interface:

```python
predict(image_path) -> list[LensDetection]
```

焦度地形图估计器可以使用 `LensDetection` 作为 ROI，也可以在完整图像范围内工作。定位与光学测量保持解耦，CLI 则把它们组合成一体化流程。

The topography estimator can consume a `LensDetection` ROI or operate on the full image field. Localization and optical measurement remain independently testable, while the CLI composes them into an integrated workflow.

## Configuration Layers / 配置层

- `backend`、`yolo` 和 `classical` 配置镜片定位。  
  `backend`, `yolo`, and `classical` configure lens localization.
- `topography` 配置 Hartmann 光斑提取和屈光度标定。  
  `topography` configures Hartmann spot extraction and refractive calibration.
- CLI 参数选择输出路径、叠加图、参考图和后端行为。  
  CLI flags select output paths, overlays, reference images, and backend behavior.

## Error Handling / 错误处理

`backend=yolo` 是默认生产模式，如果权重、CUDA 或 YOLO 运行环境不可用会直接报错，便于尽早发现部署问题。`backend=classical` 是轻量 Python 备选模式。`backend=auto` 会先尝试 YOLO，失败后回退到 classical，并在 JSON 的 `error` 字段保留原因。

`backend=yolo` is the default production mode and fails fast when weights, CUDA, or the YOLO runtime are unavailable. `backend=classical` is the lightweight Python fallback mode. `backend=auto` tries YOLO first, falls back to classical on failure, and preserves the reason in the JSON `error` field.
