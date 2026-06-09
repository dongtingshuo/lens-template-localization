# Architecture

本文档说明镜片模板识别与焦度地形图测量工程的运行时结构。项目以稳定 API、可配置算法和可验证输出为核心，覆盖从镜片定位到屈光度地形图的完整软件流程。

This document describes the runtime architecture of the lens-template recognition and refractive power topography project. The design centers on stable APIs, configurable algorithms, and verifiable outputs from localization through refractive power mapping.

## Pipeline

```text
input image
  -> lens localization
  -> optional overlay
  -> Hartmann spot detection
  -> reference/measured spot pairing
  -> displacement-to-angle conversion
  -> refractive matrix fitting
  -> power topography JSON and overlay
```

## Modules

| Module | Responsibility |
| --- | --- |
| `lens_locator.pipeline` | Selects localization backend and returns `LensLocalizationResult`. |
| `lens_locator.classical` | Provides the default lens detector based on image contrast and connected components. |
| `lens_locator.yolo` | Adapts a YOLOv5 segmentation runtime to the project output contract. |
| `lens_locator.topography` | Detects Hartmann spots, pairs reference/measured points, and fits refractive power maps. |
| `lens_locator.geometry` | Converts masks and polygons into center, bounding box, radius, angle, and contour. |
| `lens_locator.result` | Defines immutable dataclasses for localization results. |
| `lens_locator.visualize` | Draws lens overlays and power-map overlays. |
| `lens_locator.demo` | Generates deterministic synthetic inputs for demos and tests. |
| `lens_locator.config` | Loads YAML and builds runtime configuration objects. |

## Data Flow

1. `LensLocator.locate()` receives an image path and returns one or more `LensDetection` objects.
2. `RefractiveTopographyEstimator.measure()` detects bright spot centroids in the measured image.
3. If a reference image is provided, its spot centroids are used as the reference field.
4. If no reference image is provided, a regular reference grid is generated from the lens geometry and configured pitch.
5. Matched spot displacement is converted to angular deflection using `pixel_size_mm / sensor_focal_length_mm`.
6. A 2x2 refractive power matrix is fitted by least squares.
7. Eigen decomposition produces principal powers, sphere equivalent, cylinder, and axis.

## Backend Contract

Every localization backend exposes:

```python
predict(image_path) -> list[LensDetection]
```

The topography estimator consumes either a `LensDetection` ROI or the full image field. This separation keeps recognition and optical measurement independently testable while allowing the CLI to run them as one workflow.

## Configuration Layers

- `backend`, `yolo`, and `classical` configure lens localization.
- `topography` configures Hartmann spot extraction and refractive calibration.
- CLI flags select output paths, overlay generation, reference images, and backend behavior.

## Error Handling

`backend=auto` records YOLO initialization or inference failures and continues with the classical backend. Strict deployments can set `backend=yolo` to fail fast when the neural backend is unavailable. Topography estimation raises a clear error when too few matched spots are available for a stable fit.
