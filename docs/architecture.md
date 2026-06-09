# Architecture

本文档说明镜片模板识别与定位工程的运行时结构。仓库目标是提供可集成、可测试、可配置的算法包，而不是保存训练草稿、参考文献或原始实验资料。

This document describes the runtime architecture of the lens-template recognition and localization project. The repository is designed as an integration-ready algorithm package, not as a storage location for training drafts, references, or raw experiment material.

## Pipeline

```text
input image
  -> backend selection
  -> lens region detection
  -> geometry estimation
  -> result normalization
  -> JSON and optional overlay
```

1. `LensLocator` receives an image path and a `LensLocatorConfig`.
2. `backend=auto` tries the YOLOv5 segmentation adapter first.
3. If YOLO artifacts or dependencies are unavailable, the classical detector runs as a fallback.
4. Backend-specific detections are normalized into `LensDetection`.
5. The CLI writes JSON and, when requested, an annotated overlay image.

## Modules

| Module | Responsibility |
| --- | --- |
| `lens_locator.pipeline` | Selects backend, measures runtime, and returns `LensLocalizationResult`. |
| `lens_locator.yolo` | Adapts a local YOLOv5 segmentation workspace to the project result contract. |
| `lens_locator.classical` | Provides a dependency-light fallback based on contrast thresholding and connected components. |
| `lens_locator.geometry` | Converts masks and polygons into center, bounding box, radius, angle, and contour. |
| `lens_locator.result` | Defines immutable dataclasses used by the API and CLI serialization. |
| `lens_locator.visualize` | Draws overlay images for inspection and downstream reports. |
| `lens_locator.config` | Loads YAML and converts file paths into runtime configuration objects. |

## Backend Contract

Every detector backend exposes:

```python
predict(image_path) -> list[LensDetection]
```

This keeps the system open to additional backends such as ONNX Runtime, OpenVINO, RKNN, or an industrial camera SDK without changing the CLI or API result shape.

## Error Handling

`backend=auto` records the YOLO failure message in `LensLocalizationResult.error` and continues with the classical backend. `backend=yolo` raises the original exception, which is better for strict production deployments where missing model artifacts should fail fast.

## Deployment Boundary

The repository contains runtime source, tests, configuration, scripts, and formal Markdown documentation. The following materials are intentionally local-only:

- trained weights and exported models;
- raw or annotated datasets;
- copied third-party YOLOv5 workspaces;
- reference papers, meeting minutes, and hardware documents;
- generated overlays, JSON outputs, logs, and cache files.

This boundary keeps the public project professional while still allowing the completed model to be mounted into the expected local artifact paths during deployment.
