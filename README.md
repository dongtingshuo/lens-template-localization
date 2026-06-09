# 镜片模板识别与定位 / Lens Template Recognition and Localization

镜片模板识别与定位算法工程，提供镜片区域分割、中心点定位、结果结构化输出和可视化叠加能力。当前仓库只发布镜片识别与定位相关代码、配置、模型开发资产和测试，不包含其他未完成模块或参考文档。

A production-oriented lens template recognition and localization project. It provides lens-area segmentation, center localization, structured outputs, and overlay visualization. This repository publishes only the lens recognition and localization algorithm, configuration, model-development assets, and tests.

## 工程能力 / Capabilities

- 镜片分割识别：集成已训练的 YOLOv5 segmentation 权重，输出镜片区域 mask、外接框、中心点和置信度。
- 中心定位兜底：提供 `Pillow + numpy` 传统视觉后端，便于在轻量环境中完成基础定位。
- 统一调用接口：提供 `lens_locator` Python 包、`lens-locate` CLI、YAML 配置、JSON 结果和 overlay 输出。
- 可验证交付：包含最小样例数据、单元测试和模型导出脚本，便于复现与集成。

- Lens segmentation: integrates trained YOLOv5 segmentation weights and returns mask, bounding box, center point, and confidence.
- Localization fallback: includes a `Pillow + numpy` classical-vision backend for lightweight environments.
- Unified interface: ships the `lens_locator` Python package, `lens-locate` CLI, YAML config, JSON result, and overlay output.
- Verifiable delivery: includes sample data, unit tests, and an export script for reproducible integration.

## 目录结构 / Repository Layout

```text
.
├── configs/
│   └── lens_locator.yaml                 # 默认推理配置 / default inference config
├── lens_locator/                         # 工程化识别与定位包 / production Python package
├── scripts/
│   └── export_yolov5_onnx.sh             # ONNX 导出脚本 / ONNX export script
├── tests/                                # 单元测试 / unit tests
└── 镜片识别与定位算法/
    ├── README.md                         # 算法资产说明 / algorithm asset notes
    └── model_development/
        └── yolov5_segmentation/          # YOLOv5 分割模型工作区 / YOLOv5 segmentation workspace
```

`model_development/yolov5_segmentation` 保存训练入口、样例数据、权重和导出工具；正式推理入口在根目录的 `lens_locator/` 包中。

`model_development/yolov5_segmentation` stores training entry points, sample data, weights, and export utilities. The production inference entry point is the root-level `lens_locator/` package.

## 环境 / Environment

推荐使用本机已有的 `pytorch` Conda 环境：

Use the existing `pytorch` Conda environment:

```bash
conda activate pytorch
python -c "import torch, cv2, numpy, PIL, yaml, pytest; print('ok')"
```

## 快速推理 / Quick Inference

自动后端会优先使用 YOLOv5 segmentation；如果深度学习后端不可用，会切换到传统视觉后端。

The automatic backend tries YOLOv5 segmentation first and falls back to the classical-vision backend when needed.

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/model_development/yolov5_segmentation/data/fan/images/train/1 (1).png" \
  --config configs/lens_locator.yaml \
  --overlay-dir outputs/overlays \
  --json outputs/result.json \
  --pretty
```

仅运行传统视觉后端：

Run only the classical-vision backend:

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/model_development/yolov5_segmentation/data/fan/images/train/1 (1).png" \
  --backend classical \
  --overlay-dir outputs/overlays \
  --pretty
```

核心输出字段包括 `bbox_xyxy`、`center_xy`、`radius_px`、`diameter_px`、`confidence`、`angle_deg` 和 `source`。

Key output fields include `bbox_xyxy`, `center_xy`, `radius_px`, `diameter_px`, `confidence`, `angle_deg`, and `source`.

## 算法流程 / Algorithm Flow

1. 图像读取与归一化：加载输入图像并统一为后端可处理的像素格式。
2. 镜片区域检测：YOLOv5 segmentation 生成实例 mask；传统视觉后端提取显著连通区域。
3. 几何量计算：根据 mask 计算外接框、中心点、半径、直径和方向角。
4. 结果封装：统一返回 `LensDetection`，方便上层系统读取 JSON 或调用 Python API。

1. Image loading and normalization: converts input images into backend-ready pixel data.
2. Lens-region detection: YOLOv5 segmentation generates instance masks, while the classical backend extracts the dominant connected component.
3. Geometry estimation: computes bounding box, center point, radius, diameter, and orientation angle from the mask.
4. Result packaging: returns `LensDetection` for JSON output or Python API integration.

## 模型开发与导出 / Model Development and Export

YOLOv5 segmentation 工作区：

YOLOv5 segmentation workspace:

```text
镜片识别与定位算法/model_development/yolov5_segmentation/
```

ONNX 导出：

Export ONNX:

```bash
conda run -n pytorch ./scripts/export_yolov5_onnx.sh
```

## 测试 / Tests

```bash
conda run -n pytorch pytest -q
```

## 仓库元信息 / Repository Metadata

描述 / Description:

```text
镜片模板识别与中心定位算法工程 / Lens template recognition and center localization toolkit.
```

Topics:

```text
computer-vision, lens-detection, lens-localization, yolo, yolov5, segmentation, optical-metrology, python
```

## 许可证 / License

本仓库包含 YOLOv5 派生代码，因此采用 `GPL-3.0-or-later`。详见 [LICENSE](LICENSE)。

This repository includes YOLOv5-derived code and is released under `GPL-3.0-or-later`. See [LICENSE](LICENSE).
