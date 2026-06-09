# 镜片模板识别与定位 / Lens Template Recognition and Localization

面向镜片模板检测设备的计算机视觉项目，聚焦镜片区域识别、中心定位、分割推理、结果可视化和工程化调用接口。

A computer-vision project for lens template inspection devices, focused on lens region recognition, center localization, segmentation inference, result visualization, and production-style interfaces.

## 项目亮点 / Highlights

- 深度学习识别：基于本仓库已有 YOLOv5 segmentation 训练资产，识别镜片区域并输出分割结果。
- 轻量传统视觉兜底：无 GPU 或无 PyTorch 时，可用 `Pillow + numpy` 完成镜片轮廓定位。
- 工程化接口：提供 `lens_locator` Python 包、`lens-locate` CLI、JSON 输出和叠加图输出。
- 双语文档：GitHub README、算法说明和训练说明均采用中英双语表达。

- Deep-learning recognition: uses the existing YOLOv5 segmentation assets in this repository to detect and segment the lens area.
- Lightweight fallback: when GPU or PyTorch is unavailable, `Pillow + numpy` can still localize the lens contour.
- Production-style interface: ships a `lens_locator` Python package, `lens-locate` CLI, JSON output, and overlay visualization.
- Bilingual documentation: README, algorithm notes, and training notes are written in Chinese and English.

## 目录结构 / Repository Layout

```text
.
├── lens_locator/                         # 工程化识别与定位代码 / production pipeline
├── configs/lens_locator.yaml             # 默认推理配置 / default inference config
├── tests/                                # 单元测试 / unit tests
└── 镜片识别与定位算法/02.代码与实验/yolov5/ # YOLOv5 训练与推理工作区 / YOLOv5 workspace
```

## 环境选择 / Environment

本机已探测到 `conda` 环境，其中 `pytorch` 最适合当前项目：

The local Conda environment named `pytorch` is the best fit for this project:

```bash
conda activate pytorch
python -c "import torch, cv2, numpy, PIL, yaml, pytest; print('ok')"
```

该环境已经包含 `torch / torchvision / opencv / numpy / Pillow / PyYAML / pytest`，可直接运行 YOLOv5 分割推理、传统视觉兜底和测试。

It already includes `torch / torchvision / opencv / numpy / Pillow / PyYAML / pytest`, so it can run YOLOv5 segmentation inference, the classical fallback, and tests.

## 快速推理 / Quick Inference

使用自动后端：优先 YOLOv5，失败时自动切到传统视觉。

Use the automatic backend: YOLOv5 first, then classical fallback if needed.

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/02.代码与实验/yolov5/data/fan/images/train/1 (1).png" \
  --config configs/lens_locator.yaml \
  --overlay-dir outputs/overlays \
  --json outputs/result.json \
  --pretty
```

只运行轻量传统视觉：

Run only the lightweight classical backend:

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/02.代码与实验/yolov5/data/fan/images/train/1 (1).png" \
  --backend classical \
  --overlay-dir outputs/overlays \
  --pretty
```

输出字段包括 `bbox_xyxy`、`center_xy`、`radius_px`、`diameter_px`、`confidence`、`angle_deg` 和 `source`。

The output includes `bbox_xyxy`, `center_xy`, `radius_px`, `diameter_px`, `confidence`, `angle_deg`, and `source`.

## 算法路线 / Algorithm

1. YOLOv5 分割后端：加载 `best.pt`，对镜片区域做实例分割，输出中心、半径、外接框和置信度。
2. 传统视觉后端：估计背景亮度，提取与背景差异最大的连通区域，并根据椭圆外接框计算中心与半径。
3. 统一结果模型：所有后端均返回 `LensDetection`，便于后续标定、机械定位和设备控制模块复用。

1. YOLOv5 segmentation backend: loads `best.pt`, segments the lens area, and returns center, radius, bounding box, and confidence.
2. Classical vision backend: estimates background brightness, extracts the strongest connected component, and computes center/radius from an ellipse-like bounding box.
3. Unified result model: all backends return `LensDetection`, making calibration, mechanical positioning, and device-control modules easier to integrate.

## 训练与导出 / Training and Export

YOLOv5 分割训练入口保留在 `镜片识别与定位算法/02.代码与实验/yolov5/segment/train.py`，当前仓库已包含可用于演示推理的 `best.pt` 权重和小规模样例数据。

The YOLOv5 segmentation training entry point is `镜片识别与定位算法/02.代码与实验/yolov5/segment/train.py`. This repository includes `best.pt` and a small sample dataset for reproducible inference demos.

ONNX 导出：

Export ONNX:

```bash
conda run -n pytorch ./scripts/export_yolov5_onnx.sh
```

RKNN 部署建议基于导出的 ONNX，在 RKNN-Toolkit2 环境中完成量化和转换。

For RKNN deployment, convert and quantize the exported ONNX model in an RKNN-Toolkit2 environment.

## 测试 / Tests

```bash
conda run -n pytorch pytest -q
```

## GitHub About / GitHub About

建议仓库描述：

Suggested repository description:

```text
镜片模板识别与中心定位算法工程 / Lens template recognition and center localization toolkit.
```

建议 Topics：

Suggested topics:

```text
computer-vision, lens-detection, lens-localization, yolo, yolov5, segmentation, optical-metrology, python
```

## License

This repository includes YOLOv5-derived code and is released under GPL-3.0-or-later. See [LICENSE](LICENSE).
