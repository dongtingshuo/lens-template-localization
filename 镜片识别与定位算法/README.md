# 镜片识别与定位算法资产 / Lens Recognition and Localization Assets

本目录只保留镜片识别与定位算法相关资产。工程化推理入口位于根目录 `lens_locator/` 包，本目录用于保存模型开发工作区、样例数据、训练入口和模型权重。

This directory contains only assets related to the lens recognition and localization algorithm. The production inference entry point is the root-level `lens_locator/` package; this directory stores the model-development workspace, sample data, training entry points, and model weights.

## 结构 / Structure

```text
镜片识别与定位算法/
├── README.md
└── model_development/
    └── yolov5_segmentation/
        ├── data/fan.yaml                 # 样例数据配置 / sample dataset config
        ├── data/fan/                     # 最小样例数据 / minimal sample data
        ├── runs/train-seg/exp4/weights/  # 已训练权重 / trained weights
        ├── segment/train.py              # 分割训练入口 / segmentation training entry
        └── export.py                     # 模型导出入口 / model export entry
```

## 使用方式 / Usage

从仓库根目录运行：

Run from the repository root:

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/model_development/yolov5_segmentation/data/fan/images/train/1 (1).png" \
  --config configs/lens_locator.yaml \
  --overlay-dir outputs/overlays \
  --pretty
```

## 交付状态 / Delivery Status

- [x] 镜片区域分割识别 / Lens-region segmentation
- [x] 镜片中心点定位 / Lens center localization
- [x] CLI、YAML 配置和 JSON 输出 / CLI, YAML config, and JSON output
- [x] Overlay 可视化输出 / Overlay visualization output
- [x] ONNX 导出脚本 / ONNX export script

## 第三方组件 / Third-Party Component

`model_development/yolov5_segmentation` 基于 YOLOv5 segmentation 工作流整理，并随仓库许可证保持兼容。

`model_development/yolov5_segmentation` is organized around a YOLOv5 segmentation workflow and remains compatible with the repository license.
