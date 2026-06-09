#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python "镜片识别与定位算法/model_development/yolov5_segmentation/export.py" \
  --weights "镜片识别与定位算法/model_development/yolov5_segmentation/runs/train-seg/exp4/weights/best.pt" \
  --include onnx \
  --imgsz 640 \
  --opset 12
