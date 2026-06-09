#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

YOLO_ROOT="${YOLO_ROOT:-third_party/yolov5}"
WEIGHTS="${WEIGHTS:-artifacts/weights/lens-yolov5-seg.pt}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
OPSET="${OPSET:-12}"

python "$YOLO_ROOT/export.py" \
  --weights "$WEIGHTS" \
  --include onnx \
  --imgsz "$IMAGE_SIZE" \
  --opset "$OPSET"
