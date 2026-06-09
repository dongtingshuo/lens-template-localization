# Model Artifacts

训练完成的镜片分割模型属于部署工件，不直接提交到 Git。这样可以避免把大文件、原始数据、第三方训练目录和参考资料混在公开工程结构里。

The trained lens segmentation model is a deployment artifact and is not committed to Git. This keeps large files, raw data, third-party training workspaces, and reference material out of the public source tree.

## Expected Local Layout

```text
artifacts/
├── datasets/
│   └── lens.yaml
└── weights/
    └── lens-yolov5-seg.pt

third_party/
└── yolov5/
    ├── export.py
    ├── models/
    └── utils/
```

`configs/lens_locator.yaml` uses this layout by default. You can point to different paths by editing the YAML file or by constructing `YoloConfig` in Python.

## Minimal Dataset YAML

```yaml
path: /absolute/path/to/lens-dataset
train: images/train
val: images/val
names:
  0: glass
```

The runtime adapter only needs the class metadata from this file when loading YOLOv5. Training images and labels should stay outside the Git repository.

## Exporting ONNX

```bash
YOLO_ROOT=third_party/yolov5 \
WEIGHTS=artifacts/weights/lens-yolov5-seg.pt \
IMAGE_SIZE=640 \
OPSET=12 \
./scripts/export_yolov5_onnx.sh
```

The exported `.onnx` file is ignored by Git. Store deployable model binaries in your release system, model registry, or private artifact storage.

## Publication Policy

Do not commit:

- `*.pt`, `*.onnx`, `*.engine`, `*.rknn`, or `*.tflite` model files;
- raw images, labels, cache files, or training runs;
- local YOLOv5 clones or copied training workspaces;
- research PDFs, internal meeting notes, hardware datasheets, or other reference documents.

Commit:

- source code under `lens_locator/`;
- stable configuration templates under `configs/`;
- formal Markdown documentation under `docs/`;
- tests under `tests/`;
- small scripts that help reproduce export or integration workflows.
