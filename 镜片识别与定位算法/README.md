# 镜片识别与定位 / Lens Recognition and Localization

本模块保存 YOLOv5 segmentation 训练资产、样例数据和实验输出；工程化推理入口已上移到根目录 `lens_locator/` 包。

This module stores YOLOv5 segmentation training assets, sample data, and experiment outputs. The production inference entry point is now the root-level `lens_locator/` package.

## 参考链接 / References

- https://github.com/ultralytics/ultralytics

## 当前状态 / Status

- [x] 识别镜片算法 / Lens recognition algorithm
- [x] 中心点定位 / Center localization
- [x] 统一 CLI 与 JSON 输出 / Unified CLI and JSON output
- [x] ONNX 导出脚本 / ONNX export script
- [ ] RKNN 量化与板端验证 / RKNN quantization and edge-device validation

## 快速运行 / Quick Run

```bash
conda run -n pytorch python -m lens_locator.cli \
  "镜片识别与定位算法/02.代码与实验/yolov5/data/fan/images/train/1 (1).png" \
  --backend classical \
  --overlay-dir outputs/overlays \
  --pretty
```
