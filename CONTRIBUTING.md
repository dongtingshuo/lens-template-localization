# Contributing / 贡献指南

Keep algorithm changes reproducible and separate the dependency-free classical
path from optional YOLO integrations. / 算法改动需可复现，并保持无额外依赖的
经典路径与可选 YOLO 集成相互独立。

- Add deterministic tests for geometry or calibration changes. / 几何或标定
  改动需补充确定性测试。
- Document units, coordinate systems, and assumptions. / 明确记录单位、坐标系
  与假设。
- Do not commit datasets, weights, exported models, or private reference
  documents. / 不提交数据集、权重、导出模型或私有参考资料。

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```
