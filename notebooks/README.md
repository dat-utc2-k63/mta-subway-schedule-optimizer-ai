# Notebooks MTA schedule optimization

## Cấu trúc

```
notebooks/
  mta_schedule_optimization.ipynb           # 29 tuyến
  mta_schedule_optimization_single_route.ipynb
  lib/single_route_pipeline.py
  outputs/                                  # kết quả (tùy OUT_DIR trong Setup)
```

## Setup — 3 đường dẫn

Trong notebook, sửa:

```python
DATA_DIR = Path("../datasets")
SCHEDULE_DIR = Path("../datasets/schedule_current")
OUT_DIR = Path("outputs/default")
```

Ví dụ Kaggle:

```python
DATA_DIR = Path("/kaggle/input/datasets/tnguynthnh142/mta-subway-ny")
SCHEDULE_DIR = Path("/kaggle/input/datasets/tnguynthnh142/schedule-subway")
OUT_DIR = Path("/kaggle/working")
```

**Thử nghiệm mô hình:** `RUN_EXPERIMENT` = `default` | `model_mlp` | `model_lstm` (đổi cả `OUT_DIR` nếu muốn tách kết quả, vd. `outputs/model_mlp`).
