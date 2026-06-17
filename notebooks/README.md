# Notebooks MTA schedule optimization

## Kaggle (mặc định trong notebook)

| Biến | Path |
|------|------|
| `LIB_DIR` | `/kaggle/input/datasets/tnguynthnh142/libmta` → `single_route_pipeline.py` |
| `DATA_DIR` | `/kaggle/input/datasets/tnguynthnh142/mta-subway-ny` |
| `SCHEDULE_DIR` | `/kaggle/input/datasets/tnguynthnh142/schedule-subway` |
| `OUT_DIR` | `/kaggle/working/<RUN_TAG>/` |

## Local (repo)

```python
LIB_DIR = Path("lib")  # hoặc notebooks/lib khi chạy từ notebooks/
DATA_DIR = Path("../datasets")
SCHEDULE_DIR = Path("../datasets/schedule_current")
OUT_DIR = Path("outputs") / RUN_TAG
sys.path.insert(0, str(Path(".").resolve()))
import lib.single_route_pipeline as srp
```

## Demand model

Notebook chính và single-route dùng **MLP + HistGBM blend** (`DEMAND_MODEL_TYPE='blend'`).

Metrics: MAE, RMSE, R², MAPE, SMAPE → `OUT_DIR/nn_eval_summary.csv`.
