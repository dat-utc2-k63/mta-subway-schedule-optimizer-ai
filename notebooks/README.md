# Notebooks MTA schedule optimization

## Kaggle (mặc định trong notebook)

| Biến | Path |
|------|------|
| `LIB_DIR` | `/kaggle/input/datasets/tnguynthnh142/libmta` → `single_route_pipeline.py` |
| `DATA_DIR` | `/kaggle/input/datasets/tnguynthnh142/mta-subway-ny` |
| `SCHEDULE_DIR` | `/kaggle/input/datasets/tnguynthnh142/schedule-subway` |
| `OUT_DIR` | `/kaggle/working/<RUN_EXPERIMENT>/` |

```python
RUN_EXPERIMENT = "default"  # default | model_mlp | model_lstm
srp.apply_experiment(RUN_EXPERIMENT, globals())
```

## Local (repo)

```python
LIB_DIR = Path("lib")  # hoặc notebooks/lib khi chạy từ notebooks/
DATA_DIR = Path("../datasets")
SCHEDULE_DIR = Path("../datasets/schedule_current")
OUT_DIR = Path("outputs") / RUN_EXPERIMENT
sys.path.insert(0, str(Path(".").resolve()))
import lib.single_route_pipeline as srp
```

## Thí nghiệm

| `RUN_EXPERIMENT` | Mô hình |
|------------------|---------|
| `default` | MLP + HistGBM blend |
| `model_mlp` | MLP |
| `model_lstm` | LSTM |

Metrics: MAE, RMSE, R², MAPE, SMAPE → `OUT_DIR/nn_eval_summary.csv`.
