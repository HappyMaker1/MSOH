import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import CLEAN_PATH, FEATURE_PATH, PREDICTION_DIR
from src.common import PROBE_METHODS, save_csv
from src.probes import run_ridge_loog, run_mixed_effects_loog


def basic_metrics(pred):
    rows = []
    for (w, m, s), p in pred.groupby(["Window", "Model", "Signal"]):
        v = p[p["y_true"].notna() & p["y_pred"].notna()]
        if v.empty:
            continue
        e = (v["y_pred"] - v["y_true"]).to_numpy(float)
        rows.append(
            {
                "Window": int(w),
                "Model": m,
                "Signal": s,
                "N": int(len(v)),
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e**2))),
            }
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="03b: alternative prediction probes for MSOH robustness"
    )
    p.add_argument("--data", type=Path, default=CLEAN_PATH)
    p.add_argument("--features", type=Path, default=FEATURE_PATH)
    p.add_argument("--output-dir", type=Path, default=PREDICTION_DIR)
    p.add_argument("--method", choices=PROBE_METHODS, required=True)
    a = p.parse_args()

    data = pd.read_csv(a.data)
    data["Group"] = data["Group"].astype(str)
    feat = pd.read_csv(a.features)
    feat["Group"] = feat["Group"].astype(str)

    if a.method == "ridge":
        pred, diag = run_ridge_loog(data, feat)
    else:
        pred, diag = run_mixed_effects_loog(data)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(pred, a.output_dir / f"predictions_loog_{a.method}.csv")
    save_csv(diag, a.output_dir / f"probe_diagnostics_{a.method}.csv")
    save_csv(basic_metrics(pred), a.output_dir / f"basic_metrics_{a.method}.csv")
    print(f"03 probe 完成：{a.method}")


if __name__ == "__main__":
    main()
