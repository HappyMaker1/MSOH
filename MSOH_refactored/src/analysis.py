from pathlib import Path
import pandas as pd
from .metrics import (
    model_metrics,
    overall_metrics,
    group_level_errors,
    window_error_gain,
    trend_tables,
    similarity_comparison,
)


def load_all_predictions(input_dir: Path):
    files = sorted(input_dir.glob("predictions_loog_*.csv"))
    if not files:
        legacy = input_dir / "predictions_loog.csv"
        if legacy.exists():
            files = [legacy]
        else:
            raise FileNotFoundError(f"在 {input_dir} 中没有找到 predictions_loog_*.csv")
    frames = []
    required = {"Group", "Window", "Model", "Signal", "TargetDay", "y_true", "y_pred"}
    for path in files:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")
        prefix = "predictions_loog_"
        df["SourceMethod"] = path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem
        df["SourceFile"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def run_analysis(pred):
    metrics = model_metrics(pred)
    overall = overall_metrics(metrics)
    groups = group_level_errors(pred)
    gain = window_error_gain(overall)
    tbg, ta, core = trend_tables(pred)
    comp = similarity_comparison(overall, core)
    msoh = gain.copy()
    msoh["ErrorCriterion"] = pd.NA
    msoh["GainCriterion"] = pd.NA
    msoh["MSOH_Candidate"] = pd.NA
    return {
        "combined_predictions": pred,
        "model_metrics": metrics,
        "overall_model_metrics": overall,
        "group_level_errors": groups,
        "window_error_gain": gain,
        "trend_by_group": tbg,
        "trend_accuracy": ta,
        "core3_trend_summary": core,
        # Historical filename retained for compatibility.  It now contains all
        # MSOH prediction probes, not only trajectory-similarity methods.
        "similarity_method_comparison": comp,
        "msoh_analysis": msoh,
    }
