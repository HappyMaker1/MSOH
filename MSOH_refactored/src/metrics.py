import numpy as np
import pandas as pd
from config import CORE_SIGNALS, ALL_SIGNALS
from .common import is_prediction_probe, similarity_label


def core3_error(rows: pd.DataFrame) -> float:
    vals = []
    for s in CORE_SIGNALS:
        p = rows[(rows["Signal"] == s) & rows["y_true"].notna() & rows["y_pred"].notna()]
        if len(p) < 2:
            continue
        y = p["y_true"].to_numpy(float)
        pred = p["y_pred"].to_numpy(float)
        sd = float(np.std(y, ddof=1))
        if np.isclose(sd, 0):
            continue
        vals.append(float(np.sqrt(np.mean((pred - y) ** 2)) / sd))
    return float(np.mean(vals)) if vals else np.nan


def model_metrics(pred):
    rows = []
    for (source, window, model, signal), p in pred.groupby(
        ["SourceMethod", "Window", "Model", "Signal"], sort=True
    ):
        v = p[p["y_true"].notna() & p["y_pred"].notna()]
        n = len(v)
        if n:
            e = (v["y_pred"] - v["y_true"]).to_numpy(float)
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e**2)))
            sd = float(np.std(v["y_true"].to_numpy(float), ddof=1)) if n > 1 else np.nan
            nrmse = rmse / sd if not pd.isna(sd) and not np.isclose(sd, 0) else np.nan
        else:
            mae = rmse = sd = nrmse = np.nan
        rows.append(
            {
                "SourceMethod": source,
                "Window": int(window),
                "Model": model,
                "Signal": signal,
                "N": n,
                "MAE": mae,
                "RMSE": rmse,
                "True_SD": sd,
                "NRMSE_SD": nrmse,
            }
        )
    return pd.DataFrame(rows)


def overall_metrics(metrics):
    rows = []
    for (source, window, model), p in metrics.groupby(
        ["SourceMethod", "Window", "Model"], sort=True
    ):
        core = p[p["Signal"].isin(CORE_SIGNALS)]
        all4 = p[p["Signal"].isin(ALL_SIGNALS)]
        rows.append(
            {
                "SourceMethod": source,
                "Window": int(window),
                "Model": model,
                "Core3_NRMSE_Mean": float(core["NRMSE_SD"].mean()),
                "All4_NRMSE_Mean": float(all4["NRMSE_SD"].mean()),
                "Core3_Signals_N": int(core["NRMSE_SD"].notna().sum()),
                "All4_Signals_N": int(all4["NRMSE_SD"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def group_level_errors(pred):
    rows = []
    for (source, g, window, model), p in pred.groupby(
        ["SourceMethod", "Group", "Window", "Model"], sort=True
    ):
        for scope, signals in [("Core3", CORE_SIGNALS), ("All4", ALL_SIGNALS)]:
            v = p[p["Signal"].isin(signals) & p["y_true"].notna() & p["y_pred"].notna()]
            if len(v):
                e = (v["y_pred"] - v["y_true"]).to_numpy(float)
                mae = float(np.mean(np.abs(e)))
                rmse = float(np.sqrt(np.mean(e**2)))
            else:
                mae = rmse = np.nan
            rows.append(
                {
                    "SourceMethod": source,
                    "Group": g,
                    "Window": int(window),
                    "Model": model,
                    "Scope": scope,
                    "N": len(v),
                    "MAE": mae,
                    "RMSE": rmse,
                }
            )
    return pd.DataFrame(rows)


def window_error_gain(overall):
    rows = []
    sim = overall[overall["Model"].map(is_prediction_probe)]
    for (source, model), p in sim.groupby(["SourceMethod", "Model"], sort=True):
        prev_core = prev_all = None
        for _, r in p.sort_values("Window").iterrows():
            core, all4 = r["Core3_NRMSE_Mean"], r["All4_NRMSE_Mean"]
            gc = prev_core - core if prev_core is not None else np.nan
            ga = prev_all - all4 if prev_all is not None else np.nan
            rows.append(
                {
                    "SourceMethod": source,
                    "Model": model,
                    "SimilarityLabel": similarity_label(model),
                    "Window": int(r["Window"]),
                    "Error_Core3": core,
                    "Gain_Core3": gc,
                    "RelativeGain_Core3": gc / prev_core if prev_core not in (None, 0) else np.nan,
                    "Error_All4": all4,
                    "Gain_All4": ga,
                    "RelativeGain_All4": ga / prev_all if prev_all not in (None, 0) else np.nan,
                }
            )
            prev_core, prev_all = core, all4
    return pd.DataFrame(rows)


def sign_with_zero(x: float, tol: float = 1e-12) -> int:
    """Exact compatibility with the original 04 trend definition."""
    if pd.isna(x):
        return 99
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def trend_tables(pred):
    """Reproduce the original 04 trend calculation exactly.

    The key compatibility detail is sign_with_zero(tol=1e-12), not np.sign().
    Tiny numerical slopes are therefore classified as flat (0), as in the
    original script.
    """
    rows = []
    for keys, p in pred.groupby(
        ["SourceMethod", "Group", "Window", "Model", "Signal"], sort=True
    ):
        source, g, window, model, signal = keys
        v = p[p["y_true"].notna() & p["y_pred"].notna()].sort_values("TargetDay")
        if len(v) >= 2:
            x = v["TargetDay"].to_numpy(float)
            ts = float(np.polyfit(x, v["y_true"].to_numpy(float), 1)[0])
            ps = float(np.polyfit(x, v["y_pred"].to_numpy(float), 1)[0])
            correct = int(sign_with_zero(ts) == sign_with_zero(ps))
        else:
            ts = ps = correct = np.nan
        rows.append(
            {
                "SourceMethod": source,
                "Group": g,
                "Window": int(window),
                "Model": model,
                "Signal": signal,
                "TrueSlope": ts,
                "PredSlope": ps,
                "TrendCorrect": correct,
            }
        )

    by_group = pd.DataFrame(rows)
    acc = (
        by_group.groupby(["SourceMethod", "Window", "Model", "Signal"], as_index=False)
        .agg(
            N=("TrendCorrect", lambda x: int(x.notna().sum())),
            TrendAccuracy=("TrendCorrect", "mean"),
        )
    )
    core = (
        acc[acc["Signal"].isin(CORE_SIGNALS)]
        .groupby(["SourceMethod", "Window", "Model"], as_index=False)
        .agg(
            Core3_TrendAccuracy=("TrendAccuracy", "mean"),
            Core3_TrendSignalsN=("TrendAccuracy", lambda x: int(x.notna().sum())),
        )
    )
    return by_group, acc, core


def similarity_comparison(overall, core_trend):
    a = overall[overall["Model"].map(is_prediction_probe)].copy()
    b = core_trend[core_trend["Model"].map(is_prediction_probe)]
    m = a.merge(
        b[["SourceMethod", "Window", "Model", "Core3_TrendAccuracy"]],
        on=["SourceMethod", "Window", "Model"],
        how="left",
    )
    m["SimilarityLabel"] = m["Model"].map(similarity_label)
    return m.sort_values(["Window", "Core3_NRMSE_Mean"])
