"""
11_status_nonlinear_probe.py

Nonlinear longitudinal diagnostic for group-mean Status_Score.

Purpose
-------
The current Status_Score is a group-level mean score and is therefore analyzed
as a continuous outcome. This script tests whether the poor/unstable Status
prediction is caused by an overly rigid time trajectory.

Models
------
1) Spline Mixed df=3
2) Spline Mixed df=4

For each outer leave-one-group-out (LOOG) fold:
- fit a mixed-effects model on all observations from the training groups;
- fixed effects = cubic B-spline basis of Day;
- random effects = intercept + linear Day;
- use only Day 1..k from the held-out group to compute its BLUP;
- predict held-out Status_Score on Day 7..9.

This is a diagnostic probe only. It does not overwrite the existing pipeline.

Outputs
-------
results/11_status_nonlinear_probe/table/
    status_spline_predictions.csv
    status_spline_metrics.csv
    status_spline_gain_bootstrap.csv
    status_spline_fit_diagnostics.csv
    status_spline_model_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


DEFAULT_INPUT = Path("results/01_output/animal_data_clean.csv")
DEFAULT_OUTPUT = Path("results/11_status_nonlinear_probe")
WINDOWS = [2, 3, 4, 5, 6]
TARGET_DAYS = [7, 8, 9]
BOOTSTRAP = 2000
SEED = 20260924


def parse_args():
    p = argparse.ArgumentParser(
        description="11: nonlinear spline mixed-effects diagnostic for Status_Score"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def spline_basis(days, df_spline: int):
    # Explicit bounds guarantee the same basis for early and future days.
    mat = dmatrix(
        f"bs(Day, df={df_spline}, degree=3, include_intercept=False, "
        "lower_bound=1, upper_bound=9)",
        {"Day": np.asarray(days, float)},
        return_type="dataframe",
    )
    # patsy includes an intercept column by default; remove it and add our own.
    arr = np.asarray(mat, float)
    names = list(mat.design_info.column_names)
    keep = [i for i, n in enumerate(names) if n != "Intercept"]
    arr = arr[:, keep]
    return np.column_stack([np.ones(len(arr)), arr])


def random_design(days):
    d = np.asarray(days, float)
    # centered day helps numerical stability
    return np.column_stack([np.ones(len(d)), d - 4.5])


def fit_training_mixed(train_df: pd.DataFrame, df_spline: int):
    y = train_df["Status_Score"].to_numpy(float)
    X = spline_basis(train_df["Day"].to_numpy(float), df_spline)
    Z = random_design(train_df["Day"].to_numpy(float))
    groups = train_df["Group"].astype(str).to_numpy()

    model = sm.MixedLM(
        endog=y,
        exog=X,
        groups=groups,
        exog_re=Z,
    )

    # ML is preferable for out-of-sample prediction comparison.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=1000,
            disp=False,
        )

    beta = np.asarray(fit.fe_params, float)
    G = np.asarray(fit.cov_re, float)
    sigma2 = float(fit.scale)

    return fit, beta, G, sigma2


def heldout_blup(early_df, beta, G, sigma2, df_spline):
    y = early_df["Status_Score"].to_numpy(float)
    days = early_df["Day"].to_numpy(float)
    X = spline_basis(days, df_spline)
    Z = random_design(days)

    resid = y - X @ beta
    V = Z @ G @ Z.T + sigma2 * np.eye(len(y))

    # b_hat = G Z' V^-1 (y - X beta)
    try:
        sol = np.linalg.solve(V, resid)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(V) @ resid

    return G @ Z.T @ sol


def predict_future(beta, b_hat, df_spline):
    days = np.asarray(TARGET_DAYS, float)
    Xf = spline_basis(days, df_spline)
    Zf = random_design(days)
    return Xf @ beta + Zf @ b_hat


def nrmse(part):
    yt = part["y_true"].to_numpy(float)
    yp = part["y_pred"].to_numpy(float)
    if len(yt) < 2:
        return np.nan
    sd = np.std(yt, ddof=1)
    if np.isclose(sd, 0.0):
        return np.nan
    return float(np.sqrt(np.mean((yp - yt) ** 2)) / sd)


def run_probe(df, df_spline):
    groups = sorted(df["Group"].astype(str).unique().tolist())
    preds = []
    diags = []
    label = f"Status Spline Mixed df{df_spline}"
    model_name = f"Probe_Status_SplineMixed_df{df_spline}"

    for window in WINDOWS:
        for test_group in groups:
            train = df[df["Group"].astype(str) != str(test_group)].copy()
            train = train[["Group", "Day", "Status_Score"]].dropna()

            test_all = df[df["Group"].astype(str) == str(test_group)].copy()
            early = test_all[
                (test_all["Day"] <= window)
                & test_all["Status_Score"].notna()
            ][["Day", "Status_Score"]].sort_values("Day")

            if early.empty:
                continue

            fallback = ""
            converged = False
            try:
                fit, beta, G, sigma2 = fit_training_mixed(train, df_spline)
                converged = bool(getattr(fit, "converged", False))
                b_hat = heldout_blup(early, beta, G, sigma2, df_spline)
                yp = predict_future(beta, b_hat, df_spline)
            except Exception as e:
                # Transparent fallback: training mean by target day.
                fallback = repr(e)
                yp = np.asarray([
                    train.loc[train["Day"] == d, "Status_Score"].mean()
                    for d in TARGET_DAYS
                ], float)
                beta = np.full(1, np.nan)
                G = np.full((2, 2), np.nan)
                sigma2 = np.nan

            for day, pred in zip(TARGET_DAYS, yp):
                truth_row = test_all[test_all["Day"] == day]
                if truth_row.empty or pd.isna(truth_row.iloc[0]["Status_Score"]):
                    continue
                yt = float(truth_row.iloc[0]["Status_Score"])
                preds.append({
                    "Group": str(test_group),
                    "Window": int(window),
                    "Model": model_name,
                    "SimilarityLabel": label,
                    "Signal": "Status_Score",
                    "TargetDay": int(day),
                    "y_true": yt,
                    "y_pred": float(pred),
                    "error": float(pred - yt),
                })

            diags.append({
                "Group": str(test_group),
                "Window": int(window),
                "Model": model_name,
                "Spline_df": int(df_spline),
                "N_Early": int(len(early)),
                "Converged": converged,
                "Residual_Variance": sigma2,
                "RE_Intercept_Var": (
                    float(G[0, 0]) if np.ndim(G) == 2 and G.shape[0] >= 2 else np.nan
                ),
                "RE_Slope_Var": (
                    float(G[1, 1]) if np.ndim(G) == 2 and G.shape[0] >= 2 else np.nan
                ),
                "Fallback_Reason": fallback,
            })

    return pd.DataFrame(preds), pd.DataFrame(diags)


def summarize(pred):
    rows = []
    for (label, model, window), part in pred.groupby(
        ["SimilarityLabel", "Model", "Window"], sort=True
    ):
        yt = part["y_true"].to_numpy(float)
        yp = part["y_pred"].to_numpy(float)
        rows.append({
            "SimilarityLabel": label,
            "Model": model,
            "Window": int(window),
            "NRMSE": nrmse(part),
            "RMSE": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "MAE": float(np.mean(np.abs(yp - yt))),
            "N": int(len(part)),
        })
    return pd.DataFrame(rows)


def bootstrap_gain(pred, B, seed):
    rng = np.random.default_rng(seed)
    rows = []

    for (label, model), mdf in pred.groupby(["SimilarityLabel", "Model"], sort=True):
        groups = sorted(mdf["Group"].unique().tolist())
        windows = sorted(mdf["Window"].unique().tolist())

        point = {
            w: nrmse(mdf[mdf["Window"] == w])
            for w in windows
        }

        boots = np.full((B, len(windows)), np.nan, float)

        for b in range(B):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            for wi, w in enumerate(windows):
                parts = []
                for rep, g in enumerate(sampled):
                    p = mdf[(mdf["Group"] == g) & (mdf["Window"] == w)].copy()
                    if p.empty:
                        continue
                    p["_bootstrap_copy"] = rep
                    parts.append(p)
                if parts:
                    boots[b, wi] = nrmse(pd.concat(parts, ignore_index=True))

        for wi in range(1, len(windows)):
            fw = windows[wi - 1]
            tw = windows[wi]
            vals = boots[:, wi - 1] - boots[:, wi]
            vals = vals[np.isfinite(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "SimilarityLabel": label,
                "Model": model,
                "FromWindow": int(fw),
                "ToWindow": int(tw),
                "Gain_Point": float(point[fw] - point[tw]),
                "Gain_CI_Low": float(lo),
                "Gain_CI_High": float(hi),
                "P_Gain_GT_0": float(np.mean(vals > 0)),
                "P_Gain_GT_0.03": float(np.mean(vals > 0.03)),
                "Bootstrap_B": int(B),
            })

    return pd.DataFrame(rows)


def build_comparison(metrics):
    old_path = Path(
        "results/08_signal_diagnostics/table/signal_error_by_window.csv"
    )
    frames = []

    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old["Signal"] == "Status_Score"][
            ["SimilarityLabel", "Window", "NRMSE"]
        ].copy()
        frames.append(old)

    frames.append(metrics[["SimilarityLabel", "Window", "NRMSE"]].copy())

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Window", "NRMSE"])
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入数据：{args.input}")

    df = pd.read_csv(args.input)
    required = ["Group", "Day", "Status_Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("输入数据缺少字段：" + ", ".join(missing))

    pred_list = []
    diag_list = []

    print("=" * 110)
    print("11 Status_Score 非线性 spline mixed-effects 诊断")
    print("固定测试两个预设复杂度：df=3, df=4；不做事后选优。")
    print("LOOG；held-out group 仅使用 Day1..k 估计随机效应；预测 Day7-Day9。")
    print("=" * 110)

    for df_spline in [3, 4]:
        print(f"\n运行 Spline Mixed df={df_spline} ...")
        p, d = run_probe(df, df_spline)
        pred_list.append(p)
        diag_list.append(d)

    pred = pd.concat(pred_list, ignore_index=True)
    diag = pd.concat(diag_list, ignore_index=True)
    metrics = summarize(pred)
    gain = bootstrap_gain(pred, args.bootstrap, args.seed)
    comparison = build_comparison(metrics)

    table_dir = args.output_dir / "table"
    table_dir.mkdir(parents=True, exist_ok=True)

    pred.to_csv(table_dir / "status_spline_predictions.csv", index=False)
    diag.to_csv(table_dir / "status_spline_fit_diagnostics.csv", index=False)
    metrics.to_csv(table_dir / "status_spline_metrics.csv", index=False)
    gain.to_csv(table_dir / "status_spline_gain_bootstrap.csv", index=False)
    comparison.to_csv(table_dir / "status_spline_model_comparison.csv", index=False)

    print("\nSpline model metrics:")
    print(metrics.to_string(index=False))

    print("\nGain bootstrap:")
    print(gain.to_string(index=False))

    print("\nFit diagnostics summary:")
    print(
        diag.groupby(["Model"], as_index=False)
        .agg(
            N_Fits=("Group", "size"),
            N_Converged=("Converged", "sum"),
            N_Fallback=("Fallback_Reason", lambda x: int((x.astype(str) != "").sum())),
        )
        .to_string(index=False)
    )

    print("\nStatus_Score model comparison (lower NRMSE = better):")
    print(comparison.to_string(index=False))

    print("\n输出：")
    for f in [
        "status_spline_predictions.csv",
        "status_spline_fit_diagnostics.csv",
        "status_spline_metrics.csv",
        "status_spline_gain_bootstrap.csv",
        "status_spline_model_comparison.csv",
    ]:
        print(f"  {table_dir / f}")

    print("=" * 110)


if __name__ == "__main__":
    main()
