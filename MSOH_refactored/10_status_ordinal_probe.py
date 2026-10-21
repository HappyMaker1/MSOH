"""
10_status_ordinal_probe.py

Diagnostic ordinal-response probe for Status_Score.

Purpose
-------
Status_Score is an ordered animal-condition score, so treating it as a fully
continuous outcome may be inappropriate. This script adds a proportional-odds
(ordinal logistic) prediction probe while preserving the existing LOOG design.

Design
------
- Candidate observation windows: Day 1..k, k in {2,3,4,5,6}
- Fixed targets: Day 7..9
- Outer validation: leave-one-group-out (LOOG)
- Early predictors (Core3 only):
    Weight_last, Weight_slope,
    Status_Score_last, Status_Score_slope,
    Feeding_last, Feeding_slope,
    TargetGap = TargetDay - Window
- Train-fold-only standardization
- Penalized proportional-odds model with fixed L2 penalty on beta.
- The model predicts:
    * class probabilities for each ordered Status_Score level
    * expected ordinal score (used as y_pred for RMSE/NRMSE comparability)
    * modal predicted class
- No post-hoc model selection is performed.

Important
---------
This is a diagnostic probe, not a replacement for the existing pipeline yet.
NRMSE is retained only for comparability with the current MSOH framework.
Ordinal-specific metrics (exact accuracy, within-1 accuracy, ordinal log loss,
quadratic weighted kappa) are also reported.

Dependencies
------------
numpy, pandas, scipy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score


DEFAULT_INPUT = Path("results/animal_data_clean.csv")
DEFAULT_OUTPUT = Path("results/10_status_ordinal_probe")
DEFAULT_WINDOWS = [2, 3, 4, 5, 6]
DEFAULT_TARGET_DAYS = [7, 8, 9]
DEFAULT_ALPHA = 1.0
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260924

CORE3 = ["Weight", "Status_Score", "Feeding"]
MODEL_NAME = "Probe_Status_Ordinal_PO"


def parse_args():
    p = argparse.ArgumentParser(
        description="10: Status_Score proportional-odds ordinal prediction probe"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Fixed L2 penalty on ordinal-regression beta; default 1.0")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def slope(days: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(days) & np.isfinite(values)
    days = days[mask]
    values = values[mask]
    if len(values) < 2 or np.allclose(days, days[0]):
        return 0.0
    return float(np.polyfit(days, values, 1)[0])


def group_features(df: pd.DataFrame, group: str, window: int) -> dict:
    part = df[(df["Group"].astype(str) == str(group)) & (df["Day"] <= window)].copy()
    part = part.sort_values("Day")
    if part.empty:
        raise ValueError(f"No early observations for Group={group}, window={window}")

    out = {}
    for s in CORE3:
        sub = part[["Day", s]].dropna()
        if sub.empty:
            out[f"{s}_last"] = np.nan
            out[f"{s}_slope"] = np.nan
        else:
            out[f"{s}_last"] = float(sub.iloc[-1][s])
            out[f"{s}_slope"] = slope(
                sub["Day"].to_numpy(float),
                sub[s].to_numpy(float),
            )
    return out


def target_value(df: pd.DataFrame, group: str, day: int):
    part = df[
        (df["Group"].astype(str) == str(group))
        & (df["Day"] == day)
    ]
    if part.empty:
        return np.nan
    return part.iloc[0]["Status_Score"]


def softplus(x):
    # numerically stable softplus
    return np.logaddexp(0.0, x)


def inv_softplus(y):
    y = np.maximum(np.asarray(y, float), 1e-6)
    return np.log(np.expm1(y))


class PenalizedProportionalOdds:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.beta_ = None
        self.thresholds_ = None
        self.categories_ = None
        self.success_ = False
        self.message_ = ""

    def _unpack(self, params, p, k):
        beta = params[:p]
        raw = params[p:]
        theta = np.empty(k - 1, dtype=float)
        theta[0] = raw[0]
        for j in range(1, k - 1):
            theta[j] = theta[j - 1] + softplus(raw[j]) + 1e-4
        return beta, theta

    def _nll(self, params, X, y_idx, k):
        p = X.shape[1]
        beta, theta = self._unpack(params, p, k)
        eta = X @ beta

        cdf = expit(theta[None, :] - eta[:, None])
        probs = np.empty((len(y_idx), k), dtype=float)
        probs[:, 0] = cdf[:, 0]
        for j in range(1, k - 1):
            probs[:, j] = cdf[:, j] - cdf[:, j - 1]
        probs[:, -1] = 1.0 - cdf[:, -1]
        probs = np.clip(probs, 1e-12, 1.0)

        ll = np.log(probs[np.arange(len(y_idx)), y_idx]).sum()
        penalty = 0.5 * self.alpha * float(beta @ beta)
        return float(-ll + penalty)

    def fit(self, X, y, categories):
        X = np.asarray(X, float)
        y = np.asarray(y)
        cats = np.asarray(categories)
        if len(cats) < 2:
            raise ValueError("Ordinal model needs at least two ordered categories.")

        mapping = {v: i for i, v in enumerate(cats.tolist())}
        y_idx = np.asarray([mapping[v] for v in y], dtype=int)
        k = len(cats)
        p = X.shape[1]

        # empirical cumulative logits initialize thresholds
        counts = np.bincount(y_idx, minlength=k).astype(float)
        cum = np.cumsum(counts)[:-1] / max(counts.sum(), 1.0)
        cum = np.clip(cum, 0.05, 0.95)
        theta0 = np.log(cum / (1.0 - cum))
        theta0 = np.maximum.accumulate(theta0)

        raw = np.zeros(k - 1, float)
        raw[0] = theta0[0]
        if k > 2:
            diffs = np.maximum(np.diff(theta0) - 1e-4, 1e-3)
            raw[1:] = inv_softplus(diffs)

        init = np.concatenate([np.zeros(p, float), raw])
        opt = minimize(
            self._nll,
            init,
            args=(X, y_idx, k),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        self.beta_, self.thresholds_ = self._unpack(opt.x, p, k)
        self.categories_ = cats
        self.success_ = bool(opt.success)
        self.message_ = str(opt.message)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        eta = X @ self.beta_
        theta = self.thresholds_
        k = len(self.categories_)

        cdf = expit(theta[None, :] - eta[:, None])
        probs = np.empty((len(X), k), dtype=float)
        probs[:, 0] = cdf[:, 0]
        for j in range(1, k - 1):
            probs[:, j] = cdf[:, j] - cdf[:, j - 1]
        probs[:, -1] = 1.0 - cdf[:, -1]
        probs = np.clip(probs, 0.0, None)
        denom = probs.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return probs / denom


def build_training_rows(df, train_groups, window, target_days):
    rows = []
    feature_names = [
        "Weight_last", "Weight_slope",
        "Status_Score_last", "Status_Score_slope",
        "Feeding_last", "Feeding_slope",
        "TargetGap",
    ]

    for g in train_groups:
        feat = group_features(df, g, window)
        for day in target_days:
            y = target_value(df, g, day)
            if pd.isna(y):
                continue
            row = {**feat, "TargetGap": float(day - window), "y": y, "Group": str(g), "TargetDay": day}
            rows.append(row)

    out = pd.DataFrame(rows)
    return out, feature_names


def impute_scale_train_test(train_X, test_X):
    train_X = np.asarray(train_X, float)
    test_X = np.asarray(test_X, float)

    med = np.nanmedian(train_X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    train_X = np.where(np.isnan(train_X), med, train_X)
    test_X = np.where(np.isnan(test_X), med, test_X)

    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0, ddof=0)
    sd = np.where((~np.isfinite(sd)) | np.isclose(sd, 0.0), 1.0, sd)

    return (train_X - mu) / sd, (test_X - mu) / sd


def run_loog(df, windows, target_days, alpha):
    groups = sorted(df["Group"].astype(str).unique().tolist())
    all_target = df[df["Day"].isin(target_days)]["Status_Score"].dropna()
    categories = np.asarray(sorted(pd.unique(all_target)), dtype=float)

    pred_rows = []
    diag_rows = []

    for window in windows:
        for test_group in groups:
            train_groups = [g for g in groups if g != test_group]
            train_df, feature_names = build_training_rows(df, train_groups, window, target_days)

            # test rows
            test_feat = group_features(df, test_group, window)
            test_rows = []
            for day in target_days:
                y = target_value(df, test_group, day)
                if pd.isna(y):
                    continue
                test_rows.append({
                    **test_feat,
                    "TargetGap": float(day - window),
                    "y": y,
                    "TargetDay": day,
                })
            test_df = pd.DataFrame(test_rows)
            if test_df.empty:
                continue

            Xtr_raw = train_df[feature_names].to_numpy(float)
            Xte_raw = test_df[feature_names].to_numpy(float)
            Xtr, Xte = impute_scale_train_test(Xtr_raw, Xte_raw)

            model = PenalizedProportionalOdds(alpha=alpha)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        Xtr,
                        train_df["y"].to_numpy(),
                        categories=categories,
                    )
                probs = model.predict_proba(Xte)
                success = model.success_
                msg = model.message_
            except Exception as e:
                # transparent fallback: training empirical class distribution
                counts = np.asarray(
                    [(train_df["y"].to_numpy() == c).sum() for c in categories],
                    float,
                )
                probs0 = counts / counts.sum()
                probs = np.tile(probs0, (len(test_df), 1))
                success = False
                msg = f"fallback_empirical: {e}"

            expected = probs @ categories
            modal_idx = np.argmax(probs, axis=1)
            modal = categories[modal_idx]

            for i, row in test_df.reset_index(drop=True).iterrows():
                yt = float(row["y"])
                true_idx = int(np.where(categories == yt)[0][0])
                p_true = float(np.clip(probs[i, true_idx], 1e-12, 1.0))

                pred_rows.append({
                    "Group": str(test_group),
                    "Window": int(window),
                    "Model": MODEL_NAME,
                    "SimilarityLabel": "Status Ordinal PO",
                    "Signal": "Status_Score",
                    "TargetDay": int(row["TargetDay"]),
                    "y_true": yt,
                    "y_pred": float(expected[i]),
                    "y_pred_class": float(modal[i]),
                    "exact_correct": float(modal[i] == yt),
                    "within1_correct": float(abs(modal[i] - yt) <= 1),
                    "ordinal_log_loss": float(-np.log(p_true)),
                    "probabilities_json": json.dumps({
                        str(float(c)): float(p)
                        for c, p in zip(categories, probs[i])
                    }),
                })

            diag_rows.append({
                "Group": str(test_group),
                "Window": int(window),
                "N_Train_Rows": int(len(train_df)),
                "N_Test_Rows": int(len(test_df)),
                "N_Categories": int(len(categories)),
                "Categories": ",".join(map(str, categories.tolist())),
                "Alpha": float(alpha),
                "Optimizer_Success": bool(success),
                "Optimizer_Message": msg,
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(diag_rows)


def nrmse_from_part(part):
    yt = part["y_true"].to_numpy(float)
    yp = part["y_pred"].to_numpy(float)
    if len(yt) < 2:
        return np.nan
    sd = np.std(yt, ddof=1)
    if np.isclose(sd, 0.0):
        return np.nan
    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    return float(rmse / sd)


def summarize_metrics(pred):
    rows = []
    for window, part in pred.groupby("Window", sort=True):
        yt = part["y_true"].to_numpy(float)
        yp = part["y_pred"].to_numpy(float)
        ypc = part["y_pred_class"].to_numpy(float)

        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        mae = float(np.mean(np.abs(yp - yt)))
        nrmse = nrmse_from_part(part)
        exact = float(np.mean(ypc == yt))
        within1 = float(np.mean(np.abs(ypc - yt) <= 1))
        logloss = float(np.mean(part["ordinal_log_loss"]))

        try:
            qwk = float(cohen_kappa_score(yt, ypc, weights="quadratic"))
        except Exception:
            qwk = np.nan

        rows.append({
            "Window": int(window),
            "NRMSE": nrmse,
            "RMSE": rmse,
            "MAE": mae,
            "Exact_Accuracy": exact,
            "Within1_Accuracy": within1,
            "Ordinal_LogLoss": logloss,
            "Quadratic_Weighted_Kappa": qwk,
            "N": int(len(part)),
        })
    return pd.DataFrame(rows)


def bootstrap_nrmse_gain(pred, B, seed):
    groups = sorted(pred["Group"].unique().tolist())
    windows = sorted(pred["Window"].unique().tolist())
    rng = np.random.default_rng(seed)

    point = {
        int(w): nrmse_from_part(pred[pred["Window"] == w])
        for w in windows
    }

    boot = np.full((B, len(windows)), np.nan, float)

    by_group_window = {
        (g, w): pred[(pred["Group"] == g) & (pred["Window"] == w)].copy()
        for g in groups for w in windows
    }

    for b in range(B):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        for wi, w in enumerate(windows):
            parts = []
            for rep_i, g in enumerate(sampled):
                p = by_group_window[(g, w)].copy()
                if p.empty:
                    continue
                # distinguish duplicated bootstrap copies without changing values
                p["_boot_id"] = rep_i
                parts.append(p)
            if not parts:
                continue
            bp = pd.concat(parts, ignore_index=True)
            boot[b, wi] = nrmse_from_part(bp)

    error_rows = []
    for wi, w in enumerate(windows):
        vals = boot[:, wi]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        error_rows.append({
            "Window": int(w),
            "NRMSE_Point": float(point[w]),
            "NRMSE_CI_Low": float(lo),
            "NRMSE_CI_High": float(hi),
            "Bootstrap_B": int(B),
        })

    gain_rows = []
    for wi in range(1, len(windows)):
        fw, tw = windows[wi - 1], windows[wi]
        vals = boot[:, wi - 1] - boot[:, wi]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        gain_rows.append({
            "FromWindow": int(fw),
            "ToWindow": int(tw),
            "Gain_Point": float(point[fw] - point[tw]),
            "Gain_CI_Low": float(lo),
            "Gain_CI_High": float(hi),
            "P_Gain_GT_0": float(np.mean(vals > 0)),
            "P_Gain_GT_0.03": float(np.mean(vals > 0.03)),
            "Bootstrap_B": int(B),
        })

    return pd.DataFrame(error_rows), pd.DataFrame(gain_rows)


def compare_existing(metrics, existing_path):
    if not existing_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(existing_path)
    old = old[old["Signal"] == "Status_Score"].copy()
    keep = ["SimilarityLabel", "Window", "NRMSE"]
    old = old[keep]
    new = metrics[["Window", "NRMSE"]].copy()
    new["SimilarityLabel"] = "Status Ordinal PO"
    new = new[keep]
    return pd.concat([old, new], ignore_index=True).sort_values(["Window", "NRMSE"])


def main():
    args = parse_args()
    out = args.output_dir
    table_dir = out / "table"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入数据：{args.input}")

    df = pd.read_csv(args.input)
    required = ["Group", "Day", "Weight", "Status_Score", "Feeding"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("输入数据缺少字段：" + ", ".join(missing))

    # Score must be genuinely ordered/discrete.
    vals = sorted(pd.unique(df["Status_Score"].dropna()))
    if len(vals) < 2:
        raise ValueError("Status_Score 少于两个等级，无法拟合 ordinal model.")

    print("=" * 110)
    print("10 Status_Score ordinal probe")
    print(f"Score levels: {vals}")
    print(f"Fixed ridge penalty alpha = {args.alpha}")
    print("Outer validation = LOOG; target = Day7-Day9")
    print("=" * 110)

    pred, diag = run_loog(
        df,
        DEFAULT_WINDOWS,
        DEFAULT_TARGET_DAYS,
        alpha=args.alpha,
    )
    metrics = summarize_metrics(pred)
    err_boot, gain_boot = bootstrap_nrmse_gain(
        pred,
        B=args.bootstrap,
        seed=args.seed,
    )

    comparison = compare_existing(
        metrics,
        Path("results/08_signal_diagnostics/table/signal_error_by_window.csv"),
    )

    pred.to_csv(table_dir / "status_ordinal_predictions.csv", index=False)
    diag.to_csv(table_dir / "status_ordinal_fit_diagnostics.csv", index=False)
    metrics.to_csv(table_dir / "status_ordinal_metrics.csv", index=False)
    err_boot.to_csv(table_dir / "status_ordinal_error_bootstrap.csv", index=False)
    gain_boot.to_csv(table_dir / "status_ordinal_gain_bootstrap.csv", index=False)
    comparison.to_csv(table_dir / "status_model_comparison.csv", index=False)

    print("\nOrdinal probe metrics:")
    print(metrics.to_string(index=False))

    print("\nNRMSE bootstrap:")
    print(err_boot.to_string(index=False))

    print("\nNRMSE gain bootstrap:")
    print(gain_boot.to_string(index=False))

    failed = diag[~diag["Optimizer_Success"]]
    print(
        f"\nOptimizer success: {len(diag)-len(failed)}/{len(diag)} outer fits"
    )
    if len(failed):
        print("存在未正常收敛/回退的 fold，请检查 status_ordinal_fit_diagnostics.csv")

    if not comparison.empty:
        print("\nStatus_Score model comparison (lower NRMSE = better):")
        print(comparison.to_string(index=False))

    print("\n输出：")
    for f in [
        "status_ordinal_predictions.csv",
        "status_ordinal_fit_diagnostics.csv",
        "status_ordinal_metrics.csv",
        "status_ordinal_error_bootstrap.csv",
        "status_ordinal_gain_bootstrap.csv",
        "status_model_comparison.csv",
    ]:
        print(f"  {table_dir / f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
