"""
07_signal_contribution_bootstrap.py

Enhanced signal contribution analysis for the MSOH framework.

This version extends 07 by adding uncertainty-aware signal attribution:

    C_s(k) = Error_{-s}(k) - Error_all(k)

with group-level paired bootstrap:

    C_s^(b)(k) = Error_{-s}^(b(k) - Error_all^b(k)

using the SAME resampled groups for All4 and each leave-one-signal-out variant.

Outputs include:
- point contribution C_s(k)
- 95% bootstrap CI
- P(C_s(k) > 0)
- P(C_s(k) > delta_C), optional practical contribution threshold
- MSOH shifts Delta_MSOH_s
- baseline validation against existing 03/04 DTW All4 predictions

The DTW implementation matches 03_models_cv:
- local scalar cost = absolute difference
- local vector cost = Euclidean norm
- accumulated path cost / path length
- Defecation multi-point sequence uses unconstrained DTW
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_INPUT = Path("results/01_output/animal_data_clean.csv")
DEFAULT_REFERENCE = Path("results/04_output/table/combined_predictions.csv")
DEFAULT_OUTPUT = Path("results/07_output")

ALL_SIGNALS = ["Weight", "Status_Score", "Feeding", "Defecation"]
CORE3 = ["Weight", "Status_Score", "Feeding"]

DEFAULT_WINDOWS = [2, 3, 4, 5, 6]
DEFAULT_TARGET_DAYS = [7, 8, 9]

DEFAULT_K = 2
DEFAULT_DTW_RADIUS = 1
DEFAULT_DEFECATION_LAMBDA = 1.0

DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260924
DEFAULT_BASELINE_TOL = 1e-6

# Keep aligned with 06
DEFAULT_ERROR_THRESHOLD = 0.40
DEFAULT_GAIN_DELTA = 0.03
DEFAULT_ALPHA = 0.10
DEFAULT_STABILITY_THRESHOLD = 0.75
DEFAULT_GROUP_EPSILON = 0.03

# Exploratory practical signal-contribution threshold
DEFAULT_CONTRIBUTION_DELTA = 0.03

EPS = 1e-12


# ======================================================================
# CLI
# ======================================================================

def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="07 enhanced: uncertainty-aware leave-one-signal-out MSOH contribution"
    )

    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--reference-predictions",
        type=Path,
        default=DEFAULT_REFERENCE,
    )
    p.add_argument(
        "--baseline-tol",
        type=float,
        default=DEFAULT_BASELINE_TOL,
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--windows", type=str, default="2,3,4,5,6")
    p.add_argument("--target-days", type=str, default="7,8,9")

    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--dtw-radius", type=int, default=DEFAULT_DTW_RADIUS)
    p.add_argument(
        "--defecation-lambda",
        type=float,
        default=DEFAULT_DEFECATION_LAMBDA,
    )

    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument(
        "--error-threshold",
        type=float,
        default=DEFAULT_ERROR_THRESHOLD,
    )
    p.add_argument(
        "--gain-delta",
        type=float,
        default=DEFAULT_GAIN_DELTA,
    )
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument(
        "--stability-threshold",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLD,
    )
    p.add_argument(
        "--group-epsilon",
        type=float,
        default=DEFAULT_GROUP_EPSILON,
    )

    p.add_argument(
        "--contribution-delta",
        type=float,
        default=DEFAULT_CONTRIBUTION_DELTA,
        help="Exploratory practical threshold for P(C_s(k) > delta_C).",
    )

    return p.parse_args()


# ======================================================================
# Data
# ======================================================================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Group", "Day", *ALL_SIGNALS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"输入文件缺少字段：{sorted(missing)}")

    df["Group"] = df["Group"].astype(str)
    df["Day"] = pd.to_numeric(df["Day"], errors="raise").astype(int)

    for signal in ALL_SIGNALS:
        df[signal] = pd.to_numeric(df[signal], errors="coerce")

        observed_col = f"{signal}_observed"

        if observed_col not in df.columns:
            df[observed_col] = df[signal].notna()
        elif df[observed_col].dtype == object:
            parsed = (
                df[observed_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
            )
            df[observed_col] = parsed.where(
                parsed.notna(),
                df[signal].notna(),
            )
        else:
            df[observed_col] = df[observed_col].astype(bool)

    if df.duplicated(["Group", "Day"]).any():
        raise ValueError("发现重复 Group-Day 记录。")

    return df.sort_values(["Group", "Day"]).reset_index(drop=True)


# ======================================================================
# Exact DTW logic from 03
# ======================================================================

def dtw_distance(a, b, radius=1):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    n, m = len(a), len(b)

    if n == 0 or m == 0:
        return np.nan

    if radius is None:
        radius = max(n, m)

    radius = max(int(radius), abs(n - m))

    cost = np.full((n + 1, m + 1), np.inf)
    plen = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(
            max(1, i - radius),
            min(m, i + radius) + 1,
        ):
            ai = a[i - 1]
            bj = b[j - 1]

            local = (
                abs(float(ai) - float(bj))
                if np.ndim(ai) == 0
                else np.linalg.norm(ai - bj)
            )

            prev = min(
                [
                    (cost[i - 1, j], plen[i - 1, j]),
                    (cost[i, j - 1], plen[i, j - 1]),
                    (cost[i - 1, j - 1], plen[i - 1, j - 1]),
                ],
                key=lambda x: x[0],
            )

            if np.isinf(prev[0]):
                continue

            cost[i, j] = prev[0] + local
            plen[i, j] = prev[1] + 1

    return (
        float(cost[n, m] / plen[n, m])
        if plen[n, m]
        else np.nan
    )


# ======================================================================
# Fold-specific preprocessing
# ======================================================================

def fit_dense_scaler(
    df: pd.DataFrame,
    train_groups: list[str],
    window: int,
    signals: list[str],
):
    train = df[
        df["Group"].isin(train_groups)
        & (df["Day"] <= window)
    ]

    scaler = {}

    for signal in signals:
        x = train[signal].dropna().to_numpy(dtype=float)

        if len(x) == 0:
            mean, sd = 0.0, 1.0
        else:
            mean = float(np.mean(x))
            sd = float(np.std(x, ddof=0))

            if np.isclose(sd, 0.0):
                sd = 1.0

        scaler[signal] = (mean, sd)

    return scaler


def fit_defecation_scaler(
    df: pd.DataFrame,
    train_groups: list[str],
    window: int,
):
    train = df[
        df["Group"].isin(train_groups)
        & (df["Day"] <= window)
        & df["Defecation_observed"]
        & df["Defecation"].notna()
    ]

    x = train["Defecation"].to_numpy(dtype=float)

    if len(x) == 0:
        return 0.0, 1.0

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))

    if np.isclose(sd, 0.0):
        sd = 1.0

    return mean, sd


def dense_sequence(
    df: pd.DataFrame,
    group: str,
    window: int,
    signals: list[str],
    scaler,
):
    part = (
        df[
            (df["Group"] == group)
            & (df["Day"] <= window)
        ]
        .sort_values("Day")
    )

    rows = []

    for _, row in part.iterrows():
        vals = []

        for signal in signals:
            value = row[signal]

            if pd.isna(value):
                return np.empty((0, len(signals)))

            mean, sd = scaler[signal]
            vals.append((float(value) - mean) / sd)

        rows.append(vals)

    return np.asarray(rows, dtype=float)


def defecation_sequence(
    df: pd.DataFrame,
    group: str,
    window: int,
    mean: float,
    sd: float,
):
    part = (
        df[
            (df["Group"] == group)
            & (df["Day"] <= window)
            & df["Defecation_observed"]
            & df["Defecation"].notna()
        ]
        .sort_values("Day")
    )

    x = part["Defecation"].to_numpy(dtype=float)

    if len(x) == 0:
        return np.asarray([], dtype=float)

    return (x - mean) / sd


def pair_distance(
    df,
    group_a,
    group_b,
    window,
    included_signals,
    train_groups,
    dtw_radius,
    defecation_lambda,
):
    dense_signals = [
        s for s in included_signals
        if s != "Defecation"
    ]

    use_def = "Defecation" in included_signals

    dense_d = None
    def_d = None

    if dense_signals:
        scaler = fit_dense_scaler(
            df,
            train_groups,
            window,
            dense_signals,
        )

        a = dense_sequence(
            df,
            group_a,
            window,
            dense_signals,
            scaler,
        )

        b = dense_sequence(
            df,
            group_b,
            window,
            dense_signals,
            scaler,
        )

        if len(a) and len(b):
            dense_d = dtw_distance(
                a,
                b,
                dtw_radius,
            )

    if use_def:
        mean, sd = fit_defecation_scaler(
            df,
            train_groups,
            window,
        )

        a = defecation_sequence(
            df,
            group_a,
            window,
            mean,
            sd,
        )

        b = defecation_sequence(
            df,
            group_b,
            window,
            mean,
            sd,
        )

        if len(a) and len(b):
            def_d = (
                float(abs(a[0] - b[0]))
                if len(a) == 1 and len(b) == 1
                else dtw_distance(a, b, None)
            )

    if dense_d is not None and def_d is not None:
        return float(
            math.sqrt(
                dense_d ** 2
                + defecation_lambda * def_d ** 2
            )
        )

    if dense_d is not None:
        return float(dense_d)

    if def_d is not None:
        return float(def_d)

    return np.nan


# ======================================================================
# Prediction
# ======================================================================

def get_future_value(df, group, signal, day):
    part = df[
        (df["Group"] == group)
        & (df["Day"] == day)
    ]

    if part.empty:
        return np.nan

    value = part.iloc[0][signal]

    return np.nan if pd.isna(value) else float(value)


def predict_variant(
    df,
    variant,
    included_signals,
    windows,
    target_days,
    k_neighbors,
    dtw_radius,
    defecation_lambda,
):
    groups = sorted(df["Group"].unique().tolist())

    pred_rows = []
    neighbor_rows = []

    for window in windows:
        for heldout in groups:
            train_groups = [g for g in groups if g != heldout]

            distances = []

            for train_group in train_groups:
                d = pair_distance(
                    df=df,
                    group_a=heldout,
                    group_b=train_group,
                    window=window,
                    included_signals=included_signals,
                    train_groups=train_groups,
                    dtw_radius=dtw_radius,
                    defecation_lambda=defecation_lambda,
                )

                if not np.isnan(d):
                    distances.append((train_group, d))

            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k_neighbors]

            for rank, (neighbor, d) in enumerate(neighbors, start=1):
                neighbor_rows.append(
                    {
                        "Variant": variant,
                        "Window": int(window),
                        "HeldOutGroup": heldout,
                        "NeighborRank": int(rank),
                        "NeighborGroup": neighbor,
                        "Distance": float(d),
                    }
                )

            for signal in CORE3:
                for target_day in target_days:
                    y_true = get_future_value(
                        df,
                        heldout,
                        signal,
                        target_day,
                    )

                    num = 0.0
                    den = 0.0
                    n_used = 0

                    for neighbor, d in neighbors:
                        y_neighbor = get_future_value(
                            df,
                            neighbor,
                            signal,
                            target_day,
                        )

                        if np.isnan(y_neighbor):
                            continue

                        w = 1.0 / (d + EPS)
                        num += w * y_neighbor
                        den += w
                        n_used += 1

                    y_pred = num / den if den > 0 else np.nan

                    pred_rows.append(
                        {
                            "Variant": variant,
                            "IncludedSignals": "|".join(included_signals),
                            "Group": heldout,
                            "Window": int(window),
                            "Signal": signal,
                            "TargetDay": int(target_day),
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "N_Neighbors_Used": int(n_used),
                        }
                    )

    return pd.DataFrame(pred_rows), pd.DataFrame(neighbor_rows)


# ======================================================================
# Baseline validation
# ======================================================================

def validate_all4_baseline(
    all4_pred,
    reference_path,
    tol,
):
    if not reference_path.exists():
        raise FileNotFoundError(
            f"找不到基准预测文件：{reference_path}"
        )

    ref = pd.read_csv(reference_path)

    required = {
        "Group",
        "Window",
        "Model",
        "Signal",
        "TargetDay",
        "y_true",
        "y_pred",
    }

    missing = required - set(ref.columns)

    if missing:
        raise ValueError(
            f"基准预测文件缺少字段：{sorted(missing)}"
        )

    ref["Group"] = ref["Group"].astype(str)

    ref = ref[
        (ref["Model"] == "Similarity_DTW_All4_K2")
        & ref["Signal"].isin(CORE3)
    ].copy()

    if ref.empty:
        raise ValueError(
            "基准文件中没有 Similarity_DTW_All4_K2。"
        )

    keys = ["Group", "Window", "Signal", "TargetDay"]

    ref = (
        ref[keys + ["y_true", "y_pred"]]
        .drop_duplicates(keys)
        .rename(
            columns={
                "y_true": "y_true_ref",
                "y_pred": "y_pred_ref",
            }
        )
    )

    cur = (
        all4_pred[keys + ["y_true", "y_pred"]]
        .copy()
        .rename(
            columns={
                "y_true": "y_true_07",
                "y_pred": "y_pred_07",
            }
        )
    )

    cur["Group"] = cur["Group"].astype(str)

    cmp = cur.merge(
        ref,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not (cmp["_merge"] == "both").all():
        raise RuntimeError(
            "07 All4 与 03/04 的预测行不一致。"
        )

    true_diff = np.abs(
        cmp["y_true_07"].to_numpy(float)
        - cmp["y_true_ref"].to_numpy(float)
    )

    pred_diff = np.abs(
        cmp["y_pred_07"].to_numpy(float)
        - cmp["y_pred_ref"].to_numpy(float)
    )

    max_true = float(np.nanmax(true_diff))
    max_pred = float(np.nanmax(pred_diff))

    passed = (
        max_true <= tol
        and max_pred <= tol
    )

    audit = pd.DataFrame(
        [
            {
                "ReferenceFile": str(reference_path),
                "N_Rows_Compared": int(len(cmp)),
                "Tolerance": float(tol),
                "MaxAbsDiff_y_true": max_true,
                "MaxAbsDiff_y_pred": max_pred,
                "Baseline_PASS": bool(passed),
            }
        ]
    )

    if not passed:
        raise RuntimeError(
            "Baseline validation FAIL: "
            f"max |Δy_pred|={max_pred:.12g}, "
            f"tolerance={tol:.12g}"
        )

    return audit


# ======================================================================
# Group sufficient statistics
# ======================================================================

def build_group_stats(pred):
    groups = sorted(pred["Group"].unique().tolist())
    windows = sorted(int(x) for x in pred["Window"].unique())

    g_index = {g: i for i, g in enumerate(groups)}

    stats = {}

    for window in windows:
        stats[window] = {}

        for signal in CORE3:
            part = pred[
                (pred["Window"] == window)
                & (pred["Signal"] == signal)
                & pred["y_true"].notna()
                & pred["y_pred"].notna()
            ]

            n = np.zeros(len(groups), dtype=float)
            sy = np.zeros(len(groups), dtype=float)
            sy2 = np.zeros(len(groups), dtype=float)
            sse = np.zeros(len(groups), dtype=float)

            for group, gp in part.groupby("Group"):
                i = g_index[str(group)]

                y = gp["y_true"].to_numpy(float)
                p = gp["y_pred"].to_numpy(float)

                n[i] = len(y)
                sy[i] = np.sum(y)
                sy2[i] = np.sum(y ** 2)
                sse[i] = np.sum((p - y) ** 2)

            stats[window][signal] = {
                "n": n,
                "sy": sy,
                "sy2": sy2,
                "sse": sse,
            }

    return groups, windows, stats


def nrmse_from_counts(counts, signal_stats):
    n = float(np.dot(counts, signal_stats["n"]))

    if n <= 1:
        return np.nan

    sy = float(np.dot(counts, signal_stats["sy"]))
    sy2 = float(np.dot(counts, signal_stats["sy2"]))
    sse = float(np.dot(counts, signal_stats["sse"]))

    rmse = np.sqrt(sse / n)

    variance = (
        sy2 - (sy ** 2) / n
    ) / (n - 1)

    variance = max(float(variance), 0.0)
    sd = np.sqrt(variance)

    if np.isclose(sd, 0.0):
        return np.nan

    return float(rmse / sd)


def composite_error(counts, window, stats):
    vals = []

    for signal in CORE3:
        value = nrmse_from_counts(
            counts,
            stats[window][signal],
        )

        if not np.isnan(value):
            vals.append(value)

    return float(np.mean(vals)) if vals else np.nan


# ======================================================================
# Shared paired bootstrap across variants
# ======================================================================

def bootstrap_all_variants(
    predictions: pd.DataFrame,
    variants: list[str],
    b: int,
    seed: int,
):
    first = predictions[
        predictions["Variant"] == variants[0]
    ]

    groups = sorted(first["Group"].unique().tolist())
    n_groups = len(groups)

    rng = np.random.default_rng(seed)

    prob = np.repeat(1.0 / n_groups, n_groups)

    counts_matrix = rng.multinomial(
        n_groups,
        prob,
        size=b,
    )

    store = {}

    for variant in variants:
        part = predictions[
            predictions["Variant"] == variant
        ]

        vg, windows, stats = build_group_stats(part)

        if vg != groups:
            raise ValueError(
                f"{variant} 的 Group 集合与 All4 不一致。"
            )

        error_boot = np.empty(
            (b, len(windows)),
            dtype=float,
        )

        full_counts = np.ones(n_groups, dtype=float)

        point = np.asarray(
            [
                composite_error(
                    full_counts,
                    w,
                    stats,
                )
                for w in windows
            ],
            dtype=float,
        )

        for bi in range(b):
            counts = counts_matrix[bi]

            for wi, window in enumerate(windows):
                error_boot[bi, wi] = composite_error(
                    counts,
                    window,
                    stats,
                )

        gain_boot = (
            error_boot[:, :-1]
            - error_boot[:, 1:]
        )

        store[variant] = {
            "groups": groups,
            "windows": windows,
            "stats": stats,
            "point_errors": point,
            "error_boot": error_boot,
            "gain_boot": gain_boot,
        }

    return store


# ======================================================================
# Error/Gain tables
# ======================================================================

def build_error_gain_tables(
    store,
    b,
    gain_delta,
):
    error_rows = []
    gain_rows = []

    for variant, info in store.items():
        windows = info["windows"]
        point = info["point_errors"]
        error_boot = info["error_boot"]
        gain_boot = info["gain_boot"]

        for wi, window in enumerate(windows):
            vals = error_boot[:, wi]
            vals = vals[~np.isnan(vals)]

            low, high = np.percentile(vals, [2.5, 97.5])

            error_rows.append(
                {
                    "Variant": variant,
                    "Window": int(window),
                    "Error_Point": float(point[wi]),
                    "Error_CI_Low": float(low),
                    "Error_CI_High": float(high),
                    "Bootstrap_B": int(b),
                }
            )

        for gi in range(len(windows) - 1):
            vals = gain_boot[:, gi]
            vals = vals[~np.isnan(vals)]

            low, high = np.percentile(vals, [2.5, 97.5])

            gain_rows.append(
                {
                    "Variant": variant,
                    "FromWindow": int(windows[gi]),
                    "ToWindow": int(windows[gi + 1]),
                    "Gain_Point": float(
                        point[gi] - point[gi + 1]
                    ),
                    "Gain_CI_Low": float(low),
                    "Gain_CI_High": float(high),
                    "P_Gain_GT_0": float(np.mean(vals > 0)),
                    "Gain_Delta": float(gain_delta),
                    "P_Gain_GT_Delta": float(
                        np.mean(vals > gain_delta)
                    ),
                }
            )

    return (
        pd.DataFrame(error_rows),
        pd.DataFrame(gain_rows),
    )


# ======================================================================
# Group-level stability and MSOH
# ======================================================================

def build_group_error(pred):
    rows = []

    scales = {}

    for (
        variant,
        window,
        signal,
    ), part in pred.groupby(
        ["Variant", "Window", "Signal"]
    ):
        y = part["y_true"].dropna().to_numpy(float)

        scales[
            (variant, int(window), signal)
        ] = (
            float(np.std(y, ddof=1))
            if len(y) > 1
            else np.nan
        )

    for (
        variant,
        group,
        window,
    ), part in pred.groupby(
        ["Variant", "Group", "Window"]
    ):
        vals = []

        for signal in CORE3:
            sub = part[
                (part["Signal"] == signal)
                & part["y_true"].notna()
                & part["y_pred"].notna()
            ]

            if sub.empty:
                continue

            sd = scales[
                (variant, int(window), signal)
            ]

            if pd.isna(sd) or np.isclose(sd, 0):
                continue

            y = sub["y_true"].to_numpy(float)
            p = sub["y_pred"].to_numpy(float)

            rmse = np.sqrt(np.mean((p - y) ** 2))
            vals.append(float(rmse / sd))

        rows.append(
            {
                "Variant": variant,
                "Group": str(group),
                "Window": int(window),
                "GroupCore3Error": (
                    float(np.mean(vals))
                    if vals
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_stability(group_error, epsilon):
    rows = []

    for variant, part in group_error.groupby("Variant"):
        pivot = part.pivot(
            index="Group",
            columns="Window",
            values="GroupCore3Error",
        )

        windows = sorted(int(x) for x in pivot.columns)

        for fw, tw in zip(windows[:-1], windows[1:]):
            prev = pivot[fw]
            curr = pivot[tw]

            valid = prev.notna() & curr.notna()

            gain = (
                prev[valid] - curr[valid]
            ).to_numpy(float)

            rows.append(
                {
                    "Variant": variant,
                    "FromWindow": int(fw),
                    "ToWindow": int(tw),
                    "Group_Epsilon": float(epsilon),
                    "Stable_Group_Proportion": (
                        float(np.mean(np.abs(gain) <= epsilon))
                        if len(gain)
                        else np.nan
                    ),
                    "Improved_Group_Proportion": (
                        float(np.mean(gain > epsilon))
                        if len(gain)
                        else np.nan
                    ),
                    "Worsened_Group_Proportion": (
                        float(np.mean(gain < -epsilon))
                        if len(gain)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def decide_msoh(
    error_table,
    gain_table,
    stability,
    error_threshold,
    gain_delta,
    alpha,
    stability_threshold,
):
    trace = gain_table.merge(
        stability[
            [
                "Variant",
                "FromWindow",
                "ToWindow",
                "Stable_Group_Proportion",
                "Improved_Group_Proportion",
                "Worsened_Group_Proportion",
            ]
        ],
        on=[
            "Variant",
            "FromWindow",
            "ToWindow",
        ],
        how="left",
    )

    current_error = error_table[
        [
            "Variant",
            "Window",
            "Error_Point",
            "Error_CI_Low",
            "Error_CI_High",
        ]
    ].rename(
        columns={"Window": "FromWindow"}
    )

    trace = trace.merge(
        current_error,
        on=["Variant", "FromWindow"],
        how="left",
    )

    trace["Error_Threshold"] = float(error_threshold)
    trace["Gain_Delta"] = float(gain_delta)
    trace["Alpha"] = float(alpha)
    trace["Stability_Threshold"] = float(stability_threshold)

    trace["Condition_Error"] = (
        trace["Error_Point"]
        <= error_threshold
    )

    trace["Condition_Saturation"] = (
        trace["P_Gain_GT_Delta"]
        < alpha
    )

    trace["Condition_Stability"] = (
        trace["Stable_Group_Proportion"]
        >= stability_threshold
    )

    trace["All_Conditions"] = (
        trace["Condition_Error"]
        & trace["Condition_Saturation"]
        & trace["Condition_Stability"]
    )

    rows = []

    for variant, part in trace.groupby("Variant"):
        passed = (
            part[
                part["All_Conditions"]
            ]
            .sort_values("FromWindow")
        )

        if passed.empty:
            msoh = np.nan
            status = "No window satisfied all criteria"
        else:
            msoh = int(
                passed.iloc[0]["FromWindow"]
            )
            status = "Candidate MSOH identified"

        rows.append(
            {
                "Variant": variant,
                "Candidate_MSOH": msoh,
                "Decision_Status": status,
            }
        )

    return trace, pd.DataFrame(rows)


# ======================================================================
# Bootstrap signal contribution
# ======================================================================

def build_signal_contribution_bootstrap(
    store,
    contribution_delta,
):
    if "All4" not in store:
        raise ValueError("缺少 All4 bootstrap 基准。")

    base = store["All4"]

    rows = []

    for variant, info in store.items():
        if variant == "All4":
            continue

        if info["windows"] != base["windows"]:
            raise ValueError(
                f"{variant} 窗口与 All4 不一致。"
            )

        removed = variant.replace("Minus_", "")

        for wi, window in enumerate(base["windows"]):
            point_c = (
                info["point_errors"][wi]
                - base["point_errors"][wi]
            )

            boot_c = (
                info["error_boot"][:, wi]
                - base["error_boot"][:, wi]
            )

            boot_c = boot_c[
                ~np.isnan(boot_c)
            ]

            low, high = np.percentile(
                boot_c,
                [2.5, 97.5],
            )

            rows.append(
                {
                    "RemovedSignal": removed,
                    "Variant": variant,
                    "Window": int(window),
                    "Error_Without_Signal": float(
                        info["point_errors"][wi]
                    ),
                    "Error_All4": float(
                        base["point_errors"][wi]
                    ),
                    "Contribution_C_s": float(point_c),
                    "Contribution_CI_Low": float(low),
                    "Contribution_CI_High": float(high),
                    "P_C_GT_0": float(
                        np.mean(boot_c > 0)
                    ),
                    "Contribution_Delta": float(
                        contribution_delta
                    ),
                    "P_C_GT_Delta": float(
                        np.mean(
                            boot_c > contribution_delta
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_msoh_shift(msoh_table):
    base = msoh_table[
        msoh_table["Variant"] == "All4"
    ]

    if base.empty:
        raise ValueError("缺少 All4 MSOH。")

    base_msoh = base.iloc[0]["Candidate_MSOH"]

    rows = []

    for _, row in msoh_table.iterrows():
        variant = row["Variant"]

        if variant == "All4":
            continue

        removed = variant.replace("Minus_", "")
        current = row["Candidate_MSOH"]

        if pd.isna(base_msoh) or pd.isna(current):
            shift = np.nan
        else:
            shift = float(current - base_msoh)

        rows.append(
            {
                "RemovedSignal": removed,
                "Variant": variant,
                "MSOH_All4": base_msoh,
                "MSOH_Without_Signal": current,
                "Delta_MSOH_s": shift,
            }
        )

    return pd.DataFrame(rows)


def build_contribution_summary(
    contribution,
    msoh_shift,
):
    rows = []

    for signal, part in contribution.groupby("RemovedSignal"):
        shift_row = msoh_shift[
            msoh_shift["RemovedSignal"] == signal
        ]

        delta_msoh = (
            shift_row.iloc[0]["Delta_MSOH_s"]
            if not shift_row.empty
            else np.nan
        )

        peak_positive = part.loc[
            part["Contribution_C_s"].idxmax()
        ]

        peak_abs = part.loc[
            part["Contribution_C_s"].abs().idxmax()
        ]

        rows.append(
            {
                "Signal": signal,
                "Mean_Contribution_C_s": float(
                    part["Contribution_C_s"].mean()
                ),
                "Max_Positive_Contribution_C_s": float(
                    peak_positive["Contribution_C_s"]
                ),
                "Max_Positive_Window": int(
                    peak_positive["Window"]
                ),
                "P_C_GT_0_At_Max_Positive": float(
                    peak_positive["P_C_GT_0"]
                ),
                "CI_Low_At_Max_Positive": float(
                    peak_positive["Contribution_CI_Low"]
                ),
                "CI_High_At_Max_Positive": float(
                    peak_positive["Contribution_CI_High"]
                ),
                "Most_Influential_Window_ByAbs": int(
                    peak_abs["Window"]
                ),
                "Contribution_At_Most_Influential_Window": float(
                    peak_abs["Contribution_C_s"]
                ),
                "Delta_MSOH_s": delta_msoh,
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# Figures
# ======================================================================

def savefig(fig, figure_dir, stem):
    fig.tight_layout()

    fig.savefig(
        figure_dir / f"{stem}.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig.savefig(
        figure_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_contribution_ci(
    contribution,
    figure_dir,
):
    for signal, part in contribution.groupby("RemovedSignal"):
        part = part.sort_values("Window")

        y = part["Contribution_C_s"].to_numpy(float)

        low = (
            y
            - part["Contribution_CI_Low"].to_numpy(float)
        )

        high = (
            part["Contribution_CI_High"].to_numpy(float)
            - y
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(
            part["Window"],
            y,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=3,
        )

        ax.axhline(0, linewidth=1)

        ax.set_xlabel("Observation Window End Day")
        ax.set_ylabel("C_s(k)")
        ax.set_title(
            f"Signal Contribution with 95% Bootstrap CI: {signal}"
        )

        safe = signal.lower()

        savefig(
            fig,
            figure_dir,
            f"fig05_contribution_ci_{safe}",
        )


def plot_contribution_probability(
    contribution,
    figure_dir,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    for signal, part in contribution.groupby("RemovedSignal"):
        part = part.sort_values("Window")

        ax.plot(
            part["Window"],
            part["P_C_GT_0"],
            marker="o",
            label=signal,
        )

    ax.set_xlabel("Observation Window End Day")
    ax.set_ylabel("P(C_s(k) > 0)")
    ax.set_ylim(0, 1)
    ax.set_title("Bootstrap Probability of Positive Signal Contribution")
    ax.legend()

    savefig(
        fig,
        figure_dir,
        "fig06_contribution_probability",
    )


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    if args.bootstrap < 100:
        raise ValueError(
            "正式分析建议 bootstrap >= 1000；默认 5000。"
        )

    windows = parse_int_list(args.windows)
    target_days = parse_int_list(args.target_days)

    table_dir = args.output / "table"
    figure_dir = args.output / "figure"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("07 信号贡献 + 不确定性分析开始")
    print("=" * 100)

    print("[1/7] 读取清洗后的纵向数据...")
    df = load_data(args.input)

    variants = {
        "All4": ALL_SIGNALS.copy(),
        "Minus_Weight": [
            "Status_Score",
            "Feeding",
            "Defecation",
        ],
        "Minus_Status_Score": [
            "Weight",
            "Feeding",
            "Defecation",
        ],
        "Minus_Feeding": [
            "Weight",
            "Status_Score",
            "Defecation",
        ],
        "Minus_Defecation": [
            "Weight",
            "Status_Score",
            "Feeding",
        ],
    }

    print("[2/7] 运行 All4 与 leave-one-signal-out LOOG DTW 预测...")

    pred_list = []
    neigh_list = []

    for i, (variant, signals) in enumerate(
        variants.items(),
        start=1,
    ):
        print(
            f"      ({i}/{len(variants)}) "
            f"{variant}: {', '.join(signals)}"
        )

        pred, neigh = predict_variant(
            df=df,
            variant=variant,
            included_signals=signals,
            windows=windows,
            target_days=target_days,
            k_neighbors=args.k,
            dtw_radius=args.dtw_radius,
            defecation_lambda=args.defecation_lambda,
        )

        pred_list.append(pred)
        neigh_list.append(neigh)

    predictions = pd.concat(
        pred_list,
        ignore_index=True,
    )

    neighbors = pd.concat(
        neigh_list,
        ignore_index=True,
    )

    print("[3/7] Baseline validation: 07 All4 vs 03/04 DTW All4...")

    baseline_audit = validate_all4_baseline(
        predictions[
            predictions["Variant"] == "All4"
        ].copy(),
        args.reference_predictions,
        args.baseline_tol,
    )

    print(
        "      PASS: max |Δy_pred| = "
        f"{baseline_audit.iloc[0]['MaxAbsDiff_y_pred']:.3e}"
    )

    print("[4/7] 用同一轮 Group bootstrap 计算所有变体的 Error/Gain...")

    store = bootstrap_all_variants(
        predictions=predictions,
        variants=list(variants.keys()),
        b=args.bootstrap,
        seed=args.seed,
    )

    error_table, gain_table = build_error_gain_tables(
        store=store,
        b=args.bootstrap,
        gain_delta=args.gain_delta,
    )

    print("[5/7] 计算组级稳定性并应用 06 的 MSOH 规则...")

    group_error = build_group_error(predictions)

    stability = build_stability(
        group_error,
        args.group_epsilon,
    )

    trace, msoh = decide_msoh(
        error_table=error_table,
        gain_table=gain_table,
        stability=stability,
        error_threshold=args.error_threshold,
        gain_delta=args.gain_delta,
        alpha=args.alpha,
        stability_threshold=args.stability_threshold,
    )

    print("[6/7] 计算 paired-bootstrap C_s(k) 置信区间与概率...")

    contribution = build_signal_contribution_bootstrap(
        store=store,
        contribution_delta=args.contribution_delta,
    )

    msoh_shift = build_msoh_shift(msoh)

    summary = build_contribution_summary(
        contribution,
        msoh_shift,
    )

    print("[7/7] 保存表格和图形...")

    tables = {
        "baseline_validation.csv": baseline_audit,
        "signal_variant_predictions.csv": predictions,
        "signal_variant_neighbors.csv": neighbors,
        "signal_variant_error.csv": error_table,
        "signal_variant_gain.csv": gain_table,
        "signal_variant_stability.csv": stability,
        "signal_variant_msoh_trace.csv": trace,
        "signal_variant_msoh.csv": msoh,
        "signal_contribution_by_window_bootstrap.csv": contribution,
        "signal_msoh_shift.csv": msoh_shift,
        "signal_contribution_summary_bootstrap.csv": summary,
    }

    for name, table in tables.items():
        table.to_csv(
            table_dir / name,
            index=False,
            encoding="utf-8-sig",
        )

    plot_contribution_ci(
        contribution,
        figure_dir,
    )

    plot_contribution_probability(
        contribution,
        figure_dir,
    )

    print()
    print("=" * 100)
    print("07 信号贡献 + 不确定性分析完成")
    print("=" * 100)

    print("\n各变体候选 MSOH：")
    print(
        msoh[
            [
                "Variant",
                "Candidate_MSOH",
                "Decision_Status",
            ]
        ].to_string(index=False)
    )

    print("\n信号对 MSOH 的影响：")
    print(
        msoh_shift.to_string(index=False)
    )

    print("\nBootstrap 信号贡献摘要：")
    print(
        summary.to_string(index=False)
    )

    print("\n重点窗口的 C_s(k)：")

    focus = contribution[
        (
            (contribution["RemovedSignal"] == "Defecation")
            & (contribution["Window"] == 4)
        )
        |
        (
            (contribution["RemovedSignal"] == "Weight")
            & (contribution["Window"] == 4)
        )
        |
        (
            (contribution["RemovedSignal"] == "Status_Score")
            & (contribution["Window"] == 3)
        )
    ]

    if not focus.empty:
        print(
            focus[
                [
                    "RemovedSignal",
                    "Window",
                    "Contribution_C_s",
                    "Contribution_CI_Low",
                    "Contribution_CI_High",
                    "P_C_GT_0",
                    "P_C_GT_Delta",
                ]
            ].to_string(index=False)
        )

    print()
    print(
        "说明：C_s(k) 为 paired group bootstrap 贡献，"
        "同一轮 bootstrap 中 All4 与 -signal 使用完全相同的 Group 重采样。"
    )
    print(
        "注意：leave-one-signal-out 反映条件增量预测贡献，不等同于因果效应。"
    )
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] 07 增强版失败：{exc}",
            file=sys.stderr,
        )
        raise
