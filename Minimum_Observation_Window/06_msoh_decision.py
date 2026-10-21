"""
06_msoh_decision.py

Uncertainty-aware Minimum Sufficient Observation Horizon (MSOH) decision module.

Purpose
-------
Turn the existing Error(k) / Gain(k) analysis into an explicit,
uncertainty-aware MSOH decision rule without hard-coding a preferred day.

Decision logic
--------------
For a candidate window k, require:

1) Current predictive error is acceptable:
       Error(k) <= tau_E

2) The probability that the NEXT window provides a practically meaningful
   improvement is sufficiently small:
       P(Gain(k+1) > delta_G) < alpha

3) A sufficient proportion of groups are locally stable between k and k+1:
       Stable_Group_Proportion(k -> k+1) >= tau_S

where:
    Gain(k+1) = Error(k) - Error(k+1)

This script also runs threshold sensitivity analysis, because tau_E, delta_G,
alpha, and tau_S should not be chosen post hoc merely to obtain a desired MSOH.

Important
---------
- Independent bootstrap unit = Group.
- Core3 signals = Weight, Status_Score, Feeding.
- Error statistic matches 04/05:
      signal NRMSE = RMSE / SD(y_true, ddof=1)
      Error(k) = equal-weight mean of Core3 signal NRMSEs.
- For Gain bootstrap, the SAME group resample is used across all windows
  within a method, preserving pairing.
- No claim is made that welfare/safety monitoring can stop at MSOH.

Default thresholds are PROVISIONAL EXPLORATORY defaults, not final scientific
cutoffs. Use the sensitivity tables before locking the protocol.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("results/04_output/table/combined_predictions.csv")
DEFAULT_OUTPUT = Path("results/06_output")

CORE_SIGNALS = ["Weight", "Status_Score", "Feeding"]
SIMILARITY_PREFIX = "Similarity_"

DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260924

# ---------------------------------------------------------------------
# PROVISIONAL exploratory thresholds.
# These are deliberately exposed as CLI options and sensitivity grids.
# They must not be treated as locked scientific cutoffs yet.
# ---------------------------------------------------------------------
DEFAULT_ERROR_THRESHOLD = 0.40
DEFAULT_GAIN_DELTA = 0.03
DEFAULT_ALPHA = 0.10
DEFAULT_STABILITY_THRESHOLD = 0.75
DEFAULT_GROUP_EPSILON = 0.03


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="06: Uncertainty-aware MSOH decision and threshold sensitivity"
    )

    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--error-threshold", type=float, default=DEFAULT_ERROR_THRESHOLD)
    p.add_argument("--gain-delta", type=float, default=DEFAULT_GAIN_DELTA)
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
        help="Absolute group-level error change treated as locally stable.",
    )

    # Sensitivity grids
    p.add_argument(
        "--error-grid",
        type=str,
        default="0.35,0.40,0.45,0.50",
    )
    p.add_argument(
        "--gain-delta-grid",
        type=str,
        default="0.01,0.02,0.03,0.05",
    )
    p.add_argument(
        "--alpha-grid",
        type=str,
        default="0.05,0.10,0.20",
    )
    p.add_argument(
        "--stability-grid",
        type=str,
        default="0.50,0.67,0.75,0.80",
    )

    return p.parse_args()


def similarity_label(model: str) -> str:
    mapping = {
        "Similarity_EuclideanFeature_K2": "Euclidean Feature",
        "Similarity_DTW_Core3_K2": "DTW Core3",
        "Similarity_DTW_All4_K2": "DTW All4",
    }
    return mapping.get(str(model), str(model))


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)

    required = {
        "Group",
        "Window",
        "Model",
        "Signal",
        "TargetDay",
        "y_true",
        "y_pred",
        "SourceMethod",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少字段：{sorted(missing)}")

    df["Group"] = df["Group"].astype(str)

    df = df[
        df["Model"].astype(str).str.startswith(SIMILARITY_PREFIX)
        & df["Signal"].isin(CORE_SIGNALS)
    ].copy()

    if df.empty:
        raise ValueError("没有找到 Core3 Similarity 预测记录。")

    return df


def method_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["SourceMethod", "Model"]]
        .drop_duplicates()
        .sort_values(["SourceMethod", "Model"])
        .reset_index(drop=True)
    )


# =====================================================================
# Fast group-aggregate representation
# =====================================================================

def build_group_aggregates(method_df: pd.DataFrame):
    """
    Build group-level sufficient statistics for each window/signal.

    For each group:
      n
      sum_y
      sum_y2
      sse = sum((y_pred - y_true)^2)

    These are enough to recompute, under a cluster bootstrap:
      RMSE
      SD(y_true, ddof=1)
      NRMSE
    exactly for the resampled dataset, including repeated groups.
    """

    groups = sorted(method_df["Group"].unique().tolist())
    windows = sorted(int(x) for x in method_df["Window"].unique())

    g_index = {g: i for i, g in enumerate(groups)}

    stats = {}

    for window in windows:
        stats[window] = {}

        for signal in CORE_SIGNALS:
            part = method_df[
                (method_df["Window"] == window)
                & (method_df["Signal"] == signal)
                & method_df["y_true"].notna()
                & method_df["y_pred"].notna()
            ]

            n = np.zeros(len(groups), dtype=float)
            sum_y = np.zeros(len(groups), dtype=float)
            sum_y2 = np.zeros(len(groups), dtype=float)
            sse = np.zeros(len(groups), dtype=float)

            for group, gp in part.groupby("Group"):
                i = g_index[str(group)]

                y = gp["y_true"].to_numpy(dtype=float)
                p = gp["y_pred"].to_numpy(dtype=float)

                n[i] = len(y)
                sum_y[i] = np.sum(y)
                sum_y2[i] = np.sum(y ** 2)
                sse[i] = np.sum((p - y) ** 2)

            stats[window][signal] = {
                "n": n,
                "sum_y": sum_y,
                "sum_y2": sum_y2,
                "sse": sse,
            }

    return groups, windows, stats


def nrmse_from_aggregate(n, sum_y, sum_y2, sse) -> float:
    if n <= 1:
        return np.nan

    rmse = np.sqrt(sse / n)

    # Sample variance, ddof=1
    ss_centered = sum_y2 - (sum_y ** 2) / n
    variance = ss_centered / (n - 1)

    # Numerical guard
    variance = max(float(variance), 0.0)
    sd = np.sqrt(variance)

    if np.isclose(sd, 0.0):
        return np.nan

    return float(rmse / sd)


def error_from_counts(
    counts: np.ndarray,
    window: int,
    stats: dict,
) -> float:
    signal_values = []

    for signal in CORE_SIGNALS:
        s = stats[window][signal]

        n = float(np.dot(counts, s["n"]))
        sum_y = float(np.dot(counts, s["sum_y"]))
        sum_y2 = float(np.dot(counts, s["sum_y2"]))
        sse = float(np.dot(counts, s["sse"]))

        value = nrmse_from_aggregate(
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sse=sse,
        )

        if not np.isnan(value):
            signal_values.append(value)

    if not signal_values:
        return np.nan

    return float(np.mean(signal_values))


def full_sample_error(window: int, stats: dict, n_groups: int) -> float:
    counts = np.ones(n_groups, dtype=float)
    return error_from_counts(counts, window, stats)


def generate_bootstrap_counts(
    n_groups: int,
    b: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Each row is a multinomial count vector representing one cluster bootstrap:
    sample n_groups groups with replacement from n_groups groups.
    """
    prob = np.repeat(1.0 / n_groups, n_groups)
    return rng.multinomial(n_groups, prob, size=b)


def bootstrap_error_matrix(
    windows: list[int],
    stats: dict,
    counts_matrix: np.ndarray,
) -> np.ndarray:
    """
    Shape: B x n_windows.
    SAME bootstrap group counts are reused across all windows.
    """
    b = counts_matrix.shape[0]
    out = np.empty((b, len(windows)), dtype=float)

    for wi, window in enumerate(windows):
        for bi in range(b):
            out[bi, wi] = error_from_counts(
                counts_matrix[bi],
                window,
                stats,
            )

    return out


# =====================================================================
# Point estimates / bootstrap probability
# =====================================================================

def build_method_uncertainty(
    df: pd.DataFrame,
    bootstrap_b: int,
    seed: int,
    gain_delta: float,
):
    """
    Returns:
      point_error_table
      gain_table
      bootstrap_store

    bootstrap_store key:
      (SourceMethod, Model) -> dict(
          windows,
          error_boot,
          gain_boot,
          groups,
          stats
      )
    """

    error_rows = []
    gain_rows = []
    store = {}

    methods = method_table(df)

    for mi, mrow in methods.iterrows():
        source_method = mrow["SourceMethod"]
        model = mrow["Model"]

        part = df[
            (df["SourceMethod"] == source_method)
            & (df["Model"] == model)
        ].copy()

        groups, windows, stats = build_group_aggregates(part)

        # deterministic but method-distinct RNG stream
        rng = np.random.default_rng(seed + mi * 1009)

        counts = generate_bootstrap_counts(
            n_groups=len(groups),
            b=bootstrap_b,
            rng=rng,
        )

        error_boot = bootstrap_error_matrix(
            windows=windows,
            stats=stats,
            counts_matrix=counts,
        )

        point_errors = np.array(
            [
                full_sample_error(w, stats, len(groups))
                for w in windows
            ],
            dtype=float,
        )

        for wi, window in enumerate(windows):
            vals = error_boot[:, wi]
            vals = vals[~np.isnan(vals)]

            low, high = np.percentile(vals, [2.5, 97.5])

            error_rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "Window": int(window),
                "Error_Point": float(point_errors[wi]),
                "Error_CI_Low": float(low),
                "Error_CI_High": float(high),
                "N_Groups": len(groups),
                "Bootstrap_B": bootstrap_b,
            })

        # gain from k -> k+1
        gain_boot = error_boot[:, :-1] - error_boot[:, 1:]

        for gi in range(len(windows) - 1):
            from_w = windows[gi]
            to_w = windows[gi + 1]

            vals = gain_boot[:, gi]
            vals = vals[~np.isnan(vals)]

            point_gain = point_errors[gi] - point_errors[gi + 1]
            low, high = np.percentile(vals, [2.5, 97.5])

            gain_rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "FromWindow": int(from_w),
                "ToWindow": int(to_w),
                "Gain_Point": float(point_gain),
                "Gain_CI_Low": float(low),
                "Gain_CI_High": float(high),
                "P_Gain_GT_0": float(np.mean(vals > 0)),
                "Gain_Delta": float(gain_delta),
                "P_Gain_GT_Delta": float(np.mean(vals > gain_delta)),
                "P_Gain_LE_Delta": float(np.mean(vals <= gain_delta)),
                "N_Groups": len(groups),
                "Bootstrap_B": bootstrap_b,
            })

        store[(source_method, model)] = {
            "groups": groups,
            "windows": windows,
            "stats": stats,
            "point_errors": point_errors,
            "error_boot": error_boot,
            "gain_boot": gain_boot,
        }

    return (
        pd.DataFrame(error_rows),
        pd.DataFrame(gain_rows),
        store,
    )


# =====================================================================
# Group-level local stability
# =====================================================================

def group_signal_scales(method_df: pd.DataFrame):
    """
    Fixed full-sample signal SD for each window/signal.
    Used only for descriptive group-level local stability.
    """
    scale = {}

    for window in sorted(method_df["Window"].unique()):
        scale[int(window)] = {}

        for signal in CORE_SIGNALS:
            part = method_df[
                (method_df["Window"] == window)
                & (method_df["Signal"] == signal)
                & method_df["y_true"].notna()
            ]

            y = part["y_true"].to_numpy(dtype=float)

            scale[int(window)][signal] = (
                float(np.std(y, ddof=1))
                if len(y) > 1
                else np.nan
            )

    return scale


def group_window_error_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, mrow in method_table(df).iterrows():
        source_method = mrow["SourceMethod"]
        model = mrow["Model"]

        method_df = df[
            (df["SourceMethod"] == source_method)
            & (df["Model"] == model)
        ].copy()

        scales = group_signal_scales(method_df)

        for (group, window), part in method_df.groupby(
            ["Group", "Window"],
            sort=True,
        ):
            values = []

            for signal in CORE_SIGNALS:
                sub = part[
                    (part["Signal"] == signal)
                    & part["y_true"].notna()
                    & part["y_pred"].notna()
                ]

                if sub.empty:
                    continue

                sd = scales[int(window)][signal]

                if pd.isna(sd) or np.isclose(sd, 0.0):
                    continue

                err = (
                    sub["y_pred"].to_numpy(dtype=float)
                    - sub["y_true"].to_numpy(dtype=float)
                )

                rmse = float(np.sqrt(np.mean(err ** 2)))
                values.append(rmse / sd)

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "Group": str(group),
                "Window": int(window),
                "GroupCore3Error": (
                    float(np.mean(values))
                    if values
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def build_stability_table(
    group_errors: pd.DataFrame,
    epsilon: float,
) -> pd.DataFrame:
    rows = []

    keys = [
        "SourceMethod",
        "Model",
        "SimilarityLabel",
    ]

    for key_vals, part in group_errors.groupby(keys, sort=True):
        source_method, model, label = key_vals

        pivot = part.pivot(
            index="Group",
            columns="Window",
            values="GroupCore3Error",
        )

        windows = sorted(int(x) for x in pivot.columns)

        for from_w, to_w in zip(windows[:-1], windows[1:]):
            prev = pivot[from_w]
            curr = pivot[to_w]

            valid = prev.notna() & curr.notna()

            group_gain = prev[valid] - curr[valid]

            n = len(group_gain)

            if n == 0:
                stable_prop = np.nan
                improved_prop = np.nan
                worsened_prop = np.nan
            else:
                stable_prop = float(
                    np.mean(np.abs(group_gain.to_numpy()) <= epsilon)
                )
                improved_prop = float(
                    np.mean(group_gain.to_numpy() > epsilon)
                )
                worsened_prop = float(
                    np.mean(group_gain.to_numpy() < -epsilon)
                )

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": label,
                "FromWindow": int(from_w),
                "ToWindow": int(to_w),
                "Group_Epsilon": float(epsilon),
                "Stable_Group_Proportion": stable_prop,
                "Improved_Group_Proportion": improved_prop,
                "Worsened_Group_Proportion": worsened_prop,
                "N_Groups": int(n),
            })

    return pd.DataFrame(rows)


# =====================================================================
# MSOH decision
# =====================================================================

def calculate_gain_probability_for_delta(
    store: dict,
    gain_delta: float,
) -> pd.DataFrame:
    rows = []

    for (source_method, model), info in store.items():
        windows = info["windows"]
        gain_boot = info["gain_boot"]
        point_errors = info["point_errors"]

        for gi in range(len(windows) - 1):
            vals = gain_boot[:, gi]
            vals = vals[~np.isnan(vals)]

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "FromWindow": int(windows[gi]),
                "ToWindow": int(windows[gi + 1]),
                "Error_From": float(point_errors[gi]),
                "Error_To": float(point_errors[gi + 1]),
                "Gain_Point": float(
                    point_errors[gi] - point_errors[gi + 1]
                ),
                "Gain_Delta": float(gain_delta),
                "P_Gain_GT_Delta": float(np.mean(vals > gain_delta)),
                "P_Gain_LE_Delta": float(np.mean(vals <= gain_delta)),
            })

    return pd.DataFrame(rows)


def decide_msoh(
    error_table: pd.DataFrame,
    gain_prob_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    error_threshold: float,
    gain_delta: float,
    alpha: float,
    stability_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Candidate k is evaluated against transition k -> k+1.

    Conditions:
      Error(k) <= error_threshold
      P(Gain(k+1) > gain_delta) < alpha
      Stable_Group_Proportion(k -> k+1) >= stability_threshold

    The earliest k satisfying all conditions is selected.
    The final available window cannot be selected because no k+1 exists
    to test saturation.
    """

    gain = gain_prob_table.copy()
    stable = stability_table.copy()

    decision = gain.merge(
        stable[
            [
                "SourceMethod",
                "Model",
                "FromWindow",
                "ToWindow",
                "Stable_Group_Proportion",
                "Improved_Group_Proportion",
                "Worsened_Group_Proportion",
            ]
        ],
        on=[
            "SourceMethod",
            "Model",
            "FromWindow",
            "ToWindow",
        ],
        how="left",
    )

    current_error = error_table[
        [
            "SourceMethod",
            "Model",
            "Window",
            "Error_Point",
            "Error_CI_Low",
            "Error_CI_High",
        ]
    ].rename(columns={"Window": "FromWindow"})

    decision = decision.merge(
        current_error,
        on=[
            "SourceMethod",
            "Model",
            "FromWindow",
        ],
        how="left",
        suffixes=("", "_Check"),
    )

    # prefer exact current Error from error table
    if "Error_Point" in decision.columns:
        decision["Error_Current"] = decision["Error_Point"]
    else:
        decision["Error_Current"] = decision["Error_From"]

    decision["Error_Threshold"] = float(error_threshold)
    decision["Gain_Delta"] = float(gain_delta)
    decision["Alpha"] = float(alpha)
    decision["Stability_Threshold"] = float(stability_threshold)

    decision["Condition_Error"] = (
        decision["Error_Current"] <= error_threshold
    )

    decision["Condition_Saturation"] = (
        decision["P_Gain_GT_Delta"] < alpha
    )

    decision["Condition_Stability"] = (
        decision["Stable_Group_Proportion"] >= stability_threshold
    )

    decision["All_Conditions"] = (
        decision["Condition_Error"]
        & decision["Condition_Saturation"]
        & decision["Condition_Stability"]
    )

    decision = decision.sort_values(
        ["SimilarityLabel", "FromWindow"]
    ).reset_index(drop=True)

    summary_rows = []

    for (
        source_method,
        model,
        label,
    ), part in decision.groupby(
        ["SourceMethod", "Model", "SimilarityLabel"],
        sort=True,
    ):
        passed = part[part["All_Conditions"]]

        if passed.empty:
            msoh = np.nan
            status = "No window satisfied all criteria"
        else:
            msoh = int(passed.iloc[0]["FromWindow"])
            status = "Candidate MSOH identified"

        summary_rows.append({
            "SourceMethod": source_method,
            "Model": model,
            "SimilarityLabel": label,
            "Candidate_MSOH": msoh,
            "Decision_Status": status,
            "Error_Threshold": float(error_threshold),
            "Gain_Delta": float(gain_delta),
            "Alpha": float(alpha),
            "Stability_Threshold": float(stability_threshold),
        })

    return decision, pd.DataFrame(summary_rows)


# =====================================================================
# Threshold sensitivity
# =====================================================================

def threshold_sensitivity(
    error_table: pd.DataFrame,
    store: dict,
    stability_table: pd.DataFrame,
    error_grid: list[float],
    gain_delta_grid: list[float],
    alpha_grid: list[float],
    stability_grid: list[float],
) -> pd.DataFrame:
    rows = []

    gain_prob_cache = {}

    for delta in gain_delta_grid:
        gain_prob_cache[delta] = calculate_gain_probability_for_delta(
            store=store,
            gain_delta=delta,
        )

    for tau_e, delta, alpha, tau_s in itertools.product(
        error_grid,
        gain_delta_grid,
        alpha_grid,
        stability_grid,
    ):
        _, summary = decide_msoh(
            error_table=error_table,
            gain_prob_table=gain_prob_cache[delta],
            stability_table=stability_table,
            error_threshold=tau_e,
            gain_delta=delta,
            alpha=alpha,
            stability_threshold=tau_s,
        )

        for _, row in summary.iterrows():
            rows.append({
                "SimilarityLabel": row["SimilarityLabel"],
                "SourceMethod": row["SourceMethod"],
                "Model": row["Model"],
                "Error_Threshold": float(tau_e),
                "Gain_Delta": float(delta),
                "Alpha": float(alpha),
                "Stability_Threshold": float(tau_s),
                "Candidate_MSOH": row["Candidate_MSOH"],
                "Decision_Status": row["Decision_Status"],
            })

    return pd.DataFrame(rows)


def summarize_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, part in sensitivity.groupby("SimilarityLabel"):
        total = len(part)

        no_pick = part["Candidate_MSOH"].isna().sum()

        valid = part.dropna(subset=["Candidate_MSOH"])

        if valid.empty:
            rows.append({
                "SimilarityLabel": label,
                "N_Threshold_Combinations": total,
                "N_No_MSOH": int(no_pick),
                "P_No_MSOH": float(no_pick / total),
                "Most_Common_MSOH": np.nan,
                "Most_Common_MSOH_Proportion": np.nan,
                "MSOH_Min": np.nan,
                "MSOH_Max": np.nan,
            })
            continue

        counts = valid["Candidate_MSOH"].value_counts()
        mode_value = counts.index[0]
        mode_count = counts.iloc[0]

        rows.append({
            "SimilarityLabel": label,
            "N_Threshold_Combinations": total,
            "N_No_MSOH": int(no_pick),
            "P_No_MSOH": float(no_pick / total),
            "Most_Common_MSOH": int(mode_value),
            "Most_Common_MSOH_Proportion": float(mode_count / total),
            "MSOH_Min": int(valid["Candidate_MSOH"].min()),
            "MSOH_Max": int(valid["Candidate_MSOH"].max()),
        })

    return pd.DataFrame(rows)


# =====================================================================
# Plots
# =====================================================================

def savefig(fig, figure_dir: Path, stem: str):
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


def plot_decision_trace(
    decision: pd.DataFrame,
    figure_dir: Path,
):
    for label, part in decision.groupby("SimilarityLabel"):
        part = part.sort_values("FromWindow")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            part["FromWindow"],
            part["Error_Current"],
            marker="o",
            label="Error(k)",
        )

        ax.axhline(
            part["Error_Threshold"].iloc[0],
            linewidth=1,
            label="Error threshold",
        )

        ax.set_xlabel("Observation Window End Day")
        ax.set_ylabel("Core3 Mean NRMSE")
        ax.set_title(f"MSOH Decision Trace: {label}")
        ax.legend()

        safe = label.lower().replace(" ", "_")

        savefig(
            fig,
            figure_dir,
            f"fig01_decision_trace_{safe}",
        )


def plot_gain_probability(
    gain_prob: pd.DataFrame,
    alpha: float,
    figure_dir: Path,
):
    for label, part in gain_prob.groupby("SimilarityLabel"):
        part = part.sort_values("ToWindow")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            part["ToWindow"],
            part["P_Gain_GT_Delta"],
            marker="o",
        )

        ax.axhline(alpha, linewidth=1)

        ax.set_xlabel("Next Window End Day")
        ax.set_ylabel("P(Gain > delta)")
        ax.set_ylim(0, 1)
        ax.set_title(f"Probability of Meaningful Future Gain: {label}")

        safe = label.lower().replace(" ", "_")

        savefig(
            fig,
            figure_dir,
            f"fig02_gain_probability_{safe}",
        )


def plot_sensitivity_distribution(
    sensitivity: pd.DataFrame,
    figure_dir: Path,
):
    for label, part in sensitivity.groupby("SimilarityLabel"):
        display = part.copy()

        values = display["Candidate_MSOH"].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))

        if len(values) > 0:
            counts = values.value_counts().sort_index()
            ax.bar(
                counts.index.astype(int),
                counts.values,
            )

        ax.set_xlabel("Candidate MSOH")
        ax.set_ylabel("Number of threshold combinations")
        ax.set_title(f"Threshold Sensitivity: {label}")

        safe = label.lower().replace(" ", "_")

        savefig(
            fig,
            figure_dir,
            f"fig03_threshold_sensitivity_{safe}",
        )


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    if args.bootstrap < 100:
        raise ValueError("正式分析建议 bootstrap >= 1000；默认 5000。")

    table_dir = args.output / "table"
    figure_dir = args.output / "figure"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("06 MSOH 判定开始")
    print("=" * 92)
    print("[1/5] 读取预测数据并构建组级聚合统计量...")

    df = load_predictions(args.input)

    print("[2/5] 进行 Group-level bootstrap，并计算 P(Gain > delta)...")

    error_table, gain_default, store = build_method_uncertainty(
        df=df,
        bootstrap_b=args.bootstrap,
        seed=args.seed,
        gain_delta=args.gain_delta,
    )

    print("[3/5] 计算组级局部稳定性...")

    group_errors = group_window_error_table(df)

    stability = build_stability_table(
        group_errors=group_errors,
        epsilon=args.group_epsilon,
    )

    gain_prob = calculate_gain_probability_for_delta(
        store=store,
        gain_delta=args.gain_delta,
    )

    print("[4/5] 执行默认（探索性）MSOH 判定...")

    decision, summary = decide_msoh(
        error_table=error_table,
        gain_prob_table=gain_prob,
        stability_table=stability,
        error_threshold=args.error_threshold,
        gain_delta=args.gain_delta,
        alpha=args.alpha,
        stability_threshold=args.stability_threshold,
    )

    print("[5/5] 执行阈值敏感性分析...")

    error_grid = parse_float_list(args.error_grid)
    gain_delta_grid = parse_float_list(args.gain_delta_grid)
    alpha_grid = parse_float_list(args.alpha_grid)
    stability_grid = parse_float_list(args.stability_grid)

    sensitivity = threshold_sensitivity(
        error_table=error_table,
        store=store,
        stability_table=stability,
        error_grid=error_grid,
        gain_delta_grid=gain_delta_grid,
        alpha_grid=alpha_grid,
        stability_grid=stability_grid,
    )

    sensitivity_summary = summarize_sensitivity(sensitivity)

    # Save tables
    error_table.to_csv(
        table_dir / "msoh_error_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gain_default.to_csv(
        table_dir / "msoh_gain_bootstrap_default_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stability.to_csv(
        table_dir / "msoh_group_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decision.to_csv(
        table_dir / "msoh_decision_trace.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        table_dir / "msoh_candidate_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity.to_csv(
        table_dir / "msoh_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_summary.to_csv(
        table_dir / "msoh_threshold_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Plots
    plot_decision_trace(
        decision=decision,
        figure_dir=figure_dir,
    )

    plot_gain_probability(
        gain_prob=gain_prob,
        alpha=args.alpha,
        figure_dir=figure_dir,
    )

    plot_sensitivity_distribution(
        sensitivity=sensitivity,
        figure_dir=figure_dir,
    )

    print()
    print("=" * 92)
    print("06 MSOH 判定完成")
    print("=" * 92)

    print("\n当前探索性阈值：")
    print(f"  Error threshold (tau_E)     = {args.error_threshold}")
    print(f"  Meaningful gain (delta_G)   = {args.gain_delta}")
    print(f"  Probability cutoff (alpha)  = {args.alpha}")
    print(f"  Stability threshold (tau_S) = {args.stability_threshold}")
    print(f"  Group stability epsilon     = {args.group_epsilon}")

    print("\n候选 MSOH：")
    print(
        summary[
            [
                "SimilarityLabel",
                "Candidate_MSOH",
                "Decision_Status",
            ]
        ].to_string(index=False)
    )

    print("\n阈值敏感性摘要：")
    print(
        sensitivity_summary.to_string(index=False)
    )

    print()
    print("注意：以上默认阈值是探索性参数，不是最终锁定的科学判据。")
    print("最终判据应结合阈值敏感性、多数据集表现和领域可接受误差后预先固定。")
    print("=" * 92)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] 06 MSOH 判定失败：{exc}", file=sys.stderr)
        raise
