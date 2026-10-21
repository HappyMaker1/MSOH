"""
09_signal_specific_msoh.py

Signal-specific MSOH diagnostic.

Purpose
-------
Evaluate Weight, Status_Score and Feeding separately instead of averaging them
into one Core3 error. For every prediction model already present in
results/04_output/table/combined_predictions.csv, this script computes:

1) signal-specific NRMSE Error_s(k)
2) paired group-bootstrap Gain_s(k->k+1)
3) signal-specific group stability
4) predictive adequacy
5) local saturation
6) persistent saturation
7) signal-specific candidate MSOH

Important
---------
- This is a diagnostic / sensitivity analysis. It does NOT automatically select
  the best model for each signal, because choosing the best model after seeing
  the same LOOG results would create selection optimism.
- Independent bootstrap unit = Group.
- Same group bootstrap draw is reused across all windows within a model/signal,
  preserving pairing for Gain.
- Default thresholds match the current MSOH framework:
      tau_E = 0.40
      delta_G = 0.03
      alpha = 0.10
      tau_S = 0.75
      group epsilon = 0.03
- Persistent saturation requires every evaluable transition from candidate k
  onward to satisfy the local saturation rule. The final window cannot itself
  establish saturation because there is no k->k+1 transition after it.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/04_output/table/combined_predictions.csv")
DEFAULT_OUTPUT = Path("results/09_signal_specific_msoh")

CORE_SIGNALS = ["Weight", "Status_Score", "Feeding"]

DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260924
DEFAULT_ERROR_THRESHOLD = 0.40
DEFAULT_GAIN_DELTA = 0.03
DEFAULT_ALPHA = 0.10
DEFAULT_STABILITY_THRESHOLD = 0.75
DEFAULT_GROUP_EPSILON = 0.03


def parse_args():
    p = argparse.ArgumentParser(
        description="09: Signal-specific adequacy / persistent saturation / MSOH"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--error-threshold", type=float, default=DEFAULT_ERROR_THRESHOLD)
    p.add_argument("--gain-delta", type=float, default=DEFAULT_GAIN_DELTA)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument(
        "--stability-threshold", type=float, default=DEFAULT_STABILITY_THRESHOLD
    )
    p.add_argument("--group-epsilon", type=float, default=DEFAULT_GROUP_EPSILON)
    return p.parse_args()


def similarity_label(model: str) -> str:
    m = str(model)
    mapping = {
        "Similarity_DTW_All4_K2": "DTW All4",
        "Similarity_DTW_Core3_K2": "DTW Core3",
        "Similarity_EuclideanFeature_K2": "Euclidean Feature",
        "Probe_MixedEffects_BLUP": "Mixed Effects",
        "Probe_Ridge_Direct": "Ridge Direct",
    }
    return mapping.get(m, m)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)
    required = {"Group", "Window", "Model", "Signal", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"输入缺少必要字段：{missing}")

    df = df[df["Signal"].isin(CORE_SIGNALS)].copy()
    df = df[df["y_true"].notna() & df["y_pred"].notna()].copy()
    df["Group"] = df["Group"].astype(str)
    df["Window"] = df["Window"].astype(int)
    df["Model"] = df["Model"].astype(str)
    return df


def nrmse_from_rows(part: pd.DataFrame) -> float:
    if part.empty:
        return np.nan
    y = part["y_true"].to_numpy(float)
    p = part["y_pred"].to_numpy(float)
    if len(y) < 2:
        return np.nan
    sd = float(np.std(y, ddof=1))
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        return np.nan
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return rmse / sd


def build_group_stats(part: pd.DataFrame):
    """Return groups, windows, and per-window group sufficient statistics.

    For a fixed model + signal, each group contributes multiple target-day rows.
    Bootstrap resamples groups and recomputes pooled NRMSE exactly from weighted
    SSE plus weighted y moments.
    """
    groups = sorted(part["Group"].unique().tolist())
    windows = sorted(part["Window"].unique().tolist())
    gi = {g: i for i, g in enumerate(groups)}

    stats = {}
    for w in windows:
        subw = part[part["Window"] == w]
        n = np.zeros(len(groups), float)
        sy = np.zeros(len(groups), float)
        sy2 = np.zeros(len(groups), float)
        sse = np.zeros(len(groups), float)

        for g, sg in subw.groupby("Group"):
            idx = gi[g]
            y = sg["y_true"].to_numpy(float)
            p = sg["y_pred"].to_numpy(float)
            n[idx] = len(y)
            sy[idx] = np.sum(y)
            sy2[idx] = np.sum(y * y)
            sse[idx] = np.sum((p - y) ** 2)

        stats[w] = {"n": n, "sy": sy, "sy2": sy2, "sse": sse}

    return groups, windows, stats


def nrmse_from_counts(counts: np.ndarray, stat: dict) -> float:
    n = float(np.sum(counts * stat["n"]))
    if n <= 1:
        return np.nan
    sy = float(np.sum(counts * stat["sy"]))
    sy2 = float(np.sum(counts * stat["sy2"]))
    sse = float(np.sum(counts * stat["sse"]))

    var_num = sy2 - (sy * sy / n)
    if var_num <= 0:
        return np.nan
    sd = np.sqrt(var_num / (n - 1.0))
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        return np.nan
    rmse = np.sqrt(sse / n)
    return float(rmse / sd)


def bootstrap_model_signal(
    part: pd.DataFrame,
    b: int,
    seed: int,
):
    groups, windows, stats = build_group_stats(part)
    ng = len(groups)
    if ng < 2:
        raise ValueError("Group 数量不足，无法 bootstrap。")

    rng = np.random.default_rng(seed)
    counts_mat = rng.multinomial(ng, np.repeat(1.0 / ng, ng), size=b)

    full_counts = np.ones(ng, float)
    point = np.array(
        [nrmse_from_counts(full_counts, stats[w]) for w in windows], float
    )

    boot = np.empty((b, len(windows)), float)
    for bi in range(b):
        c = counts_mat[bi]
        for wi, w in enumerate(windows):
            boot[bi, wi] = nrmse_from_counts(c, stats[w])

    gain_boot = boot[:, :-1] - boot[:, 1:]
    return groups, windows, point, boot, gain_boot


def group_error_table(part: pd.DataFrame) -> pd.DataFrame:
    """Group-level signal NRMSE-like error for local stability.

    The denominator is the full-sample SD of y_true for the given signal/window,
    matching the scale used by the signal-level NRMSE. Stability is descriptive
    and compares absolute changes in group-level normalized RMSE.
    """
    rows = []
    for w in sorted(part["Window"].unique()):
        sw = part[part["Window"] == w]
        y_all = sw["y_true"].to_numpy(float)
        if len(y_all) < 2:
            continue
        sd = float(np.std(y_all, ddof=1))
        if not np.isfinite(sd) or np.isclose(sd, 0.0):
            continue

        for g, sg in sw.groupby("Group"):
            y = sg["y_true"].to_numpy(float)
            p = sg["y_pred"].to_numpy(float)
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            rows.append(
                {
                    "Group": g,
                    "Window": int(w),
                    "Group_Error": rmse / sd,
                }
            )
    return pd.DataFrame(rows)


def stability_table(part: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    ge = group_error_table(part)
    windows = sorted(ge["Window"].unique().tolist())
    rows = []
    for i in range(len(windows) - 1):
        a, b = windows[i], windows[i + 1]
        left = ge[ge["Window"] == a][["Group", "Group_Error"]].rename(
            columns={"Group_Error": "E_from"}
        )
        right = ge[ge["Window"] == b][["Group", "Group_Error"]].rename(
            columns={"Group_Error": "E_to"}
        )
        m = left.merge(right, on="Group", how="inner")
        if m.empty:
            prop = np.nan
            n = 0
        else:
            change = np.abs(m["E_to"] - m["E_from"])
            prop = float(np.mean(change <= epsilon))
            n = len(m)
        rows.append(
            {
                "FromWindow": int(a),
                "ToWindow": int(b),
                "Stable_Group_Proportion": prop,
                "N_Groups_Compared": int(n),
                "Group_Epsilon": float(epsilon),
            }
        )
    return pd.DataFrame(rows)


def evaluate_one(
    model: str,
    signal: str,
    part: pd.DataFrame,
    b: int,
    seed: int,
    tau_e: float,
    delta_g: float,
    alpha: float,
    tau_s: float,
    epsilon: float,
):
    groups, windows, point, boot, gain_boot = bootstrap_model_signal(part, b, seed)
    stability = stability_table(part, epsilon)
    stab_lookup = {
        (int(r.FromWindow), int(r.ToWindow)): float(r.Stable_Group_Proportion)
        for r in stability.itertuples()
    }

    error_rows = []
    gain_rows = []

    for wi, w in enumerate(windows):
        vals = boot[:, wi]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        adequate = bool(point[wi] <= tau_e)
        error_rows.append(
            {
                "SimilarityLabel": similarity_label(model),
                "Model": model,
                "Signal": signal,
                "Window": int(w),
                "Error_Point": float(point[wi]),
                "Error_CI_Low": float(lo),
                "Error_CI_High": float(hi),
                "Adequate": adequate,
                "Error_Threshold": float(tau_e),
                "N_Groups": len(groups),
                "Bootstrap_B": int(b),
            }
        )

    transition_local = {}
    for gi in range(len(windows) - 1):
        a, c = int(windows[gi]), int(windows[gi + 1])
        vals = gain_boot[:, gi]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        p_gt_delta = float(np.mean(vals > delta_g))
        stable_prop = stab_lookup.get((a, c), np.nan)
        local_sat = bool(
            np.isfinite(stable_prop)
            and p_gt_delta < alpha
            and stable_prop >= tau_s
        )
        transition_local[(a, c)] = local_sat
        gain_rows.append(
            {
                "SimilarityLabel": similarity_label(model),
                "Model": model,
                "Signal": signal,
                "FromWindow": a,
                "ToWindow": c,
                "Gain_Point": float(point[gi] - point[gi + 1]),
                "Gain_CI_Low": float(lo),
                "Gain_CI_High": float(hi),
                "P_Gain_GT_0": float(np.mean(vals > 0.0)),
                "Gain_Delta": float(delta_g),
                "P_Gain_GT_Delta": p_gt_delta,
                "Stable_Group_Proportion": stable_prop,
                "Stability_Threshold": float(tau_s),
                "Local_Saturation": local_sat,
            }
        )

    # Persistent saturation can only start at a window that has a next window.
    persistent = {}
    for i, w in enumerate(windows[:-1]):
        required = [
            transition_local[(int(windows[j]), int(windows[j + 1]))]
            for j in range(i, len(windows) - 1)
        ]
        persistent[int(w)] = bool(all(required))

    adequacy_windows = [
        int(windows[i]) for i in range(len(windows)) if point[i] <= tau_e
    ]
    adequacy_onset = min(adequacy_windows) if adequacy_windows else np.nan

    ps_windows = [w for w, ok in persistent.items() if ok]
    ps_onset = min(ps_windows) if ps_windows else np.nan

    candidate_windows = []
    for i, w in enumerate(windows[:-1]):
        w = int(w)
        if point[i] <= tau_e and persistent.get(w, False):
            candidate_windows.append(w)
    candidate_msoh = min(candidate_windows) if candidate_windows else np.nan

    if np.isfinite(candidate_msoh):
        status = "Signal-specific MSOH identified"
    elif np.isfinite(adequacy_onset) and np.isfinite(ps_onset):
        status = "Adequacy and persistent saturation both observed, but not at a common eligible window"
    elif np.isfinite(adequacy_onset):
        status = "Predictive adequacy reached; persistent saturation not established"
    elif np.isfinite(ps_onset):
        status = "Persistent saturation established; predictive adequacy not reached"
    else:
        status = "Neither predictive adequacy nor persistent saturation established"

    summary = {
        "SimilarityLabel": similarity_label(model),
        "Model": model,
        "Signal": signal,
        "Candidate_MSOH": candidate_msoh,
        "Adequacy_Onset": adequacy_onset,
        "Persistent_Saturation_Onset": ps_onset,
        "Decision_Status": status,
        "Best_Error": float(np.nanmin(point)),
        "Best_Error_Window": int(windows[int(np.nanargmin(point))]),
        "Last_Window": int(windows[-1]),
        "Last_Window_Error": float(point[-1]),
    }

    return pd.DataFrame(error_rows), pd.DataFrame(gain_rows), pd.DataFrame([summary])


def build_signal_overview(summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive overview only; does not choose a final model."""
    rows = []
    for signal, part in summary.groupby("Signal"):
        last_best = part.loc[part["Last_Window_Error"].idxmin()]
        n_msoh = int(part["Candidate_MSOH"].notna().sum())
        n_adequate = int(part["Adequacy_Onset"].notna().sum())
        n_sat = int(part["Persistent_Saturation_Onset"].notna().sum())
        rows.append(
            {
                "Signal": signal,
                "N_Models": int(len(part)),
                "N_Models_With_MSOH": n_msoh,
                "N_Models_Adequate": n_adequate,
                "N_Models_Persistently_Saturated": n_sat,
                "Lowest_Last_Window_Error_Model": last_best["SimilarityLabel"],
                "Lowest_Last_Window_Error": float(last_best["Last_Window_Error"]),
                "Note": "Descriptive only; do not select final predictor from this table without a pre-specified/nested selection rule.",
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.bootstrap < 200:
        raise ValueError("bootstrap 建议至少 200；诊断默认 2000。")

    df = load_predictions(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    table_dir = args.output / "table"
    table_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["Model"].unique().tolist())

    all_error = []
    all_gain = []
    all_summary = []

    # Deterministic, distinct seed per model/signal while preserving paired
    # resampling across windows inside each model/signal.
    counter = 0
    for model in models:
        for signal in CORE_SIGNALS:
            part = df[(df["Model"] == model) & (df["Signal"] == signal)].copy()
            if part.empty:
                continue
            seed = args.seed + counter * 1009
            counter += 1
            e, g, s = evaluate_one(
                model=model,
                signal=signal,
                part=part,
                b=args.bootstrap,
                seed=seed,
                tau_e=args.error_threshold,
                delta_g=args.gain_delta,
                alpha=args.alpha,
                tau_s=args.stability_threshold,
                epsilon=args.group_epsilon,
            )
            all_error.append(e)
            all_gain.append(g)
            all_summary.append(s)

    error = pd.concat(all_error, ignore_index=True)
    gain = pd.concat(all_gain, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    overview = build_signal_overview(summary)

    error.to_csv(table_dir / "signal_specific_error.csv", index=False)
    gain.to_csv(table_dir / "signal_specific_gain_saturation.csv", index=False)
    summary.to_csv(table_dir / "signal_specific_msoh_summary.csv", index=False)
    overview.to_csv(table_dir / "signal_overview.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("=" * 110)
    print(
        f"09 Signal-specific MSOH 完成 | B={args.bootstrap} | "
        f"tau_E={args.error_threshold} | delta_G={args.gain_delta} | "
        f"alpha={args.alpha} | tau_S={args.stability_threshold}"
    )
    print("\n各模型 × 各信号判定：")
    print(
        summary[
            [
                "SimilarityLabel",
                "Signal",
                "Candidate_MSOH",
                "Adequacy_Onset",
                "Persistent_Saturation_Onset",
                "Best_Error",
                "Best_Error_Window",
                "Last_Window_Error",
                "Decision_Status",
            ]
        ].to_string(index=False)
    )

    print("\n按信号概览（仅描述，不自动选模型）：")
    print(overview.to_string(index=False))

    print("\n输出：")
    print(f"  {table_dir / 'signal_specific_error.csv'}")
    print(f"  {table_dir / 'signal_specific_gain_saturation.csv'}")
    print(f"  {table_dir / 'signal_specific_msoh_summary.csv'}")
    print(f"  {table_dir / 'signal_overview.csv'}")
    print("=" * 110)


if __name__ == "__main__":
    main()
