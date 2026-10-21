import argparse
from pathlib import Path
import pandas as pd

from config import (
    ANALYSIS_DIR,
    MSOH_DIR,
    BOOTSTRAP_B,
    RANDOM_SEED,
    ERROR_THRESHOLD,
    GAIN_DELTA,
    ALPHA,
    STABILITY_THRESHOLD,
    GROUP_EPSILON,
)
from src.common import similarity_label, is_prediction_probe, ensure_dirs, save_csv
from src.bootstrap import (
    legacy06_method_bootstrap,
    legacy06_error_gain_tables,
    group_errors,
    stability_table,
)
from src.msoh import decide_msoh, threshold_sensitivity, summarize_sensitivity
from src.plots import msoh_plots


def flist(x):
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ANALYSIS_DIR / "table/combined_predictions.csv")
    p.add_argument("--output", type=Path, default=MSOH_DIR)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--error-threshold", type=float, default=ERROR_THRESHOLD)
    p.add_argument("--gain-delta", type=float, default=GAIN_DELTA)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--stability-threshold", type=float, default=STABILITY_THRESHOLD)
    p.add_argument("--group-epsilon", type=float, default=GROUP_EPSILON)
    p.add_argument("--error-grid", default="0.35,0.40,0.45,0.50")
    p.add_argument("--gain-delta-grid", default="0.01,0.02,0.03,0.05")
    p.add_argument("--alpha-grid", default="0.05,0.10,0.20")
    p.add_argument("--stability-grid", default="0.50,0.67,0.75,0.80")
    a = p.parse_args()

    pred = pd.read_csv(a.input)
    pred = pred[
        pred["Model"].map(is_prediction_probe)
        & pred["Signal"].isin(["Weight", "Status_Score", "Feeding"])
    ].copy()
    pred["Group"] = pred["Group"].astype(str)
    pred["SimilarityLabel"] = pred["Model"].map(similarity_label)

    # Exact original 06 RNG semantics: method-distinct multinomial streams
    # seed + method_index*1009, with the same resample shared across windows.
    store = legacy06_method_bootstrap(pred, a.bootstrap, a.seed)
    error, gain = legacy06_error_gain_tables(store, a.gain_delta)

    temp = pred.copy()
    temp["Entity"] = temp["SimilarityLabel"]
    ge = group_errors(temp, "Entity")
    stable = stability_table(ge, "Entity", a.group_epsilon).rename(columns={"Entity": "SimilarityLabel"})

    # Generic MSOH functions use a single entity key.
    err = error.rename(columns={"SimilarityLabel": "Entity"})
    gn = gain.rename(columns={"SimilarityLabel": "Entity"})
    st = stable.rename(columns={"SimilarityLabel": "Entity"})
    trace, summary = decide_msoh(
        err,
        gn,
        st,
        "Entity",
        a.error_threshold,
        a.gain_delta,
        a.alpha,
        a.stability_threshold,
    )
    trace = trace.rename(columns={"Entity": "SimilarityLabel"})
    summary = summary.rename(columns={"Entity": "SimilarityLabel"})

    sens = threshold_sensitivity(
        err,
        store,
        st,
        "Entity",
        flist(a.error_grid),
        flist(a.gain_delta_grid),
        flist(a.alpha_grid),
        flist(a.stability_grid),
    ).rename(columns={"Entity": "SimilarityLabel"})
    sens_summary = summarize_sensitivity(
        sens.rename(columns={"SimilarityLabel": "Entity"}), "Entity"
    ).rename(columns={"Entity": "SimilarityLabel"})

    table, fig = ensure_dirs(a.output)
    files = {
        "msoh_error_bootstrap": error,
        "msoh_gain_bootstrap_default_delta": gain,
        "msoh_group_stability": stable,
        "msoh_decision_trace": trace,
        "msoh_candidate_table": summary,
        "msoh_threshold_sensitivity": sens,
        "msoh_threshold_sensitivity_summary": sens_summary,
    }
    for n, d in files.items():
        save_csv(d, table / f"{n}.csv")

    msoh_plots(error, gain, summary, fig, a.alpha)
    print("=" * 92)
    print("06 MSOH 判定完成（Adequacy / Persistent Saturation 已拆分）")
    print(summary[["SimilarityLabel", "Candidate_MSOH", "Adequacy_Onset", "Persistent_Saturation_Onset", "Decision_Status"]].to_string(index=False))
    print("\n阈值敏感性摘要：")
    print(sens_summary.to_string(index=False))
    print("=" * 92)


if __name__ == "__main__":
    main()
