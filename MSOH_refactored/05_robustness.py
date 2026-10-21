import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import ANALYSIS_DIR, ROBUSTNESS_DIR, BOOTSTRAP_B, RANDOM_SEED
from src.common import is_similarity_model, ensure_dirs, save_csv
from src.bootstrap import (
    legacy05_bootstrap_error_ci,
    legacy05_bootstrap_gain_ci,
    legacy05_paired_method_difference,
    group_errors,
    stability_table,
)
from src.plots import robustness_plots

# Original 05 default; deliberately separate from 06 GROUP_EPSILON=0.03.
DEFAULT_CHANGE_TOL = 0.01


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ANALYSIS_DIR / "table/combined_predictions.csv")
    p.add_argument("--output", type=Path, default=ROBUSTNESS_DIR)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--change-tol", type=float, default=DEFAULT_CHANGE_TOL)
    a = p.parse_args()

    pred = pd.read_csv(a.input)
    sim = pred[
        pred["Model"].map(is_similarity_model)
        & pred["Signal"].isin(["Weight", "Status_Score", "Feeding"])
    ].copy()
    sim["Group"] = sim["Group"].astype(str)

    # Exact original 05 RNG semantics: one RNG stream consumed sequentially by
    # Error bootstrap -> Gain bootstrap -> paired method differences.
    rng = np.random.default_rng(a.seed)
    error = legacy05_bootstrap_error_ci(sim, a.bootstrap, rng)
    gain = legacy05_bootstrap_gain_ci(sim, a.bootstrap, rng)
    paired = legacy05_paired_method_difference(sim, a.bootstrap, rng)

    # Keep refactored shared stability machinery for descriptive group changes.
    temp = sim.copy()
    temp["Entity"] = temp["Model"].map({
        "Similarity_EuclideanFeature_K2": "Euclidean Feature",
        "Similarity_DTW_Core3_K2": "DTW Core3",
        "Similarity_DTW_All4_K2": "DTW All4",
    }).fillna(temp["Model"])
    ge = group_errors(temp, "Entity")
    changes = stability_table(ge, "Entity", a.change_tol).rename(columns={"Entity": "SimilarityLabel"})
    summary = changes.copy()

    table, fig = ensure_dirs(a.output)
    save_csv(error, table / "bootstrap_error_ci.csv")
    save_csv(gain, table / "bootstrap_gain_ci.csv")
    save_csv(paired, table / "paired_method_difference.csv")
    save_csv(changes, table / "group_window_changes.csv")
    save_csv(summary, table / "robustness_summary.csv")
    robustness_plots(error, gain, paired, summary, fig)

    print("=" * 92)
    print("05 稳健性分析完成（原版 RNG 兼容；Group bootstrap）")
    print(error[["SimilarityLabel", "Window", "Error_Point", "Error_CI_Low", "Error_CI_High"]].to_string(index=False))
    print("\nGain:")
    print(gain[["SimilarityLabel", "FromWindow", "ToWindow", "Gain_Point", "Gain_CI_Low", "Gain_CI_High", "P_Gain_GT_0"]].to_string(index=False))
    print("=" * 92)


if __name__ == "__main__":
    main()
