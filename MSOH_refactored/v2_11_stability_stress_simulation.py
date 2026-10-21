from __future__ import annotations

"""
v2_11_stability_stress_simulation.py

MSOH V2 — stability-gate stress test.

目标
----
专门检验当前 Local Saturation 中的 group-stability gate：

    P(Gain > delta_G) < alpha
    AND
    Stable_Group_Proportion >= tau_S

在“总体已经饱和，但组级误差仍有波动”的情况下，
tau_S = 0.75 是否会导致过多 Not Identified。

本脚本固定 true MSOH = D4，系统改变：
1) post-MSOH residual fluctuation
2) between-group heterogeneity
3) N groups
4) tau_S

注意
----
- 正式阈值仍冻结：
    tau_E = 0.40
    delta_G = 0.03
    alpha = 0.10
    group_epsilon = 0.03
- tau_S=0.75 为 primary
- tau_S=0.50 / 0.25 仅为预先声明的 sensitivity
- 本脚本不能用于按真实数据结果反向挑 tau_S
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2_simstudy" / "stability_stress"

WINDOWS = np.array([2, 3, 4, 5, 6], dtype=int)
TRUE_MSOH = 4

TAU_E = 0.40
DELTA_G = 0.03
ALPHA = 0.10
GROUP_EPSILON = 0.03

TAU_S_GRID = [0.75, 0.50, 0.25]
N_GROUPS_GRID = [12, 24, 48]

POST_NOISE_GRID = [0.008, 0.02, 0.04, 0.06, 0.10]
GROUP_HETEROGENEITY_GRID = [0.10, 0.20, 0.30, 0.40]

N_REPLICATES = 500
BOOTSTRAP_B = 1000
BASE_SEED = 20260929

# population-level target error curve:
# before D4: informative gain
# D4 onward: plateau around 0.30
TARGET_ERROR = np.array(
    [0.78, 0.55, 0.31, 0.30, 0.30],
    dtype=float,
)

PRE_NOISE_SD = 0.05


def generate_group_errors(
    rng: np.random.Generator,
    n_groups: int,
    group_heterogeneity: float,
    post_noise_sd: float,
) -> np.ndarray:
    """
    shape = [group, window]

    - between-group heterogeneity controls persistent difficulty differences;
    - post_noise_sd controls residual window-to-window fluctuation after D4;
    - population mean still follows the same saturated D4 target curve.
    """
    multiplier = rng.lognormal(
        mean=-0.5 * group_heterogeneity**2,
        sigma=group_heterogeneity,
        size=n_groups,
    )

    base = TARGET_ERROR[None, :] * multiplier[:, None]

    noise = np.zeros_like(base)

    for wi, w in enumerate(WINDOWS):
        sd = (
            POST_NOISE_GRID[0]
            if False
            else (
                post_noise_sd
                if w >= TRUE_MSOH
                else PRE_NOISE_SD
            )
        )

        noise[:, wi] = rng.normal(
            0.0,
            sd,
            size=n_groups,
        )

    return np.clip(
        base + noise,
        1e-6,
        None,
    )


def paired_bootstrap_p_gain_gt_delta(
    rng: np.random.Generator,
    group_errors: np.ndarray,
    from_idx: int,
    to_idx: int,
) -> float:
    n_groups = group_errors.shape[0]

    idx = rng.integers(
        0,
        n_groups,
        size=(BOOTSTRAP_B, n_groups),
    )

    e_from = group_errors[
        idx,
        from_idx,
    ].mean(
        axis=1
    )

    e_to = group_errors[
        idx,
        to_idx,
    ].mean(
        axis=1
    )

    gain = (
        e_from
        - e_to
    )

    return float(
        np.mean(
            gain > DELTA_G
        )
    )


def evaluate_one(
    rng: np.random.Generator,
    group_errors: np.ndarray,
    tau_s: float,
) -> dict:
    point_error = group_errors.mean(
        axis=0
    )

    p_gain = []
    stability = []
    local_sat = []

    for ti in range(
        len(WINDOWS) - 1
    ):
        p = paired_bootstrap_p_gain_gt_delta(
            rng,
            group_errors,
            ti,
            ti + 1,
        )

        stab = float(
            np.mean(
                np.abs(
                    group_errors[:, ti]
                    - group_errors[:, ti + 1]
                )
                <= GROUP_EPSILON
            )
        )

        local = (
            p < ALPHA
            and stab >= tau_s
        )

        p_gain.append(p)
        stability.append(stab)
        local_sat.append(local)

    persistent = np.zeros(
        len(WINDOWS),
        dtype=bool,
    )

    # final window cannot establish saturation
    for wi in range(
        len(WINDOWS) - 1
    ):
        persistent[wi] = bool(
            all(
                local_sat[wi:]
            )
        )

    adequacy = (
        point_error <= TAU_E
    )

    estimated = None

    for wi, w in enumerate(
        WINDOWS[:-1]
    ):
        if (
            adequacy[wi]
            and persistent[wi]
        ):
            estimated = int(w)
            break

    return {
        "Estimated_MSOH": estimated,
        "Point_Error": point_error,
        "P_Gain": np.asarray(
            p_gain,
            dtype=float,
        ),
        "Stability": np.asarray(
            stability,
            dtype=float,
        ),
        "Local_Saturation": np.asarray(
            local_sat,
            dtype=bool,
        ),
        "Persistent": persistent,
        "Adequacy": adequacy,
    }


def recovery_class(
    estimated: int | None,
) -> str:
    if estimated is None:
        return "not_identified"

    if estimated == TRUE_MSOH:
        return "exact"

    if estimated < TRUE_MSOH:
        return "early"

    return "late"


def run():
    rows = []
    transition_rows = []

    n_cells = (
        len(N_GROUPS_GRID)
        * len(TAU_S_GRID)
        * len(POST_NOISE_GRID)
        * len(GROUP_HETEROGENEITY_GRID)
    )

    seed_seq = np.random.SeedSequence(
        BASE_SEED
    )

    child_seeds = seed_seq.spawn(
        n_cells
    )

    seed_i = 0

    for n_groups in N_GROUPS_GRID:
        for tau_s in TAU_S_GRID:
            for post_noise in POST_NOISE_GRID:
                for group_het in GROUP_HETEROGENEITY_GRID:

                    rng = np.random.default_rng(
                        child_seeds[seed_i]
                    )
                    seed_i += 1

                    for rep in range(
                        N_REPLICATES
                    ):
                        gerr = generate_group_errors(
                            rng=rng,
                            n_groups=n_groups,
                            group_heterogeneity=group_het,
                            post_noise_sd=post_noise,
                        )

                        result = evaluate_one(
                            rng=rng,
                            group_errors=gerr,
                            tau_s=tau_s,
                        )

                        estimated = result[
                            "Estimated_MSOH"
                        ]

                        rows.append(
                            {
                                "True_MSOH": TRUE_MSOH,
                                "N_Groups": n_groups,
                                "Tau_S": tau_s,
                                "Post_Noise_SD": post_noise,
                                "Group_Heterogeneity": group_het,
                                "Replicate": rep,
                                "Estimated_MSOH": estimated,
                                "Recovery_Class": recovery_class(
                                    estimated
                                ),
                            }
                        )

                        # retain transition diagnostics for a sample of replicates
                        if rep < 20:
                            for ti in range(
                                len(WINDOWS) - 1
                            ):
                                transition_rows.append(
                                    {
                                        "N_Groups": n_groups,
                                        "Tau_S": tau_s,
                                        "Post_Noise_SD": post_noise,
                                        "Group_Heterogeneity": group_het,
                                        "Replicate": rep,
                                        "FromWindow": int(
                                            WINDOWS[ti]
                                        ),
                                        "ToWindow": int(
                                            WINDOWS[ti + 1]
                                        ),
                                        "P_Gain_GT_Delta": float(
                                            result[
                                                "P_Gain"
                                            ][ti]
                                        ),
                                        "Stable_Group_Proportion": float(
                                            result[
                                                "Stability"
                                            ][ti]
                                        ),
                                        "Local_Saturation": bool(
                                            result[
                                                "Local_Saturation"
                                            ][ti]
                                        ),
                                        "Estimated_MSOH": estimated,
                                    }
                                )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            transition_rows
        ),
    )


def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "N_Groups",
        "Tau_S",
        "Post_Noise_SD",
        "Group_Heterogeneity",
    ]

    for keys, part in results.groupby(
        group_cols,
        sort=True,
    ):
        (
            n_groups,
            tau_s,
            post_noise,
            group_het,
        ) = keys

        rows.append(
            {
                "True_MSOH": TRUE_MSOH,
                "N_Groups": int(
                    n_groups
                ),
                "Tau_S": float(
                    tau_s
                ),
                "Post_Noise_SD": float(
                    post_noise
                ),
                "Group_Heterogeneity": float(
                    group_het
                ),
                "N_Replicates": int(
                    len(part)
                ),
                "Exact_Rate": float(
                    (
                        part[
                            "Recovery_Class"
                        ]
                        == "exact"
                    ).mean()
                ),
                "Not_Identified_Rate": float(
                    (
                        part[
                            "Recovery_Class"
                        ]
                        == "not_identified"
                    ).mean()
                ),
                "Early_Rate": float(
                    (
                        part[
                            "Recovery_Class"
                        ]
                        == "early"
                    ).mean()
                ),
                "Late_Rate": float(
                    (
                        part[
                            "Recovery_Class"
                        ]
                        == "late"
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_threshold_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate across the stress grid to show how each tau_S behaves,
    without selecting a winner.
    """
    rows = []

    for (
        n_groups,
        tau_s,
    ), part in summary.groupby(
        [
            "N_Groups",
            "Tau_S",
        ],
        sort=True,
    ):
        rows.append(
            {
                "N_Groups": int(
                    n_groups
                ),
                "Tau_S": float(
                    tau_s
                ),
                "Mean_Exact_Rate": float(
                    part[
                        "Exact_Rate"
                    ].mean()
                ),
                "Median_Exact_Rate": float(
                    part[
                        "Exact_Rate"
                    ].median()
                ),
                "Mean_Not_Identified_Rate": float(
                    part[
                        "Not_Identified_Rate"
                    ].mean()
                ),
                "Worst_Not_Identified_Rate": float(
                    part[
                        "Not_Identified_Rate"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 118
    )
    print(
        "MSOH V2 stability-gate stress simulation"
    )
    print(
        "=" * 118
    )
    print(
        f"true MSOH=D{TRUE_MSOH}; "
        f"replicates/cell={N_REPLICATES}; "
        f"bootstrap B={BOOTSTRAP_B}; "
        f"seed={BASE_SEED}"
    )
    print(
        f"Frozen: tau_E={TAU_E}, "
        f"delta_G={DELTA_G}, "
        f"alpha={ALPHA}, "
        f"group_epsilon={GROUP_EPSILON}"
    )
    print(
        f"N groups: {N_GROUPS_GRID}"
    )
    print(
        f"tau_S: {TAU_S_GRID} "
        "(0.75 primary; 0.50/0.25 sensitivity)"
    )
    print(
        f"post-MSOH noise: "
        f"{POST_NOISE_GRID}"
    )
    print(
        f"group heterogeneity: "
        f"{GROUP_HETEROGENEITY_GRID}"
    )
    print()

    results, transitions = run()
    summary = summarize(
        results
    )
    threshold_summary = (
        build_threshold_summary(
            summary
        )
    )

    results.to_csv(
        OUT
        / "replicate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT
        / "stress_grid_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_summary.to_csv(
        OUT
        / "tau_s_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transitions.to_csv(
        OUT
        / "transition_diagnostics_first20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = summary[
        (
            summary[
                "N_Groups"
            ]
            == 12
        )
        & (
            summary[
                "Tau_S"
            ]
            == 0.75
        )
    ].copy()

    primary = primary.sort_values(
        [
            "Post_Noise_SD",
            "Group_Heterogeneity",
        ]
    )

    print(
        "Primary stress grid: "
        "N=12, tau_S=0.75"
    )

    print(
        primary[
            [
                "Post_Noise_SD",
                "Group_Heterogeneity",
                "Exact_Rate",
                "Not_Identified_Rate",
                "Early_Rate",
                "Late_Rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nTau_S aggregate operating characteristics:"
    )

    print(
        threshold_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nInterpretation rule:"
    )
    print(
        "If P(Gain>delta) already indicates population-level saturation "
        "but Not_Identified_Rate rises sharply as post-MSOH group fluctuation "
        "increases, then the hard stability gate is the source of non-identification."
    )
    print(
        "Do not choose a new tau_S from the real-data result. "
        "Use this stress study only to characterize the gate's operating behavior."
    )

    print(
        f"\nOutput: {OUT}"
    )
    print(
        "=" * 118
    )


if __name__ == "__main__":
    main()
