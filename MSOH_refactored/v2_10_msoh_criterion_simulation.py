from __future__ import annotations

"""
v2_10_msoh_criterion_simulation.py

MSOH V2 — criterion-level Monte Carlo simulation study.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2_simstudy" / "criterion_calibration"

WINDOWS = np.array([2, 3, 4, 5, 6], dtype=int)

TAU_E = 0.40
DELTA_G = 0.03
ALPHA = 0.10
GROUP_EPSILON = 0.03

TAU_S_GRID = [0.75, 0.50, 0.25]
N_GROUPS_GRID = [12, 24, 48]

N_REPLICATES = 1000
BOOTSTRAP_B = 1000
BASE_SEED = 20260929


@dataclass(frozen=True)
class Scenario:
    name: str
    true_msoh: int | None
    error_curve: tuple[float, float, float, float, float]
    post_noise_sd: float
    pre_noise_sd: float


SCENARIOS = [
    Scenario("true_D3", 3, (0.66, 0.32, 0.31, 0.30, 0.30), 0.008, 0.050),
    Scenario("true_D4", 4, (0.78, 0.55, 0.31, 0.30, 0.30), 0.008, 0.050),
    Scenario("true_D5", 5, (0.82, 0.66, 0.52, 0.31, 0.30), 0.008, 0.050),
    Scenario("no_msoh", None, (0.86, 0.73, 0.63, 0.55, 0.49), 0.030, 0.050),
]


def generate_group_errors(rng, scenario, n_groups):
    target = np.asarray(scenario.error_curve, dtype=float)
    multiplier = rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15, size=n_groups)
    base = target[None, :] * multiplier[:, None]
    noise = np.zeros_like(base)

    for wi, w in enumerate(WINDOWS):
        if scenario.true_msoh is not None and w >= scenario.true_msoh:
            sd = scenario.post_noise_sd
        else:
            sd = scenario.pre_noise_sd
        noise[:, wi] = rng.normal(0.0, sd, size=n_groups)

    return np.clip(base + noise, 1e-6, None)


def bootstrap_gain_probability(rng, group_errors, from_idx, to_idx, B):
    n_groups = group_errors.shape[0]
    idx = rng.integers(0, n_groups, size=(B, n_groups))
    a = group_errors[idx, from_idx].mean(axis=1)
    b = group_errors[idx, to_idx].mean(axis=1)
    gain = a - b
    return float(np.mean(gain > DELTA_G))


def decision_for_one_replicate(rng, group_errors, tau_s):
    point_error = group_errors.mean(axis=0)
    p_gain, stable, local = [], [], []

    for i in range(len(WINDOWS) - 1):
        p = bootstrap_gain_probability(rng, group_errors, i, i + 1, BOOTSTRAP_B)
        s = float(np.mean(np.abs(group_errors[:, i] - group_errors[:, i + 1]) <= GROUP_EPSILON))
        ls = p < ALPHA and s >= tau_s
        p_gain.append(p)
        stable.append(s)
        local.append(ls)

    persistent = [False] * len(WINDOWS)
    for i in range(len(WINDOWS) - 1):
        persistent[i] = bool(all(local[i:]))

    adequacy = point_error <= TAU_E
    candidate = None
    for i, w in enumerate(WINDOWS[:-1]):
        if adequacy[i] and persistent[i]:
            candidate = int(w)
            break

    return {
        "estimated_msoh": candidate,
        "point_error": point_error,
        "p_gain": np.asarray(p_gain),
        "stable": np.asarray(stable),
        "local": np.asarray(local, dtype=bool),
        "persistent": np.asarray(persistent, dtype=bool),
        "adequacy": np.asarray(adequacy, dtype=bool),
    }


def classify_recovery(true_msoh, estimated):
    if true_msoh is None:
        return "correct_no_msoh" if estimated is None else "false_finite"
    if estimated is None:
        return "not_identified"
    if estimated == true_msoh:
        return "exact"
    return "early" if estimated < true_msoh else "late"


def run_simulation():
    rows, trace_rows = [], []
    seed_seq = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_seq.spawn(len(SCENARIOS) * len(N_GROUPS_GRID) * len(TAU_S_GRID))
    seed_i = 0

    for scenario in SCENARIOS:
        for n_groups in N_GROUPS_GRID:
            for tau_s in TAU_S_GRID:
                rng = np.random.default_rng(child_seeds[seed_i])
                seed_i += 1

                for rep in range(N_REPLICATES):
                    gerr = generate_group_errors(rng, scenario, n_groups)
                    d = decision_for_one_replicate(rng, gerr, tau_s)
                    est = d["estimated_msoh"]

                    rows.append({
                        "Scenario": scenario.name,
                        "True_MSOH": scenario.true_msoh,
                        "N_Groups": n_groups,
                        "Tau_S": tau_s,
                        "Replicate": rep,
                        "Estimated_MSOH": est,
                        "Recovery_Class": classify_recovery(scenario.true_msoh, est),
                    })

                    if rep < 25:
                        for wi, w in enumerate(WINDOWS):
                            trace_rows.append({
                                "Scenario": scenario.name,
                                "True_MSOH": scenario.true_msoh,
                                "N_Groups": n_groups,
                                "Tau_S": tau_s,
                                "Replicate": rep,
                                "Record_Type": "window",
                                "Window": int(w),
                                "Error_Point": float(d["point_error"][wi]),
                                "Adequate": bool(d["adequacy"][wi]),
                                "Persistent_Saturation": bool(d["persistent"][wi]),
                                "Estimated_MSOH": est,
                            })
                        for ti in range(len(WINDOWS) - 1):
                            trace_rows.append({
                                "Scenario": scenario.name,
                                "True_MSOH": scenario.true_msoh,
                                "N_Groups": n_groups,
                                "Tau_S": tau_s,
                                "Replicate": rep,
                                "Record_Type": "transition",
                                "Window": f"{WINDOWS[ti]}->{WINDOWS[ti+1]}",
                                "Estimated_MSOH": est,
                                "P_Gain_GT_Delta": float(d["p_gain"][ti]),
                                "Stable_Group_Proportion": float(d["stable"][ti]),
                                "Local_Saturation": bool(d["local"][ti]),
                            })

    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def summarize(results):
    out = []
    for keys, p in results.groupby(
        ["Scenario", "True_MSOH", "N_Groups", "Tau_S"],
        dropna=False, sort=True
    ):
        scenario, true_msoh, n_groups, tau_s = keys
        total = len(p)
        exact = int((p["Recovery_Class"] == "exact").sum())
        correct_no = int((p["Recovery_Class"] == "correct_no_msoh").sum())
        primary_success = correct_no if pd.isna(true_msoh) else exact

        finite_est = p["Estimated_MSOH"].dropna()
        within1 = 0
        if not pd.isna(true_msoh):
            within1 = int(((finite_est - float(true_msoh)).abs() <= 1).sum())

        out.append({
            "Scenario": scenario,
            "True_MSOH": true_msoh,
            "N_Groups": int(n_groups),
            "Tau_S": float(tau_s),
            "N_Replicates": int(total),
            "Primary_Success_Rate": primary_success / total,
            "Exact_Rate": exact / total,
            "Within_1_Day_Rate": within1 / total if not pd.isna(true_msoh) else np.nan,
            "Not_Identified_Rate": float((p["Recovery_Class"] == "not_identified").mean()),
            "Early_Rate": float((p["Recovery_Class"] == "early").mean()),
            "Late_Rate": float((p["Recovery_Class"] == "late").mean()),
            "False_Finite_Rate": float((p["Recovery_Class"] == "false_finite").mean()),
        })

    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("MSOH V2 criterion-level Monte Carlo simulation")
    print("=" * 112)
    print(f"Replicates/cell={N_REPLICATES}, bootstrap B={BOOTSTRAP_B}, seed={BASE_SEED}")
    print(f"Frozen: tau_E={TAU_E}, delta_G={DELTA_G}, alpha={ALPHA}, group_epsilon={GROUP_EPSILON}")
    print("Tau_S grid:", TAU_S_GRID, "(0.75 = primary; others = sensitivity only)")
    print("N groups grid:", N_GROUPS_GRID)
    print()

    results, trace = run_simulation()
    summary = summarize(results)

    results.to_csv(OUT / "replicate_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(OUT / "diagnostic_trace_first25.csv", index=False, encoding="utf-8-sig")

    primary = summary[summary["Tau_S"].eq(0.75)].copy()
    sensitivity = summary[summary["N_Groups"].eq(12)].copy()

    print("Primary tau_S=0.75:")
    print(primary[
        ["Scenario", "True_MSOH", "N_Groups", "Primary_Success_Rate", "Exact_Rate",
         "Within_1_Day_Rate", "Not_Identified_Rate", "False_Finite_Rate"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nN=12 tau_S sensitivity:")
    print(sensitivity[
        ["Scenario", "True_MSOH", "Tau_S", "Primary_Success_Rate", "Exact_Rate",
         "Not_Identified_Rate", "False_Finite_Rate"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nInterpretation rule:")
    print("Do not select tau_S by whichever value makes the historical dataset produce a finite MSOH.")
    print("Use this study to understand operating characteristics first; any future rule change must be justified prospectively.")
    print(f"\nOutput: {OUT}")
    print("=" * 112)


if __name__ == "__main__":
    main()
