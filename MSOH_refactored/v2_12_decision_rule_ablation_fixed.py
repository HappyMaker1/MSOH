from __future__ import annotations

"""
v2_12_decision_rule_ablation.py

MSOH V2 — decision-rule ablation simulation (fixed shared-resampling version).

Key fix
-------
R1_gain_only and R3_gain_plus_diagnostic have the SAME decision rule by design.
Therefore they must use the exact same bootstrap-derived P(Gain > delta) values
within each simulated replicate. This version computes the stochastic diagnostics
once per replicate and applies all rules to the same diagnostics.

This removes Monte Carlo discrepancies between R1 and R3 and makes the ablation
a true rule comparison.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2_simstudy" / "decision_rule_ablation"

WINDOWS = np.array([2, 3, 4, 5, 6], dtype=int)

TAU_E = 0.40
DELTA_G = 0.03
ALPHA = 0.10
TAU_S = 0.75
GROUP_EPSILON = 0.03

N_GROUPS_GRID = [12, 24, 48]
N_REPLICATES = 1000
BOOTSTRAP_B = 1000
BASE_SEED = 20260930


@dataclass(frozen=True)
class Scenario:
    name: str
    truth_type: str
    true_msoh_population: int | None
    true_msoh_groupwise: int | None
    error_curve: tuple[float, float, float, float, float]
    pre_noise_sd: float
    post_noise_sd: float
    group_heterogeneity: float
    divergence_sd: float


SCENARIOS = [
    Scenario(
        "finite_D4_clean",
        "population_and_groupwise_D4",
        4, 4,
        (0.78, 0.55, 0.31, 0.30, 0.30),
        0.05, 0.008, 0.20, 0.0,
    ),
    Scenario(
        "population_D4_group_divergent",
        "population_D4_groupwise_none",
        4, None,
        (0.78, 0.55, 0.31, 0.30, 0.30),
        0.05, 0.055, 0.25, 0.060,
    ),
    Scenario(
        "continued_information",
        "no_population_msoh",
        None, None,
        (0.78, 0.61, 0.46, 0.34, 0.25),
        0.05, 0.04, 0.20, 0.0,
    ),
    Scenario(
        "saturated_inadequate",
        "no_msoh_inadequate",
        None, None,
        (0.72, 0.58, 0.52, 0.51, 0.51),
        0.04, 0.015, 0.20, 0.0,
    ),
]

RULES = [
    "R1_gain_only",
    "R2_hard_stability",
    "R3_gain_plus_diagnostic",
]


def generate_group_errors(rng, scenario, n_groups):
    target = np.asarray(scenario.error_curve, dtype=float)
    multiplier = rng.lognormal(
        mean=-0.5 * scenario.group_heterogeneity**2,
        sigma=scenario.group_heterogeneity,
        size=n_groups,
    )
    base = target[None, :] * multiplier[:, None]
    noise = np.zeros_like(base)

    for wi, w in enumerate(WINDOWS):
        sd = scenario.pre_noise_sd if w < 4 else scenario.post_noise_sd
        noise[:, wi] = rng.normal(0.0, sd, size=n_groups)

    if scenario.divergence_sd > 0:
        for wi in [3, 4]:
            d = rng.normal(0.0, scenario.divergence_sd, size=n_groups)
            noise[:, wi] += d - d.mean()

    return np.clip(base + noise, 1e-6, None)


def compute_shared_diagnostics(rng, errors):
    point_error = errors.mean(axis=0)
    adequacy = point_error <= TAU_E

    p_gain = []
    stability = []

    n = errors.shape[0]

    # Use one paired resample matrix for ALL transitions in this replicate.
    # This is stricter pairing across windows and guarantees identical diagnostics
    # for rules that share the same saturation definition.
    idx = rng.integers(
        0,
        n,
        size=(BOOTSTRAP_B, n),
    )

    boot_error = np.zeros(
        (BOOTSTRAP_B, len(WINDOWS)),
        dtype=float,
    )

    for wi in range(len(WINDOWS)):
        boot_error[:, wi] = errors[idx, wi].mean(axis=1)

    for i in range(len(WINDOWS) - 1):
        gain = boot_error[:, i] - boot_error[:, i + 1]
        p_gain.append(float(np.mean(gain > DELTA_G)))

        stability.append(
            float(
                np.mean(
                    np.abs(errors[:, i] - errors[:, i + 1])
                    <= GROUP_EPSILON
                )
            )
        )

    return {
        "Point_Error": point_error,
        "Adequacy": np.asarray(adequacy, dtype=bool),
        "P_Gain": np.asarray(p_gain, dtype=float),
        "Stability": np.asarray(stability, dtype=float),
    }


def apply_rule(diag, rule):
    local = []

    for i in range(len(WINDOWS) - 1):
        p = diag["P_Gain"][i]
        stab = diag["Stability"][i]

        if rule == "R2_hard_stability":
            ls = (p < ALPHA) and (stab >= TAU_S)
        else:
            ls = p < ALPHA

        local.append(bool(ls))

    persistent = np.zeros(len(WINDOWS), dtype=bool)
    for i in range(len(WINDOWS) - 1):
        persistent[i] = bool(all(local[i:]))

    estimated = None
    for i, w in enumerate(WINDOWS[:-1]):
        if diag["Adequacy"][i] and persistent[i]:
            estimated = int(w)
            break

    return {
        "Estimated_MSOH": estimated,
        "Local": np.asarray(local, dtype=bool),
        "Persistent": persistent,
    }


def classify(true_msoh, estimated):
    if true_msoh is None:
        return "correct_no_msoh" if estimated is None else "false_finite"
    if estimated is None:
        return "not_identified"
    if estimated == true_msoh:
        return "exact"
    return "early" if estimated < true_msoh else "late"


def run():
    rows = []
    traces = []

    seed_seq = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_seq.spawn(len(SCENARIOS) * len(N_GROUPS_GRID))
    si = 0

    for scenario in SCENARIOS:
        for n_groups in N_GROUPS_GRID:
            rng = np.random.default_rng(child_seeds[si])
            si += 1

            for rep in range(N_REPLICATES):
                errors = generate_group_errors(rng, scenario, n_groups)
                diag = compute_shared_diagnostics(rng, errors)

                results = {
                    rule: apply_rule(diag, rule)
                    for rule in RULES
                }

                # Internal invariant: R1 and R3 must be exactly identical.
                if (
                    results["R1_gain_only"]["Estimated_MSOH"]
                    != results["R3_gain_plus_diagnostic"]["Estimated_MSOH"]
                    or not np.array_equal(
                        results["R1_gain_only"]["Local"],
                        results["R3_gain_plus_diagnostic"]["Local"],
                    )
                ):
                    raise AssertionError("R1 and R3 diverged; shared-diagnostic invariant failed.")

                for rule in RULES:
                    est = results[rule]["Estimated_MSOH"]

                    rows.append(
                        {
                            "Scenario": scenario.name,
                            "Truth_Type": scenario.truth_type,
                            "N_Groups": n_groups,
                            "Rule": rule,
                            "Replicate": rep,
                            "True_MSOH_Population": scenario.true_msoh_population,
                            "True_MSOH_Groupwise": scenario.true_msoh_groupwise,
                            "Estimated_MSOH": est,
                            "Population_Recovery": classify(
                                scenario.true_msoh_population, est
                            ),
                            "Groupwise_Recovery": classify(
                                scenario.true_msoh_groupwise, est
                            ),
                        }
                    )

                    if rep < 20:
                        for ti in range(len(WINDOWS) - 1):
                            traces.append(
                                {
                                    "Scenario": scenario.name,
                                    "N_Groups": n_groups,
                                    "Rule": rule,
                                    "Replicate": rep,
                                    "FromWindow": int(WINDOWS[ti]),
                                    "ToWindow": int(WINDOWS[ti + 1]),
                                    "P_Gain_GT_Delta": float(diag["P_Gain"][ti]),
                                    "Stable_Group_Proportion": float(
                                        diag["Stability"][ti]
                                    ),
                                    "Local_Saturation": bool(
                                        results[rule]["Local"][ti]
                                    ),
                                    "Estimated_MSOH": est,
                                }
                            )

    return pd.DataFrame(rows), pd.DataFrame(traces)


def summarize(results):
    rows = []

    for keys, part in results.groupby(
        ["Scenario", "Truth_Type", "N_Groups", "Rule"],
        sort=True,
    ):
        scenario, truth_type, n_groups, rule = keys

        rows.append(
            {
                "Scenario": scenario,
                "Truth_Type": truth_type,
                "N_Groups": int(n_groups),
                "Rule": rule,
                "N_Replicates": int(len(part)),
                "Population_Exact_Rate": float(
                    (part["Population_Recovery"] == "exact").mean()
                ),
                "Population_Correct_No_MSOH_Rate": float(
                    (part["Population_Recovery"] == "correct_no_msoh").mean()
                ),
                "Population_Not_Identified_Rate": float(
                    (part["Population_Recovery"] == "not_identified").mean()
                ),
                "Population_Early_Rate": float(
                    (part["Population_Recovery"] == "early").mean()
                ),
                "Population_Late_Rate": float(
                    (part["Population_Recovery"] == "late").mean()
                ),
                "Population_False_Finite_Rate": float(
                    (part["Population_Recovery"] == "false_finite").mean()
                ),
                "Groupwise_Exact_Rate": float(
                    (part["Groupwise_Recovery"] == "exact").mean()
                ),
                "Groupwise_Correct_No_MSOH_Rate": float(
                    (part["Groupwise_Recovery"] == "correct_no_msoh").mean()
                ),
                "Groupwise_False_Finite_Rate": float(
                    (part["Groupwise_Recovery"] == "false_finite").mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("MSOH V2 decision-rule ablation — shared-resampling fixed version")
    print("=" * 120)
    print(
        f"replicates/cell={N_REPLICATES}; bootstrap B={BOOTSTRAP_B}; "
        f"seed={BASE_SEED}"
    )
    print(
        f"Frozen: tau_E={TAU_E}, delta_G={DELTA_G}, alpha={ALPHA}, "
        f"tau_S={TAU_S}, group_epsilon={GROUP_EPSILON}"
    )
    print("Rules:", RULES)
    print("N groups:", N_GROUPS_GRID)
    print()

    results, traces = run()
    summary = summarize(results)

    results.to_csv(
        OUT / "replicate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT / "ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    traces.to_csv(
        OUT / "transition_diagnostics_first20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n12 = summary[summary["N_Groups"].eq(12)].copy()

    print("N=12 rule comparison:")
    print(
        n12[
            [
                "Scenario",
                "Rule",
                "Population_Exact_Rate",
                "Population_Correct_No_MSOH_Rate",
                "Population_Not_Identified_Rate",
                "Population_Early_Rate",
                "Population_Late_Rate",
                "Population_False_Finite_Rate",
                "Groupwise_Correct_No_MSOH_Rate",
                "Groupwise_False_Finite_Rate",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nInvariant check:")
    print("R1_gain_only and R3_gain_plus_diagnostic use identical decisions by construction: PASS")

    print("\nKey interpretation:")
    print(
        "R1/R3 target population-level information saturation. "
        "R2 targets a stronger population+group-stability notion."
    )
    print(
        "population_D4_group_divergent intentionally separates the two estimands."
    )

    print(f"\nOutput: {OUT}")
    print("=" * 120)


if __name__ == "__main__":
    main()
