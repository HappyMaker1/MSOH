from __future__ import annotations

import numpy as np
import pandas as pd

from src_v2.decision import build_signal_decisions
from src_v2.aggregation import build_aggregation


CORE = ["Weight", "Status_Score", "Feeding"]
ALL = CORE + ["Defecation"]


def registry_fixture():
    return pd.DataFrame(
        [
            {
                "name": "Weight",
                "role": "core3",
                "density": "dense",
                "outcome_kind": "continuous",
                "error_threshold": 0.40,
                "gain_delta": 0.03,
                "alpha": 0.10,
            },
            {
                "name": "Status_Score",
                "role": "core3",
                "density": "dense",
                "outcome_kind": "continuous_group_mean",
                "error_threshold": 0.40,
                "gain_delta": 0.03,
                "alpha": 0.10,
            },
            {
                "name": "Feeding",
                "role": "core3",
                "density": "dense",
                "outcome_kind": "continuous",
                "error_threshold": 0.40,
                "gain_delta": 0.03,
                "alpha": 0.10,
            },
            {
                "name": "Defecation",
                "role": "extended_all4",
                "density": "planned_sparse",
                "outcome_kind": "continuous_sparse",
                "error_threshold": 0.40,
                "gain_delta": 0.03,
                "alpha": 0.10,
            },
        ]
    )


def _dense_errors(kind):
    if kind == "early":
        return [0.65, 0.35, 0.34, 0.33, 0.32]
    if kind == "late":
        return [0.75, 0.65, 0.55, 0.35, 0.34]
    if kind == "adequate_not_sat":
        return [0.70, 0.35, 0.34, 0.33, 0.32]
    if kind == "sat_inadequate":
        return [0.75, 0.70, 0.68, 0.67, 0.66]
    if kind == "not_reached":
        return [0.90, 0.82, 0.74, 0.68, 0.62]
    raise ValueError(kind)


def _dense_transition_controls(kind):
    # returns (P_Gain_GT_Delta, stability) per D2->3,3->4,4->5,5->6
    if kind == "early":
        return [
            (0.90, 0.30),
            (0.05, 0.90),
            (0.04, 0.90),
            (0.03, 0.90),
        ]
    if kind == "late":
        return [
            (0.90, 0.30),
            (0.80, 0.30),
            (0.70, 0.30),
            (0.04, 0.90),
        ]
    if kind == "adequate_not_sat":
        return [
            (0.90, 0.30),
            (0.60, 0.50),
            (0.40, 0.50),
            (0.30, 0.50),
        ]
    if kind == "sat_inadequate":
        return [
            (0.05, 0.90),
            (0.05, 0.90),
            (0.05, 0.90),
            (0.05, 0.90),
        ]
    if kind == "not_reached":
        return [
            (0.90, 0.20),
            (0.80, 0.20),
            (0.70, 0.20),
            (0.60, 0.20),
        ]
    raise ValueError(kind)


def make_fixture(kind):
    error_rows = []
    gain_rows = []
    stability_rows = []

    windows = [2, 3, 4, 5, 6]

    for signal in CORE:
        errors = _dense_errors(kind)
        controls = _dense_transition_controls(kind)

        for w, e in zip(windows, errors):
            error_rows.append(
                {
                    "Signal": signal,
                    "Window": w,
                    "Error_Point": e,
                    "Error_CI_Low": max(0.0, e - 0.05),
                    "Error_CI_High": e + 0.05,
                }
            )

        for i, (p, stab) in enumerate(controls):
            a = windows[i]
            b = windows[i + 1]
            gain_rows.append(
                {
                    "Signal": signal,
                    "FromWindow": a,
                    "ToWindow": b,
                    "TransitionType": "adjacent_day",
                    "Gain_Point": errors[i] - errors[i + 1],
                    "Gain_CI_Low": -0.01,
                    "Gain_CI_High": 0.10,
                    "P_Gain_GT_Delta": p,
                }
            )
            stability_rows.append(
                {
                    "Signal": signal,
                    "FromWindow": a,
                    "ToWindow": b,
                    "Stable_Group_Proportion": stab,
                }
            )

    # Planned-sparse Defecation fixture. It is intentionally adequate by D6,
    # but must remain formal-indeterminate under Gate B.
    def_windows = [2, 4, 6]
    def_errors = [0.85, 0.45, 0.30]
    def_controls = [(0.70, 0.30), (0.20, 0.40)]

    for w, e in zip(def_windows, def_errors):
        error_rows.append(
            {
                "Signal": "Defecation",
                "Window": w,
                "Error_Point": e,
                "Error_CI_Low": max(0.0, e - 0.05),
                "Error_CI_High": e + 0.10,
            }
        )

    for i, (p, stab) in enumerate(def_controls):
        a = def_windows[i]
        b = def_windows[i + 1]
        gain_rows.append(
            {
                "Signal": "Defecation",
                "FromWindow": a,
                "ToWindow": b,
                "TransitionType": "sparse_observation_update",
                "Gain_Point": def_errors[i] - def_errors[i + 1],
                "Gain_CI_Low": 0.0,
                "Gain_CI_High": 0.50,
                "P_Gain_GT_Delta": p,
            }
        )
        stability_rows.append(
            {
                "Signal": "Defecation",
                "FromWindow": a,
                "ToWindow": b,
                "Stable_Group_Proportion": stab,
            }
        )

    return (
        pd.DataFrame(error_rows),
        pd.DataFrame(gain_rows),
        pd.DataFrame(stability_rows),
        registry_fixture(),
    )


def expected_for_kind(kind):
    if kind == "early":
        return {
            "dense_status": "Identified",
            "dense_msoh": 3.0,
            "core_status": "Identified",
            "core_msoh": 3.0,
        }
    if kind == "late":
        return {
            "dense_status": "Identified",
            "dense_msoh": 5.0,
            "core_status": "Identified",
            "core_msoh": 5.0,
        }
    if kind == "adequate_not_sat":
        return {
            "dense_status": "Adequate_not_saturated",
            "dense_msoh": np.nan,
            "core_status": "Not_identified_within_evaluated_horizon",
            "core_msoh": np.nan,
        }
    if kind == "sat_inadequate":
        return {
            "dense_status": "Saturated_inadequate",
            "dense_msoh": np.nan,
            "core_status": "Not_identified_within_evaluated_horizon",
            "core_msoh": np.nan,
        }
    if kind == "not_reached":
        return {
            "dense_status": "Not_reached",
            "dense_msoh": np.nan,
            "core_status": "Not_identified_within_evaluated_horizon",
            "core_msoh": np.nan,
        }
    raise ValueError(kind)


def validate_synthetic_cases():
    cases = [
        "early",
        "late",
        "adequate_not_sat",
        "sat_inadequate",
        "not_reached",
    ]

    rows = []

    for kind in cases:
        e, g, st, reg = make_fixture(kind)
        _, _, summary = build_signal_decisions(
            error_table=e,
            gain_table=g,
            stability_table=st,
            registry=reg,
            tau_s=0.75,
        )

        audit, overall, final = build_aggregation(summary)

        exp = expected_for_kind(kind)

        for signal in CORE:
            row = summary[summary.Signal == signal].iloc[0]

            status_ok = row.Decision_Status == exp["dense_status"]

            if np.isnan(exp["dense_msoh"]):
                msoh_ok = pd.isna(row.Candidate_MSOH)
            else:
                msoh_ok = float(row.Candidate_MSOH) == exp["dense_msoh"]

            rows.append(
                {
                    "Case": kind,
                    "Check": f"{signal}_decision",
                    "Observed": f"{row.Decision_Status}|{row.Candidate_MSOH}",
                    "Expected": f"{exp['dense_status']}|{exp['dense_msoh']}",
                    "PASS": bool(status_ok and msoh_ok),
                }
            )

        core = overall[overall.Analysis_Level == "Core3"].iloc[0]

        core_status_ok = core.Decision_Status == exp["core_status"]

        if np.isnan(exp["core_msoh"]):
            core_msoh_ok = pd.isna(core.MSOH)
        else:
            core_msoh_ok = float(core.MSOH) == exp["core_msoh"]

        rows.append(
            {
                "Case": kind,
                "Check": "Core3_aggregation",
                "Observed": f"{core.Decision_Status}|{core.MSOH}",
                "Expected": f"{exp['core_status']}|{exp['core_msoh']}",
                "PASS": bool(core_status_ok and core_msoh_ok),
            }
        )

        d = summary[summary.Signal == "Defecation"].iloc[0]
        rows.append(
            {
                "Case": kind,
                "Check": "Defecation_sparse_indeterminate",
                "Observed": f"{d.Decision_Status}|{d.Candidate_MSOH}",
                "Expected": "Indeterminate_due_to_sparse_schedule|nan",
                "PASS": bool(
                    d.Decision_Status
                    == "Indeterminate_due_to_sparse_schedule"
                    and pd.isna(d.Candidate_MSOH)
                ),
            }
        )

        all4 = overall[overall.Analysis_Level == "Extended_All4"].iloc[0]
        rows.append(
            {
                "Case": kind,
                "Check": "Extended_All4_sparse_propagation",
                "Observed": f"{all4.Decision_Status}|{all4.MSOH}",
                "Expected": "Indeterminate_due_to_sparse_Defecation_schedule|nan",
                "PASS": bool(
                    all4.Decision_Status
                    == "Indeterminate_due_to_sparse_Defecation_schedule"
                    and pd.isna(all4.MSOH)
                ),
            }
        )

    out = pd.DataFrame(rows)

    if not out.PASS.all():
        bad = out[~out.PASS]
        raise AssertionError(
            "Synthetic decision validation failed:\n"
            + bad.to_string(index=False)
        )

    return out


def validate_real_regression(
    signal_summary: pd.DataFrame,
    overall: pd.DataFrame,
):
    expected_signal = {
        "Feeding": {
            "Decision_Status": "Adequate_not_saturated",
            "Adequacy_Onset": 3.0,
            "Candidate_MSOH": np.nan,
        },
        "Status_Score": {
            "Decision_Status": "Not_reached",
            "Adequacy_Onset": np.nan,
            "Candidate_MSOH": np.nan,
        },
        "Weight": {
            "Decision_Status": "Not_reached",
            "Adequacy_Onset": np.nan,
            "Candidate_MSOH": np.nan,
        },
        "Defecation": {
            "Decision_Status": "Indeterminate_due_to_sparse_schedule",
            "Adequacy_Onset": 6.0,
            "Candidate_MSOH": np.nan,
        },
    }

    rows = []

    for signal, exp in expected_signal.items():
        obs = signal_summary[
            signal_summary.Signal.astype(str) == signal
        ].iloc[0]

        checks = [
            (
                "Decision_Status",
                str(obs.Decision_Status),
                exp["Decision_Status"],
                str(obs.Decision_Status) == exp["Decision_Status"],
            ),
            (
                "Adequacy_Onset",
                obs.Adequacy_Onset,
                exp["Adequacy_Onset"],
                (
                    pd.isna(obs.Adequacy_Onset)
                    and pd.isna(exp["Adequacy_Onset"])
                )
                or (
                    pd.notna(obs.Adequacy_Onset)
                    and pd.notna(exp["Adequacy_Onset"])
                    and float(obs.Adequacy_Onset)
                    == float(exp["Adequacy_Onset"])
                ),
            ),
            (
                "Candidate_MSOH",
                obs.Candidate_MSOH,
                exp["Candidate_MSOH"],
                pd.isna(obs.Candidate_MSOH),
            ),
        ]

        for field, observed, expected, ok in checks:
            rows.append(
                {
                    "Level": "Signal",
                    "Entity": signal,
                    "Field": field,
                    "Observed": observed,
                    "Expected": expected,
                    "PASS": bool(ok),
                }
            )

    expected_overall = {
        "Core3": "Not_identified_within_evaluated_horizon",
        "Extended_All4": "Indeterminate_due_to_sparse_Defecation_schedule",
    }

    for level, expected_status in expected_overall.items():
        obs = overall[
            overall.Analysis_Level.astype(str) == level
        ].iloc[0]

        rows.append(
            {
                "Level": "Overall",
                "Entity": level,
                "Field": "Decision_Status",
                "Observed": obs.Decision_Status,
                "Expected": expected_status,
                "PASS": bool(obs.Decision_Status == expected_status),
            }
        )

        rows.append(
            {
                "Level": "Overall",
                "Entity": level,
                "Field": "MSOH",
                "Observed": obs.MSOH,
                "Expected": np.nan,
                "PASS": bool(pd.isna(obs.MSOH)),
            }
        )

    out = pd.DataFrame(rows)

    if not out.PASS.all():
        bad = out[~out.PASS]
        raise AssertionError(
            "Real-data regression validation failed:\n"
            + bad.to_string(index=False)
        )

    return out
