from __future__ import annotations

"""
v2_14_simstudy_report.py

Aggregate Phase 10-13 simulation outputs into a compact evidence report.
This script does not modify config_v2.py or any real-data result.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_v2_simstudy"
OUT = BASE / "finalization_report"

FILES = {
    "criterion": BASE / "criterion_calibration" / "summary.csv",
    "stress": BASE / "stability_stress" / "stress_grid_summary.csv",
    "stress_tau": BASE / "stability_stress" / "tau_s_summary.csv",
    "ablation": BASE / "decision_rule_ablation" / "ablation_summary.csv",
    "raw_dgm": BASE / "raw_longitudinal_dgm" / "raw_dgm_summary.csv",
}


def require_files():
    missing = [
        str(p)
        for p in FILES.values()
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required simulation outputs:\n"
            + "\n".join(missing)
            + "\nRun run_msoh_v2_simstudy.py first."
        )


def fmt(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3f}"
    return str(x)


def main():
    require_files()
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    criterion = pd.read_csv(
        FILES["criterion"]
    )
    stress = pd.read_csv(
        FILES["stress"]
    )
    stress_tau = pd.read_csv(
        FILES["stress_tau"]
    )
    ablation = pd.read_csv(
        FILES["ablation"]
    )
    raw = pd.read_csv(
        FILES["raw_dgm"]
    )

    # Evidence snapshots only. No automatic final rule selection.
    crit_primary = criterion[
        criterion["Tau_S"].eq(0.75)
    ].copy()

    stress_primary = stress[
        (stress["N_Groups"].eq(12))
        & (stress["Tau_S"].eq(0.75))
    ].copy()

    abl_n12 = ablation[
        ablation["N_Groups"].eq(12)
    ].copy()

    raw_n12 = raw[
        raw["N_Groups"].eq(12)
    ].copy()

    crit_primary.to_csv(
        OUT / "criterion_primary_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stress_primary.to_csv(
        OUT / "stress_primary_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )
    abl_n12.to_csv(
        OUT / "ablation_n12_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_n12.to_csv(
        OUT / "raw_dgm_n12_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = []

    lines.append("# MSOH V2 simulation-study finalization report")
    lines.append("")
    lines.append("This report summarizes operating characteristics only; it does not automatically change the formal MSOH rule.")
    lines.append("")

    lines.append("## 1. Criterion-level positive control")
    lines.append("")
    lines.append(
        "The idealized known-MSOH scenarios test whether the decision engine can recover clear D3/D4/D5 and no-MSOH truths."
    )
    lines.append("")
    lines.append("Primary tau_S=0.75 summary:")
    lines.append("")
    lines.append(
        crit_primary[
            [
                "Scenario",
                "True_MSOH",
                "N_Groups",
                "Primary_Success_Rate",
                "Not_Identified_Rate",
                "False_Finite_Rate",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## 2. Stability-gate stress test")
    lines.append("")
    lines.append(
        "This test asks whether a hard StableGroupProp >= 0.75 gate causes non-identification when population-level information has plateaued but group-level errors still fluctuate."
    )
    lines.append("")
    lines.append(
        stress_tau.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## 3. Rule ablation")
    lines.append("")
    lines.append(
        "R1/R3 target population-level information saturation. R2 adds a stronger group-wise stability requirement."
    )
    lines.append("")
    lines.append(
        abl_n12[
            [
                "Scenario",
                "Rule",
                "Population_Exact_Rate",
                "Population_Correct_No_MSOH_Rate",
                "Population_Not_Identified_Rate",
                "Population_False_Finite_Rate",
                "Groupwise_Correct_No_MSOH_Rate",
                "Groupwise_False_Finite_Rate",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## 4. Raw longitudinal DGM")
    lines.append("")
    lines.append(
        "Raw synthetic longitudinal signals are generated, predicted under outer LOOG with a prespecified Ridge Direct probe, and passed through the MSOH decision rules."
    )
    lines.append("")
    lines.append(
        raw_n12[
            [
                "Scenario",
                "True_MSOH",
                "Rule",
                "Exact_Rate",
                "Correct_No_MSOH_Rate",
                "Not_Identified_Rate",
                "False_Finite_Rate",
                "Early_Rate",
                "Late_Rate",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## 5. Decision point before formal freeze")
    lines.append("")
    lines.append(
        "The remaining scientific choice is definitional: whether MSOH should mean population-level predictive information saturation, or the stronger conjunction of population saturation and group-wise error stability."
    )
    lines.append("")
    lines.append(
        "Do not choose the rule by which one produces a more attractive real-data MSOH. Choose it by the intended scientific estimand and the simulation operating characteristics."
    )
    lines.append("")

    report = "\n".join(lines)

    path = OUT / "MSOH_V2_simulation_finalization_report.md"
    path.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 112)
    print("MSOH V2 simulation finalization report")
    print("=" * 112)
    print(f"Report: {path}")
    print()
    print("All simulation blocks found and summarized.")
    print(
        "No formal decision rule was changed automatically."
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
