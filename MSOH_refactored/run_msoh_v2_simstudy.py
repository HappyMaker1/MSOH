from __future__ import annotations

"""
run_msoh_v2_simstudy.py

One-command runner for MSOH V2 simulation-study finalization.

Default behavior:
- skips a phase when its expected output already exists;
- therefore your completed Phase 10 / Phase 11 results are not recomputed;
- runs Phase 12, 13, and 14 as needed.

Use --force to rerun everything.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PHASES = [
    (
        "Phase 10 criterion calibration",
        "v2_10_msoh_criterion_simulation.py",
        ROOT
        / "results_v2_simstudy"
        / "criterion_calibration"
        / "summary.csv",
    ),
    (
        "Phase 11 stability stress",
        "v2_11_stability_stress_simulation.py",
        ROOT
        / "results_v2_simstudy"
        / "stability_stress"
        / "stress_grid_summary.csv",
    ),
    (
        "Phase 12 rule ablation",
        "v2_12_decision_rule_ablation.py",
        ROOT
        / "results_v2_simstudy"
        / "decision_rule_ablation"
        / "ablation_summary.csv",
    ),
    (
        "Phase 13 raw longitudinal DGM",
        "v2_13_raw_longitudinal_dgm.py",
        ROOT
        / "results_v2_simstudy"
        / "raw_longitudinal_dgm"
        / "raw_dgm_summary.csv",
    ),
    (
        "Phase 14 finalization report",
        "v2_14_simstudy_report.py",
        ROOT
        / "results_v2_simstudy"
        / "finalization_report"
        / "MSOH_V2_simulation_finalization_report.md",
    ),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--force",
        action="store_true",
        help="rerun phases even if expected outputs already exist",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 108)
    print("MSOH V2 simulation-study one-command runner")
    print("=" * 108)
    print(f"Python: {sys.executable}")
    print(f"Project root: {ROOT}")
    print(
        "Mode:",
        "FORCE rerun all"
        if args.force
        else "skip completed outputs",
    )
    print()

    for title, script_name, expected in PHASES:
        script = ROOT / script_name

        if not script.exists():
            raise FileNotFoundError(
                f"Missing script: {script}"
            )

        if (
            expected.exists()
            and not args.force
        ):
            print(
                f"[SKIP] {title}: "
                f"{expected}"
            )
            continue

        print(
            f"\n[RUN ] {title}"
        )
        print(
            f"      {script}"
        )

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
            ],
            cwd=str(ROOT),
        )

        if proc.returncode != 0:
            raise SystemExit(
                f"{title} failed with exit code "
                f"{proc.returncode}"
            )

        if not expected.exists():
            raise FileNotFoundError(
                f"{title} finished but expected output is missing: "
                f"{expected}"
            )

        print(
            f"[PASS] {title}"
        )

    print()
    print("=" * 108)
    print("MSOH V2 simulation-study pipeline complete")
    print(
        "Final report:",
        ROOT
        / "results_v2_simstudy"
        / "finalization_report"
        / "MSOH_V2_simulation_finalization_report.md",
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
