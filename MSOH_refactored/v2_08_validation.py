from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config_v2 import RESULTS_V2
from src_v2.validation import (
    validate_synthetic_cases,
    validate_real_regression,
)


DEFAULT_SIGNAL_SUMMARY = (
    RESULTS_V2
    / "06_signal_decisions"
    / "signal_decision_summary.csv"
)

DEFAULT_OVERALL = (
    RESULTS_V2
    / "07_overall_msoh"
    / "overall_msoh_decision.csv"
)

DEFAULT_BASELINE_MANIFEST = (
    RESULTS_V2
    / "00_baseline"
    / "v1_baseline_manifest.csv"
)

DEFAULT_BASELINE_SNAPSHOT = (
    RESULTS_V2
    / "00_baseline"
    / "v1_key_metrics_snapshot.csv"
)

DEFAULT_OUTPUT = (
    RESULTS_V2
    / "08_validation"
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MSOH V2 Phase 7: "
            "synthetic decision fixtures + real-data regression validation"
        )
    )

    p.add_argument(
        "--signal-summary",
        type=Path,
        default=DEFAULT_SIGNAL_SUMMARY,
    )

    p.add_argument(
        "--overall",
        type=Path,
        default=DEFAULT_OVERALL,
    )

    p.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
    )

    p.add_argument(
        "--baseline-snapshot",
        type=Path,
        default=DEFAULT_BASELINE_SNAPSHOT,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return p.parse_args()


def main():
    args = parse_args()

    for path in [
        args.signal_summary,
        args.overall,
        args.baseline_manifest,
        args.baseline_snapshot,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Validation prerequisite missing: {path}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    synthetic = validate_synthetic_cases()

    signal_summary = pd.read_csv(
        args.signal_summary
    )

    overall = pd.read_csv(
        args.overall
    )

    real = validate_real_regression(
        signal_summary=signal_summary,
        overall=overall,
    )

    baseline_audit = pd.DataFrame(
        [
            {
                "Artifact": "v1_baseline_manifest.csv",
                "Path": str(args.baseline_manifest),
                "Exists": args.baseline_manifest.exists(),
            },
            {
                "Artifact": "v1_key_metrics_snapshot.csv",
                "Path": str(args.baseline_snapshot),
                "Exists": args.baseline_snapshot.exists(),
            },
        ]
    )

    synthetic.to_csv(
        args.output_dir
        / "synthetic_decision_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    real.to_csv(
        args.output_dir
        / "real_data_regression_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_audit.to_csv(
        args.output_dir
        / "v1_baseline_presence_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_summary = pd.DataFrame(
        [
            {
                "Validation_Block": "Synthetic decision fixtures",
                "N_Checks": len(synthetic),
                "N_Passed": int(synthetic.PASS.sum()),
                "Status": "PASS" if synthetic.PASS.all() else "FAIL",
            },
            {
                "Validation_Block": "Real-data regression",
                "N_Checks": len(real),
                "N_Passed": int(real.PASS.sum()),
                "Status": "PASS" if real.PASS.all() else "FAIL",
            },
            {
                "Validation_Block": "V1 frozen baseline artifacts present",
                "N_Checks": len(baseline_audit),
                "N_Passed": int(baseline_audit.Exists.sum()),
                "Status": "PASS" if baseline_audit.Exists.all() else "FAIL",
            },
        ]
    )

    validation_summary.to_csv(
        args.output_dir
        / "validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not (
        synthetic.PASS.all()
        and real.PASS.all()
        and baseline_audit.Exists.all()
    ):
        raise RuntimeError(
            "Phase 7 validation failed."
        )

    print("=" * 116)
    print("MSOH V2 Phase 7 完成：validation / regression tests")
    print("=" * 116)

    print("\nValidation summary:")
    print(validation_summary.to_string(index=False))

    print("\nSynthetic scenarios covered:")
    print(
        "  early finite MSOH / late finite MSOH / "
        "adequate-but-not-saturated / saturated-but-inadequate / "
        "not reached / sparse-schedule indeterminate"
    )

    print("\nReal-data regression:")
    print(
        real[
            [
                "Level",
                "Entity",
                "Field",
                "Observed",
                "Expected",
                "PASS",
            ]
        ].to_string(index=False)
    )

    print(
        "\nImportant scope note: "
        "this Phase 7 validates the frozen V2 decision/aggregation logic "
        "with controlled synthetic fixtures and locks the current real-data "
        "result by regression test. It does NOT claim an end-to-end rerun of "
        "the historical V1 simulated animal dataset, because that dataset's "
        "current local input path is not part of the V2 configuration."
    )

    print("\nPhase 7 checks: PASS")
    print(f"\n输出目录: {args.output_dir}")
    print("=" * 116)


if __name__ == "__main__":
    main()
