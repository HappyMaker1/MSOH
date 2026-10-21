from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SIM_RESULTS = ROOT / "results_v2_simulation"

SIGNAL_SUMMARY = (
    SIM_RESULTS
    / "06_signal_decisions"
    / "signal_decision_summary.csv"
)

OVERALL = (
    SIM_RESULTS
    / "07_overall_msoh"
    / "overall_msoh_decision.csv"
)

METRICS = (
    SIM_RESULTS
    / "04_metrics"
    / "signal_metrics_by_window.csv"
)

GAIN = (
    SIM_RESULTS
    / "05_uncertainty"
    / "signal_gain_bootstrap.csv"
)

OUTPUT = (
    SIM_RESULTS
    / "09_positive_control"
)


HISTORICAL_V1_REFERENCE = pd.DataFrame(
    [
        {
            "Framework": "Historical V1 simulation",
            "Method": "DTW All4",
            "Historical_MSOH": 4.0,
        },
        {
            "Framework": "Historical V1 simulation",
            "Method": "DTW Core3",
            "Historical_MSOH": 5.0,
        },
        {
            "Framework": "Historical V1 simulation",
            "Method": "Euclidean Feature",
            "Historical_MSOH": np.nan,
        },
    ]
)


def main():
    for path in [
        SIGNAL_SUMMARY,
        OVERALL,
        METRICS,
        GAIN,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 simulation V2 output：{path}"
            )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal = pd.read_csv(
        SIGNAL_SUMMARY
    )

    overall = pd.read_csv(
        OVERALL
    )

    metrics = pd.read_csv(
        METRICS
    )

    gain = pd.read_csv(
        GAIN
    )

    core = overall[
        overall[
            "Analysis_Level"
        ]
        == "Core3"
    ].iloc[0]

    all4 = overall[
        overall[
            "Analysis_Level"
        ]
        == "Extended_All4"
    ].iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "Framework": "V2 simulation",
                "Method": "Core3 signal-first nested framework",
                "Historical_MSOH": np.nan,
                "V2_MSOH": core["MSOH"],
                "V2_Status": core["Decision_Status"],
            },
            {
                "Framework": "V2 simulation",
                "Method": "Extended All4 signal-first nested framework",
                "Historical_MSOH": np.nan,
                "V2_MSOH": all4["MSOH"],
                "V2_Status": all4["Decision_Status"],
            },
        ]
    )

    HISTORICAL_V1_REFERENCE.to_csv(
        OUTPUT / "historical_v1_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT / "v2_positive_control_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    signal.to_csv(
        OUTPUT / "v2_signal_decisions_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT / "v2_signal_metrics_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gain.to_csv(
        OUTPUT / "v2_gain_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 116)
    print("MSOH V2 simulation positive-control report")
    print("=" * 116)

    print("\nHistorical V1 reference (descriptive only):")
    print(
        HISTORICAL_V1_REFERENCE.to_string(
            index=False
        )
    )

    print("\nV2 signal decisions:")
    print(
        signal[
            [
                "Signal",
                "Adequacy_Onset",
                "Persistent_Saturation_Onset",
                "Candidate_MSOH",
                "Decision_Status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nV2 overall:")
    print(
        overall[
            [
                "Analysis_Level",
                "MSOH",
                "Decision_Status",
                "Limiting_Signals",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nInterpretation rule: "
        "V2 is NOT required to reproduce V1 Day4/Day5 exactly, "
        "because V2 uses signal-first nested predictor selection and "
        "persistent-saturation aggregation. The historical V1 values are "
        "a positive-control reference, not a forced target."
    )

    print(
        "\nIf V2 identifies a finite Core3 MSOH, that supports positive-control "
        "sensitivity under the stricter V2 framework. If it does not, inspect "
        "the signal-specific bottleneck and saturation trace rather than tuning "
        "thresholds post hoc."
    )

    print(f"\n输出目录: {OUTPUT}")
    print("=" * 116)


if __name__ == "__main__":
    main()
