from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config_v2 import RESULTS_V2
from src_v2.aggregation import (
    build_aggregation,
)


DEFAULT_INPUT = (
    RESULTS_V2
    / "06_signal_decisions"
    / "signal_decision_summary.csv"
)

DEFAULT_OUTPUT = (
    RESULTS_V2
    / "07_overall_msoh"
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MSOH V2 Phase 6: "
            "Core3 + Extended All4 aggregation"
        )
    )

    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return p.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到 Phase 5 summary：{args.input}"
        )

    signal_summary = pd.read_csv(
        args.input
    )

    (
        audit,
        overall,
        final_summary,
    ) = build_aggregation(
        signal_summary
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        args.output_dir
        / "aggregation_signal_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        args.output_dir
        / "overall_msoh_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_summary.to_csv(
        args.output_dir
        / "final_msoh_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Gate C audits
    # --------------------------------------------------------
    core = overall[
        overall[
            "Analysis_Level"
        ]
        == "Core3"
    ]

    extended = overall[
        overall[
            "Analysis_Level"
        ]
        == "Extended_All4"
    ]

    if len(core) != 1:
        raise RuntimeError(
            "Gate C audit failed: Core3 row missing."
        )

    if len(extended) != 1:
        raise RuntimeError(
            "Gate C audit failed: Extended All4 row missing."
        )

    defecation_status = (
        signal_summary.loc[
            signal_summary[
                "Signal"
            ]
            == "Defecation",
            "Decision_Status",
        ]
        .iloc[0]
    )

    if (
        defecation_status
        == "Indeterminate_due_to_sparse_schedule"
        and extended.iloc[0][
            "Decision_Status"
        ]
        != "Indeterminate_due_to_sparse_Defecation_schedule"
    ):
        raise RuntimeError(
            "Gate C audit failed: "
            "sparse Defecation indeterminacy was not propagated."
        )

    print(
        "=" * 118
    )

    print(
        "MSOH V2 Phase 6 完成："
        "Core3 + Extended All4 aggregation"
    )

    print(
        "=" * 118
    )

    print(
        "\nSignal aggregation audit:"
    )

    print(
        audit[
            [
                "Signal",
                "Signal_Status",
                "Signal_MSOH",
                "Finite_Signal_MSOH",
                "Adequacy_Onset",
                "Persistent_Saturation_Onset",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOverall decision:"
    )

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
        "\nInterpretation:"
    )

    for row in (
        overall.itertuples(
            index=False
        )
    ):
        print(
            f"  {row.Analysis_Level}: "
            f"{row.Reason}"
        )

    print(
        "\nGate C checks: PASS"
    )

    print(
        f"\n输出目录: "
        f"{args.output_dir}"
    )

    print(
        "=" * 118
    )


if __name__ == "__main__":
    main()
