from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config_v2 import RESULTS_V2

from src_v2.uncertainty import (
    run_paired_bootstrap,
)


DEFAULT_INPUT = (
    RESULTS_V2
    / "03_predictions"
    / "nested_outer_predictions.csv"
)

DEFAULT_PHASE3_METRICS = (
    RESULTS_V2
    / "04_metrics"
    / "signal_metrics_by_window.csv"
)

DEFAULT_PHASE3_GAINS = (
    RESULTS_V2
    / "04_metrics"
    / "signal_gain_by_transition.csv"
)

DEFAULT_OUTPUT = (
    RESULTS_V2
    / "05_uncertainty"
)

DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260928
DEFAULT_GAIN_DELTA = 0.03
DEFAULT_GROUP_EPSILON = 0.03


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MSOH V2 Phase 4: "
            "paired group bootstrap uncertainty"
        )
    )

    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    p.add_argument(
        "--phase3-metrics",
        type=Path,
        default=DEFAULT_PHASE3_METRICS,
    )

    p.add_argument(
        "--phase3-gains",
        type=Path,
        default=DEFAULT_PHASE3_GAINS,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    p.add_argument(
        "--gain-delta",
        type=float,
        default=DEFAULT_GAIN_DELTA,
    )

    p.add_argument(
        "--group-epsilon",
        type=float,
        default=DEFAULT_GROUP_EPSILON,
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.bootstrap < 1000:
        raise ValueError(
            "Phase 4 正式分析要求 bootstrap >= 1000。"
        )

    for path in [
        args.input,
        args.phase3_metrics,
        args.phase3_gains,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到输入文件：{path}"
            )

    predictions = pd.read_csv(
        args.input
    )

    phase3_metrics = pd.read_csv(
        args.phase3_metrics
    )

    phase3_gains = pd.read_csv(
        args.phase3_gains
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        error_uncertainty,
        gain_uncertainty,
        stability_uncertainty,
        audit,
    ) = run_paired_bootstrap(
        predictions=predictions,
        bootstrap_b=args.bootstrap,
        seed=args.seed,
        gain_delta=args.gain_delta,
        group_epsilon=args.group_epsilon,
    )

    # --------------------------------------------------------
    # Regression checks against Phase 3 point estimates
    # --------------------------------------------------------
    check_error = (
        error_uncertainty.merge(
            phase3_metrics[
                [
                    "Signal",
                    "Window",
                    "NRMSE",
                ]
            ],
            on=[
                "Signal",
                "Window",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    check_error[
        "AbsDiff_vs_Phase3"
    ] = np.abs(
        check_error[
            "Error_Point"
        ]
        - check_error[
            "NRMSE"
        ]
    )

    max_error_diff = float(
        check_error[
            "AbsDiff_vs_Phase3"
        ].max()
    )

    check_gain = (
        gain_uncertainty.merge(
            phase3_gains[
                [
                    "Signal",
                    "FromWindow",
                    "ToWindow",
                    "Gain_NRMSE",
                ]
            ],
            on=[
                "Signal",
                "FromWindow",
                "ToWindow",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    check_gain[
        "AbsDiff_vs_Phase3"
    ] = np.abs(
        check_gain[
            "Gain_Point"
        ]
        - check_gain[
            "Gain_NRMSE"
        ]
    )

    max_gain_diff = float(
        check_gain[
            "AbsDiff_vs_Phase3"
        ].max()
    )

    tolerance = 1e-10

    if (
        max_error_diff
        > tolerance
    ):
        raise RuntimeError(
            "Phase 4 Error point estimate "
            "与 Phase 3 不一致："
            f"max diff={max_error_diff}"
        )

    if (
        max_gain_diff
        > tolerance
    ):
        raise RuntimeError(
            "Phase 4 Gain point estimate "
            "与 Phase 3 不一致："
            f"max diff={max_gain_diff}"
        )

    if (
        error_uncertainty[
            "Bootstrap_Valid_N"
        ].min()
        < args.bootstrap * 0.95
    ):
        raise RuntimeError(
            "部分 Error bootstrap 有效比例 <95%。"
        )

    if (
        gain_uncertainty[
            "Bootstrap_Valid_N"
        ].min()
        < args.bootstrap * 0.95
    ):
        raise RuntimeError(
            "部分 Gain bootstrap 有效比例 <95%。"
        )

    error_uncertainty.to_csv(
        args.output_dir
        / "signal_error_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gain_uncertainty.to_csv(
        args.output_dir
        / "signal_gain_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stability_uncertainty.to_csv(
        args.output_dir
        / "signal_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        args.output_dir
        / "bootstrap_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    check_error.to_csv(
        args.output_dir
        / "phase3_error_regression_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    check_gain.to_csv(
        args.output_dir
        / "phase3_gain_regression_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "=" * 112
    )

    print(
        "MSOH V2 Phase 4 完成："
        "paired group bootstrap uncertainty"
    )

    print(
        "=" * 112
    )

    print(
        f"\nBootstrap B={args.bootstrap}, "
        f"base seed={args.seed}, "
        f"delta_G={args.gain_delta}, "
        f"group epsilon={args.group_epsilon}"
    )

    print(
        "\nError uncertainty:"
    )

    print(
        error_uncertainty[
            [
                "Signal",
                "Window",
                "Error_Point",
                "Error_CI_Low",
                "Error_CI_High",
                "Bootstrap_Valid_N",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nGain uncertainty:"
    )

    print(
        gain_uncertainty[
            [
                "Signal",
                "FromWindow",
                "ToWindow",
                "Gain_Point",
                "Gain_CI_Low",
                "Gain_CI_High",
                "P_Gain_GT_0",
                "P_Gain_GT_Delta",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nGroup stability:"
    )

    print(
        stability_uncertainty[
            [
                "Signal",
                "FromWindow",
                "ToWindow",
                "Stable_Group_Proportion",
                "Stability_CI_Low",
                "Stability_CI_High",
                "N_Groups_Compared",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRegression checks:"
    )

    print(
        f"  max |Error Phase4 - Phase3| = "
        f"{max_error_diff:.3e}"
    )

    print(
        f"  max |Gain Phase4 - Phase3|  = "
        f"{max_gain_diff:.3e}"
    )

    print(
        "\nPhase 4 checks: PASS"
    )

    print(
        f"\n输出目录: "
        f"{args.output_dir}"
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()
