from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config_v2 import (
    CLEAN_DATA_PATH,
    RESULTS_V2,
    SIGNAL_SPECS,
    DEFAULT_THRESHOLDS,
    WINDOWS,
    TARGET_DAYS,
)

from src_v2.signal_registry import (
    build_registry,
    validate_registry_against_data,
)

from src_v2.sparse_schedule import (
    infer_sparse_schedule,
)

from src_v2.nested_selection import (
    choose_model,
    predict_model,
    future_truth,
    InvalidModelFit,
)


# ============================================================
# Gate A: frozen model roles
# ============================================================

PRIMARY_EXCLUDED_MODELS = {
    "mixed_effects",
}

MODEL_ROLE_ROWS = [
    {
        "Model": "euclidean_feature",
        "Role": "primary_nested_candidate",
        "Reason": "Stable nonparametric trajectory-feature comparator.",
    },
    {
        "Model": "dtw_core3",
        "Role": "primary_nested_candidate",
        "Reason": "Stable Core3 trajectory-similarity comparator.",
    },
    {
        "Model": "dtw_all4",
        "Role": "primary_nested_candidate",
        "Reason": "Schedule-aware All4 trajectory-similarity comparator.",
    },
    {
        "Model": "ridge_direct",
        "Role": "primary_nested_candidate",
        "Reason": "Regularized direct predictor.",
    },
    {
        "Model": "mixed_effects",
        "Role": "sensitivity_only",
        "Reason": (
            "Excluded from formal nested selection after the pre-Phase-3 "
            "Gate A numerical-stability audit: repeated inner-CV fits showed "
            "non-convergence / optimizer fallback. Historical diagnostic "
            "results are retained, but Mixed Effects does not determine "
            "formal V2 outer predictions."
        ),
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MSOH V2 Phase 2 "
            "nested signal-specific prediction "
            "(Gate A frozen primary panel)"
        )
    )

    p.add_argument(
        "--input",
        type=Path,
        default=CLEAN_DATA_PATH,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            RESULTS_V2
            / "03_predictions"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input
    )

    registry = build_registry(
        SIGNAL_SPECS,
        DEFAULT_THRESHOLDS,
    )

    validate_registry_against_data(
        df,
        registry,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Freeze Gate A model-role metadata.
    pd.DataFrame(
        MODEL_ROLE_ROWS
    ).to_csv(
        args.output_dir
        / "model_role_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    defecation_spec = (
        registry[
            "Defecation"
        ]
    )

    defecation_schedule, _, _ = (
        infer_sparse_schedule(
            df,
            signal="Defecation",
            observed_col=(
                defecation_spec
                .observed_col
            ),
            candidate_windows=WINDOWS,
            target_days=TARGET_DAYS,
        )
    )

    groups = sorted(
        df[
            "Group"
        ].astype(str).unique()
    )

    prediction_rows = []
    selection_rows = []
    score_rows = []
    diagnostic_rows = []

    for signal, spec in (
        registry.items()
    ):
        if signal == "Defecation":
            windows = list(
                defecation_schedule
                .effective_windows
            )

            target_days = list(
                defecation_schedule
                .observed_target_days
            )

        else:
            windows = list(
                WINDOWS
            )

            target_days = list(
                TARGET_DAYS
            )

        # Gate A:
        # Mixed Effects is retained as a sensitivity/diagnostic model,
        # but is no longer eligible to determine formal nested predictions.
        candidates = [
            model
            for model in list(
                spec.predictor_panel
            )
            if model
            not in PRIMARY_EXCLUDED_MODELS
        ]

        if not candidates:
            raise RuntimeError(
                "Gate A filtering removed every primary candidate "
                f"for signal={signal}"
            )

        print(
            f"\n[{signal}] "
            f"windows={windows}, "
            f"targets={target_days}"
        )

        print(
            "primary candidate models="
            f"{candidates}"
        )

        excluded = [
            model
            for model in list(
                spec.predictor_panel
            )
            if model
            in PRIMARY_EXCLUDED_MODELS
        ]

        if excluded:
            print(
                "sensitivity-only models="
                f"{excluded}"
            )

        for window in windows:
            print(
                f"  Window D{window}"
            )

            for outer_group in groups:
                train_outer = df[
                    df[
                        "Group"
                    ].astype(str)
                    != outer_group
                ].copy()

                test_outer = df[
                    df[
                        "Group"
                    ].astype(str)
                    == outer_group
                ].copy()

                (
                    chosen,
                    score_table,
                    diagnostics,
                ) = choose_model(
                    train_outer,
                    signal=signal,
                    window=window,
                    target_days=target_days,
                    candidates=candidates,
                )

                selection_rows.append(
                    {
                        "Signal":
                            signal,
                        "Window":
                            int(window),
                        "OuterGroup":
                            outer_group,
                        "SelectedModel":
                            chosen,
                        "GateA_Panel":
                            "|".join(
                                candidates
                            ),
                    }
                )

                for _, row in (
                    score_table
                    .iterrows()
                ):
                    score_rows.append(
                        {
                            "Signal":
                                signal,
                            "Window":
                                int(
                                    window
                                ),
                            "OuterGroup":
                                outer_group,
                            "CandidateModel":
                                row[
                                    "Model"
                                ],
                            "Inner_NRMSE":
                                row[
                                    "Inner_NRMSE"
                                ],
                            "ValidCandidate":
                                row[
                                    "ValidCandidate"
                                ],
                        }
                    )

                for diagnostic in (
                    diagnostics
                ):
                    row = {
                        "Signal":
                            signal,
                        "Window":
                            int(
                                window
                            ),
                        "OuterGroup":
                            outer_group,
                    }

                    row.update(
                        diagnostic
                    )

                    diagnostic_rows.append(
                        row
                    )

                try:
                    (
                        predictions,
                        final_diagnostic,
                    ) = predict_model(
                        chosen,
                        train_outer,
                        test_outer,
                        signal,
                        window,
                        target_days,
                    )

                except InvalidModelFit as exc:
                    raise RuntimeError(
                        "Selected model became invalid "
                        "on outer fit: "
                        f"signal={signal}, "
                        f"window={window}, "
                        f"outer_group={outer_group}, "
                        f"model={chosen}, "
                        f"reason={exc}"
                    ) from exc

                final_row = {
                    "Signal":
                        signal,
                    "Window":
                        int(
                            window
                        ),
                    "OuterGroup":
                        outer_group,
                    "CandidateModel":
                        chosen,
                    "InnerValidationGroup":
                        "__OUTER_FIT__",
                    "Status":
                        "outer_fit",
                }

                final_row.update(
                    final_diagnostic
                )

                diagnostic_rows.append(
                    final_row
                )

                truth = future_truth(
                    test_outer,
                    outer_group,
                    signal,
                    target_days,
                )

                for _, row in (
                    truth.iterrows()
                ):
                    day = int(
                        row[
                            "Day"
                        ]
                    )

                    if (
                        day
                        not in predictions
                    ):
                        continue

                    prediction = (
                        predictions[
                            day
                        ]
                    )

                    if not np.isfinite(
                        prediction
                    ):
                        continue

                    prediction_rows.append(
                        {
                            "Signal":
                                signal,
                            "Window":
                                int(
                                    window
                                ),
                            "OuterGroup":
                                outer_group,
                            "TargetDay":
                                day,
                            "SelectedModel":
                                chosen,
                            "y_true":
                                float(
                                    row[
                                        "y_true"
                                    ]
                                ),
                            "y_pred":
                                float(
                                    prediction
                                ),
                            "error":
                                float(
                                    prediction
                                    - row[
                                        "y_true"
                                    ]
                                ),
                        }
                    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    selections = pd.DataFrame(
        selection_rows
    )

    scores = pd.DataFrame(
        score_rows
    )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    predictions.to_csv(
        args.output_dir
        / "nested_outer_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selections.to_csv(
        args.output_dir
        / "nested_model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scores.to_csv(
        args.output_dir
        / "nested_inner_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics.to_csv(
        args.output_dir
        / "nested_fit_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    frequency = (
        selections
        .groupby(
            [
                "Signal",
                "Window",
                "SelectedModel",
            ]
        )
        .size()
        .reset_index(
            name="N_Selected"
        )
        .sort_values(
            [
                "Signal",
                "Window",
                "N_Selected",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )

    frequency.to_csv(
        args.output_dir
        / "nested_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = (
        predictions
        .groupby(
            [
                "Signal",
                "Window",
            ]
        )
        .agg(
            N_Predictions=(
                "y_pred",
                "size",
            ),
            N_Groups=(
                "OuterGroup",
                "nunique",
            ),
            N_TargetDays=(
                "TargetDay",
                "nunique",
            ),
        )
        .reset_index()
    )

    audit.to_csv(
        args.output_dir
        / "prediction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Gate A audit checks.
    if (
        selections[
            "SelectedModel"
        ]
        .isin(
            PRIMARY_EXCLUDED_MODELS
        )
        .any()
    ):
        raise RuntimeError(
            "Gate A violation: sensitivity-only model "
            "was selected in the formal nested analysis."
        )

    expected_dense_predictions = (
        len(groups)
        * len(TARGET_DAYS)
    )

    for signal in [
        "Weight",
        "Status_Score",
        "Feeding",
    ]:
        part = audit[
            audit[
                "Signal"
            ] == signal
        ]

        if len(part) != len(
            WINDOWS
        ):
            raise RuntimeError(
                f"Incomplete audit rows for {signal}"
            )

        if not (
            part[
                "N_Predictions"
            ]
            == expected_dense_predictions
        ).all():
            raise RuntimeError(
                "Incomplete outer predictions "
                f"for {signal}"
            )

    defecation_part = audit[
        audit[
            "Signal"
        ] == "Defecation"
    ]

    if set(
        defecation_part[
            "Window"
        ].astype(int)
    ) != set(
        defecation_schedule
        .effective_windows
    ):
        raise RuntimeError(
            "Defecation window audit failed."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MSOH V2 Phase 2 完成：Gate A 正式冻结"
    )

    print(
        "=" * 100
    )

    print(
        "\nPrimary nested panel:"
    )

    print(
        "  Euclidean Feature / "
        "DTW Core3 / DTW All4 / Ridge Direct"
    )

    print(
        "Sensitivity only:"
    )

    print(
        "  Mixed Effects "
        "(does not determine formal outer predictions)"
    )

    print(
        "\nPrediction audit:"
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\nSelection frequency:"
    )

    print(
        frequency.to_string(
            index=False
        )
    )

    print(
        "\nGate A checks: PASS"
    )

    print(
        f"\n输出目录: "
        f"{args.output_dir}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
