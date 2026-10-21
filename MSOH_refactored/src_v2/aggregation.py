from __future__ import annotations

import numpy as np
import pandas as pd


CORE3_SIGNALS = [
    "Weight",
    "Status_Score",
    "Feeding",
]

EXTENDED_SIGNALS = [
    "Weight",
    "Status_Score",
    "Feeding",
    "Defecation",
]


def _finite_msoh(row: pd.Series) -> bool:
    return bool(
        str(row["Decision_Status"]) == "Identified"
        and pd.notna(
            row["Candidate_MSOH"]
        )
    )


def _limiting_reason(row: pd.Series) -> str:
    status = str(
        row["Decision_Status"]
    )

    signal = str(
        row["Signal"]
    )

    if status == "Adequate_not_saturated":
        return (
            f"{signal}: predictive adequacy reached, "
            "but persistent saturation not established"
        )

    if status == "Saturated_inadequate":
        return (
            f"{signal}: persistent saturation reached, "
            "but predictive adequacy not established"
        )

    if status == "Not_reached":
        return (
            f"{signal}: neither a finite signal MSOH nor "
            "the required joint adequacy+saturation state "
            "was established within the evaluated horizon"
        )

    if status == "Adequate_and_saturated_but_no_common_window":
        return (
            f"{signal}: adequacy and persistent saturation "
            "were observed, but not at a common eligible window"
        )

    if status == "Indeterminate_due_to_sparse_schedule":
        return (
            f"{signal}: indeterminate because the planned-sparse "
            "schedule does not support a finite MSOH determination"
        )

    if status == "Identified":
        return (
            f"{signal}: finite signal MSOH identified"
        )

    return (
        f"{signal}: {status}"
    )


def build_aggregation(
    signal_summary: pd.DataFrame,
):
    required = {
        "Signal",
        "Candidate_MSOH",
        "Decision_Status",
        "Adequacy_Onset",
        "Persistent_Saturation_Onset",
        "Last_Window",
        "Last_Window_Error",
        "Reason",
    }

    missing = (
        required
        - set(
            signal_summary.columns
        )
    )

    if missing:
        raise ValueError(
            "signal_decision_summary.csv 缺少字段："
            f"{sorted(missing)}"
        )

    signal_summary = (
        signal_summary
        .copy()
    )

    signal_summary[
        "Signal"
    ] = (
        signal_summary[
            "Signal"
        ].astype(str)
    )

    actual = set(
        signal_summary[
            "Signal"
        ]
    )

    needed = set(
        EXTENDED_SIGNALS
    )

    if not needed.issubset(
        actual
    ):
        raise ValueError(
            "缺少正式信号："
            f"{sorted(needed-actual)}"
        )

    audit_rows = []

    for signal in EXTENDED_SIGNALS:
        row = (
            signal_summary[
                signal_summary[
                    "Signal"
                ]
                == signal
            ]
            .iloc[0]
        )

        audit_rows.append(
            {
                "Signal":
                    signal,
                "Signal_Status":
                    str(
                        row[
                            "Decision_Status"
                        ]
                    ),
                "Signal_MSOH":
                    row[
                        "Candidate_MSOH"
                    ],
                "Finite_Signal_MSOH":
                    _finite_msoh(
                        row
                    ),
                "Adequacy_Onset":
                    row[
                        "Adequacy_Onset"
                    ],
                "Persistent_Saturation_Onset":
                    row[
                        "Persistent_Saturation_Onset"
                    ],
                "Last_Window":
                    int(
                        row[
                            "Last_Window"
                        ]
                    ),
                "Last_Window_Error":
                    float(
                        row[
                            "Last_Window_Error"
                        ]
                    ),
                "Aggregation_Note":
                    _limiting_reason(
                        row
                    ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    core = audit[
        audit[
            "Signal"
        ].isin(
            CORE3_SIGNALS
        )
    ].copy()

    core_all_finite = bool(
        core[
            "Finite_Signal_MSOH"
        ].all()
    )

    if core_all_finite:
        core_msoh = float(
            core[
                "Signal_MSOH"
            ].max()
        )

        core_status = (
            "Identified"
        )

        core_limiting = (
            core.loc[
                core[
                    "Signal_MSOH"
                ]
                == core_msoh,
                "Signal",
            ]
            .astype(str)
            .tolist()
        )

        core_reason = (
            "All Core3 signals have finite signal-specific MSOH; "
            "Core3 MSOH is the maximum of Weight, Status_Score, "
            "and Feeding signal MSOH values."
        )

    else:
        core_msoh = np.nan

        core_status = (
            "Not_identified_within_evaluated_horizon"
        )

        core_limiting = (
            core.loc[
                ~core[
                    "Finite_Signal_MSOH"
                ],
                "Signal",
            ]
            .astype(str)
            .tolist()
        )

        details = (
            core.loc[
                ~core[
                    "Finite_Signal_MSOH"
                ],
                "Aggregation_Note",
            ]
            .astype(str)
            .tolist()
        )

        core_reason = (
            "Core3 MSOH is not identified within the evaluated "
            "horizon because not all three core signals have a "
            "finite signal-specific MSOH. "
            + " | ".join(
                details
            )
        )

    def_row = (
        audit[
            audit[
                "Signal"
            ]
            == "Defecation"
        ]
        .iloc[0]
    )

    def_status = str(
        def_row[
            "Signal_Status"
        ]
    )

    def_finite = bool(
        def_row[
            "Finite_Signal_MSOH"
        ]
    )

    if (
        core_all_finite
        and def_finite
    ):
        extended_msoh = float(
            max(
                core_msoh,
                float(
                    def_row[
                        "Signal_MSOH"
                    ]
                ),
            )
        )

        extended_status = (
            "Identified"
        )

        extended_limiting = (
            audit.loc[
                audit[
                    "Signal_MSOH"
                ]
                == extended_msoh,
                "Signal",
            ]
            .astype(str)
            .tolist()
        )

        extended_reason = (
            "Core3 and Defecation all have finite signal-specific "
            "MSOH; Extended All4 MSOH is their maximum."
        )

    elif (
        def_status
        == "Indeterminate_due_to_sparse_schedule"
    ):
        extended_msoh = np.nan

        extended_status = (
            "Indeterminate_due_to_sparse_Defecation_schedule"
        )

        extended_limiting = (
            core_limiting
            + [
                "Defecation"
            ]
        )

        extended_reason = (
            "Extended All4 MSOH is indeterminate because Defecation "
            "cannot be assigned a finite signal-specific MSOH under "
            "the planned-sparse schedule. "
            "Core3 is reported separately and is not mechanically "
            "converted to No-MSOH by the sparse signal. "
            f"Core3 status: {core_status}."
        )

    else:
        extended_msoh = np.nan

        extended_status = (
            "Not_identified_within_evaluated_horizon"
        )

        extended_limiting = (
            core.loc[
                ~core[
                    "Finite_Signal_MSOH"
                ],
                "Signal",
            ]
            .astype(str)
            .tolist()
        )

        if not def_finite:
            extended_limiting.append(
                "Defecation"
            )

        extended_reason = (
            "Extended All4 MSOH is not identified within the "
            "evaluated horizon because at least one required signal "
            "does not have a finite signal-specific MSOH."
        )

    core_limiting = list(
        dict.fromkeys(
            core_limiting
        )
    )

    extended_limiting = list(
        dict.fromkeys(
            extended_limiting
        )
    )

    overall = pd.DataFrame(
        [
            {
                "Analysis_Level":
                    "Core3",
                "Signals":
                    "Weight|Status_Score|Feeding",
                "MSOH":
                    core_msoh,
                "Decision_Status":
                    core_status,
                "Limiting_Signals":
                    "|".join(
                        core_limiting
                    ),
                "Reason":
                    core_reason,
            },
            {
                "Analysis_Level":
                    "Extended_All4",
                "Signals":
                    "Weight|Status_Score|Feeding|Defecation",
                "MSOH":
                    extended_msoh,
                "Decision_Status":
                    extended_status,
                "Limiting_Signals":
                    "|".join(
                        extended_limiting
                    ),
                "Reason":
                    extended_reason,
            },
        ]
    )

    # Compact machine-readable one-row summary.
    final_summary = pd.DataFrame(
        [
            {
                "Core3_MSOH":
                    core_msoh,
                "Core3_Status":
                    core_status,
                "Core3_Limiting_Signals":
                    "|".join(
                        core_limiting
                    ),
                "Extended_All4_MSOH":
                    extended_msoh,
                "Extended_All4_Status":
                    extended_status,
                "Extended_All4_Limiting_Signals":
                    "|".join(
                        extended_limiting
                    ),
                "Defecation_Status":
                    def_status,
            }
        ]
    )

    return (
        audit,
        overall,
        final_summary,
    )
