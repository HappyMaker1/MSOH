from __future__ import annotations

import numpy as np
import pandas as pd


def _nrmse_from_group_counts(
    counts: np.ndarray,
    stat: dict,
) -> float:
    n = float(
        np.sum(
            counts
            * stat["n"]
        )
    )

    if n <= 1:
        return np.nan

    sy = float(
        np.sum(
            counts
            * stat["sy"]
        )
    )

    sy2 = float(
        np.sum(
            counts
            * stat["sy2"]
        )
    )

    sse = float(
        np.sum(
            counts
            * stat["sse"]
        )
    )

    var_num = (
        sy2
        - (
            sy * sy
            / n
        )
    )

    if var_num <= 0:
        return np.nan

    sd = np.sqrt(
        var_num
        / (
            n - 1.0
        )
    )

    if (
        not np.isfinite(
            sd
        )
        or np.isclose(
            sd,
            0.0,
        )
    ):
        return np.nan

    rmse = np.sqrt(
        sse / n
    )

    return float(
        rmse / sd
    )


def _build_signal_group_stats(
    part: pd.DataFrame,
):
    groups = sorted(
        part[
            "OuterGroup"
        ].astype(str).unique()
    )

    windows = sorted(
        part[
            "Window"
        ].astype(int).unique()
    )

    group_index = {
        group: i
        for i, group
        in enumerate(groups)
    }

    stats = {}

    for window in windows:
        q = part[
            part[
                "Window"
            ].astype(int)
            == int(window)
        ]

        n = np.zeros(
            len(groups),
            dtype=float,
        )

        sy = np.zeros(
            len(groups),
            dtype=float,
        )

        sy2 = np.zeros(
            len(groups),
            dtype=float,
        )

        sse = np.zeros(
            len(groups),
            dtype=float,
        )

        for group, gp in q.groupby(
            "OuterGroup",
            sort=False,
        ):
            idx = group_index[
                str(group)
            ]

            y = gp[
                "y_true"
            ].to_numpy(
                float
            )

            p = gp[
                "y_pred"
            ].to_numpy(
                float
            )

            n[idx] = len(y)
            sy[idx] = float(
                np.sum(y)
            )
            sy2[idx] = float(
                np.sum(
                    y * y
                )
            )
            sse[idx] = float(
                np.sum(
                    (
                        p - y
                    )
                    ** 2
                )
            )

        stats[
            int(window)
        ] = {
            "n": n,
            "sy": sy,
            "sy2": sy2,
            "sse": sse,
        }

    return (
        groups,
        windows,
        stats,
    )


def _full_sample_errors(
    groups,
    windows,
    stats,
):
    counts = np.ones(
        len(groups),
        dtype=float,
    )

    return np.asarray(
        [
            _nrmse_from_group_counts(
                counts,
                stats[
                    int(window)
                ],
            )
            for window in windows
        ],
        dtype=float,
    )


def _bootstrap_error_matrix(
    groups,
    windows,
    stats,
    counts_matrix,
):
    out = np.empty(
        (
            counts_matrix.shape[0],
            len(windows),
        ),
        dtype=float,
    )

    for bi, counts in enumerate(
        counts_matrix
    ):
        for wi, window in enumerate(
            windows
        ):
            out[
                bi,
                wi,
            ] = (
                _nrmse_from_group_counts(
                    counts,
                    stats[
                        int(window)
                    ],
                )
            )

    return out


def _group_error_matrix(
    part: pd.DataFrame,
    groups,
    windows,
):
    """
    Group-level NRMSE-like error.

    The denominator is the full-sample SD(y_true)
    for each signal/window, matching the formal
    signal-level NRMSE scale.
    """
    matrix = np.full(
        (
            len(groups),
            len(windows),
        ),
        np.nan,
        dtype=float,
    )

    group_index = {
        group: i
        for i, group
        in enumerate(groups)
    }

    for wi, window in enumerate(
        windows
    ):
        q = part[
            part[
                "Window"
            ].astype(int)
            == int(window)
        ]

        y_all = q[
            "y_true"
        ].to_numpy(
            float
        )

        if len(y_all) < 2:
            continue

        sd = float(
            np.std(
                y_all,
                ddof=1,
            )
        )

        if (
            not np.isfinite(sd)
            or np.isclose(
                sd,
                0.0,
            )
        ):
            continue

        for group, gp in q.groupby(
            "OuterGroup",
            sort=False,
        ):
            residual = (
                gp[
                    "y_pred"
                ].to_numpy(
                    float
                )
                - gp[
                    "y_true"
                ].to_numpy(
                    float
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        residual
                        * residual
                    )
                )
            )

            matrix[
                group_index[
                    str(group)
                ],
                wi,
            ] = (
                rmse / sd
            )

    return matrix


def _percentile_ci(
    values,
    low=2.5,
    high=97.5,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            0,
        )

    lo, hi = np.percentile(
        values,
        [
            low,
            high,
        ],
    )

    return (
        float(lo),
        float(hi),
        int(
            len(values)
        ),
    )


def bootstrap_signal(
    signal: str,
    part: pd.DataFrame,
    bootstrap_b: int,
    seed: int,
    gain_delta: float,
    group_epsilon: float,
):
    groups, windows, stats = (
        _build_signal_group_stats(
            part
        )
    )

    n_groups = len(
        groups
    )

    if n_groups < 2:
        raise ValueError(
            f"{signal}: group 数量不足。"
        )

    rng = np.random.default_rng(
        seed
    )

    # One shared group resample is reused across every
    # window for this signal. This is what makes Gain paired.
    counts_matrix = rng.multinomial(
        n_groups,
        np.repeat(
            1.0 / n_groups,
            n_groups,
        ),
        size=bootstrap_b,
    )

    point_errors = (
        _full_sample_errors(
            groups,
            windows,
            stats,
        )
    )

    error_boot = (
        _bootstrap_error_matrix(
            groups,
            windows,
            stats,
            counts_matrix,
        )
    )

    error_rows = []

    for wi, window in enumerate(
        windows
    ):
        values = error_boot[
            :,
            wi,
        ]

        lo, hi, n_valid = (
            _percentile_ci(
                values
            )
        )

        error_rows.append(
            {
                "Signal":
                    signal,
                "Window":
                    int(window),
                "Error_Point":
                    float(
                        point_errors[
                            wi
                        ]
                    ),
                "Error_CI_Low":
                    lo,
                "Error_CI_High":
                    hi,
                "Bootstrap_Valid_N":
                    n_valid,
                "N_Groups":
                    n_groups,
                "Bootstrap_B":
                    int(
                        bootstrap_b
                    ),
                "Seed":
                    int(seed),
            }
        )

    gain_rows = []

    for i in range(
        len(windows)
        - 1
    ):
        from_window = int(
            windows[i]
        )

        to_window = int(
            windows[
                i + 1
            ]
        )

        values = (
            error_boot[
                :,
                i,
            ]
            - error_boot[
                :,
                i + 1,
            ]
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        lo, hi, n_valid = (
            _percentile_ci(
                finite
            )
        )

        point_gain = float(
            point_errors[
                i
            ]
            - point_errors[
                i + 1
            ]
        )

        gain_rows.append(
            {
                "Signal":
                    signal,
                "FromWindow":
                    from_window,
                "ToWindow":
                    to_window,
                "WindowIncrement":
                    int(
                        to_window
                        - from_window
                    ),
                "TransitionType":
                    (
                        "adjacent_day"
                        if (
                            to_window
                            - from_window
                        ) == 1
                        else "sparse_observation_update"
                    ),
                "Gain_Point":
                    point_gain,
                "Gain_CI_Low":
                    lo,
                "Gain_CI_High":
                    hi,
                "P_Gain_GT_0":
                    float(
                        np.mean(
                            finite > 0.0
                        )
                    )
                    if len(
                        finite
                    )
                    else np.nan,
                "Gain_Delta":
                    float(
                        gain_delta
                    ),
                "P_Gain_GT_Delta":
                    float(
                        np.mean(
                            finite
                            > gain_delta
                        )
                    )
                    if len(
                        finite
                    )
                    else np.nan,
                "P_Gain_LE_Delta":
                    float(
                        np.mean(
                            finite
                            <= gain_delta
                        )
                    )
                    if len(
                        finite
                    )
                    else np.nan,
                "Bootstrap_Valid_N":
                    n_valid,
                "N_Groups":
                    n_groups,
                "Bootstrap_B":
                    int(
                        bootstrap_b
                    ),
                "Seed":
                    int(seed),
            }
        )

    # --------------------------------------------------------
    # Group stability
    # --------------------------------------------------------
    group_error = _group_error_matrix(
        part,
        groups,
        windows,
    )

    stability_rows = []

    for i in range(
        len(windows)
        - 1
    ):
        from_window = int(
            windows[i]
        )

        to_window = int(
            windows[
                i + 1
            ]
        )

        left = group_error[
            :,
            i,
        ]

        right = group_error[
            :,
            i + 1,
        ]

        valid = (
            np.isfinite(left)
            & np.isfinite(right)
        )

        stable_indicator = (
            np.abs(
                right
                - left
            )
            <= group_epsilon
        )

        point_stability = (
            float(
                np.mean(
                    stable_indicator[
                        valid
                    ]
                )
            )
            if np.any(
                valid
            )
            else np.nan
        )

        # Bootstrap CI is descriptive. Phase 5 will keep the
        # pre-specified decision rule based on point stability.
        stability_boot = []

        for counts in (
            counts_matrix
        ):
            effective_counts = (
                counts[
                    valid
                ]
            )

            denom = float(
                effective_counts.sum()
            )

            if denom <= 0:
                stability_boot.append(
                    np.nan
                )
                continue

            stability_boot.append(
                float(
                    np.sum(
                        effective_counts
                        * stable_indicator[
                            valid
                        ].astype(
                            float
                        )
                    )
                    / denom
                )
            )

        lo, hi, n_valid_boot = (
            _percentile_ci(
                stability_boot
            )
        )

        stability_rows.append(
            {
                "Signal":
                    signal,
                "FromWindow":
                    from_window,
                "ToWindow":
                    to_window,
                "WindowIncrement":
                    int(
                        to_window
                        - from_window
                    ),
                "TransitionType":
                    (
                        "adjacent_day"
                        if (
                            to_window
                            - from_window
                        ) == 1
                        else "sparse_observation_update"
                    ),
                "Stable_Group_Proportion":
                    point_stability,
                "Stability_CI_Low":
                    lo,
                "Stability_CI_High":
                    hi,
                "Group_Epsilon":
                    float(
                        group_epsilon
                    ),
                "N_Groups_Compared":
                    int(
                        valid.sum()
                    ),
                "Bootstrap_Valid_N":
                    int(
                        n_valid_boot
                    ),
                "Bootstrap_B":
                    int(
                        bootstrap_b
                    ),
                "Seed":
                    int(seed),
            }
        )

    return (
        pd.DataFrame(
            error_rows
        ),
        pd.DataFrame(
            gain_rows
        ),
        pd.DataFrame(
            stability_rows
        ),
        {
            "Signal":
                signal,
            "N_Groups":
                n_groups,
            "Windows":
                ",".join(
                    str(
                        int(w)
                    )
                    for w in windows
                ),
            "Bootstrap_B":
                int(
                    bootstrap_b
                ),
            "Seed":
                int(seed),
        },
    )


def run_paired_bootstrap(
    predictions: pd.DataFrame,
    bootstrap_b: int,
    seed: int,
    gain_delta: float,
    group_epsilon: float,
):
    all_error = []
    all_gain = []
    all_stability = []
    audit_rows = []

    signals = sorted(
        predictions[
            "Signal"
        ].astype(str).unique()
    )

    for si, signal in enumerate(
        signals
    ):
        part = predictions[
            predictions[
                "Signal"
            ].astype(str)
            == signal
        ].copy()

        signal_seed = (
            int(seed)
            + si * 1009
        )

        (
            error,
            gain,
            stability,
            audit,
        ) = bootstrap_signal(
            signal=signal,
            part=part,
            bootstrap_b=bootstrap_b,
            seed=signal_seed,
            gain_delta=gain_delta,
            group_epsilon=group_epsilon,
        )

        all_error.append(
            error
        )

        all_gain.append(
            gain
        )

        all_stability.append(
            stability
        )

        audit_rows.append(
            audit
        )

    return (
        pd.concat(
            all_error,
            ignore_index=True,
        ),
        pd.concat(
            all_gain,
            ignore_index=True,
        ),
        pd.concat(
            all_stability,
            ignore_index=True,
        ),
        pd.DataFrame(
            audit_rows
        ),
    )
