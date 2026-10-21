from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:
    ConvergenceWarning = Warning


MODEL_PRIORITY = {
    "mixed_effects": 0,
    "ridge_direct": 1,
    "dtw_core3": 2,
    "dtw_all4": 3,
    "euclidean_feature": 4,
}


class InvalidModelFit(RuntimeError):
    """Raised when a candidate model fit is not reliable enough to score."""


def safe_slope(days, vals):
    days = np.asarray(days, float)
    vals = np.asarray(vals, float)
    mask = np.isfinite(days) & np.isfinite(vals)
    days = days[mask]
    vals = vals[mask]

    if len(vals) < 2:
        return np.nan

    if np.allclose(days, days[0]):
        return np.nan

    return float(np.polyfit(days, vals, 1)[0])


def make_features(df, groups, window, signals):
    rows = []

    for group in groups:
        part = df[
            (df["Group"].astype(str) == str(group))
            & (df["Day"] <= window)
        ].sort_values("Day")

        row = {"Group": str(group)}

        for signal in signals:
            observed_col = f"{signal}_observed"

            if observed_col in part.columns:
                q = part[part[observed_col] == 1]
            else:
                q = part[part[signal].notna()]

            values = q[signal].to_numpy(float)
            days = q["Day"].to_numpy(float)

            row[f"{signal}_last"] = (
                float(values[-1])
                if len(values)
                else np.nan
            )

            row[f"{signal}_slope"] = safe_slope(
                days,
                values,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def future_truth(df, group, signal, target_days):
    part = df[
        (df["Group"].astype(str) == str(group))
        & (df["Day"].isin(target_days))
    ].copy()

    observed_col = f"{signal}_observed"

    if observed_col in part.columns:
        part = part[
            part[observed_col] == 1
        ]

    part = part[
        part[signal].notna()
    ]

    return (
        part[["Day", signal]]
        .rename(
            columns={
                signal: "y_true",
            }
        )
    )


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    denom = np.std(
        y_true,
        ddof=1,
    )

    if denom <= 1e-12:
        return np.nan

    rmse = np.sqrt(
        np.mean(
            (y_pred - y_true) ** 2
        )
    )

    return float(
        rmse / denom
    )


def _target_values(
    train_df,
    signal,
    day,
):
    q = train_df[
        train_df["Day"] == day
    ].copy()

    observed_col = (
        f"{signal}_observed"
    )

    if observed_col in q.columns:
        q = q[
            q[observed_col] == 1
        ]

    return (
        q[["Group", signal]]
        .dropna()
    )


def ridge_predict(
    train_df,
    test_df,
    signal,
    window,
    target_days,
):
    train_groups = sorted(
        train_df[
            "Group"
        ].astype(str).unique()
    )

    test_group = str(
        test_df[
            "Group"
        ].astype(str).iloc[0]
    )

    feature_signals = [
        "Weight",
        "Status_Score",
        "Feeding",
        "Defecation",
    ]

    feature_train = make_features(
        train_df,
        train_groups,
        window,
        feature_signals,
    )

    feature_test = make_features(
        test_df,
        [test_group],
        window,
        feature_signals,
    )

    feature_cols = [
        c
        for c in feature_train.columns
        if c != "Group"
    ]

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(
            feature_train[
                feature_cols
            ]
        )
    )

    X_test = scaler.transform(
        imputer.transform(
            feature_test[
                feature_cols
            ]
        )
    )

    group_index = {
        group: i
        for i, group
        in enumerate(train_groups)
    }

    predictions = {}

    for day in target_days:
        target = _target_values(
            train_df,
            signal,
            day,
        )

        keep = []
        y = []

        for _, row in target.iterrows():
            group = str(
                row["Group"]
            )

            if group in group_index:
                keep.append(
                    group_index[group]
                )
                y.append(
                    float(
                        row[signal]
                    )
                )

        if len(y) < 3:
            predictions[day] = np.nan
            continue

        model = Ridge(
            alpha=1.0
        )

        model.fit(
            X_train[keep],
            np.asarray(y),
        )

        predictions[day] = float(
            model.predict(
                X_test
            )[0]
        )

    return predictions, {
        "Model": "ridge_direct",
        "FitValid": True,
        "Converged": True,
        "FallbackUsed": False,
        "Optimizer": "",
        "WarningCount": 0,
        "WarningTypes": "",
        "Message": "",
    }


def euclidean_predict(
    train_df,
    test_df,
    signal,
    window,
    target_days,
    k=2,
):
    train_groups = sorted(
        train_df[
            "Group"
        ].astype(str).unique()
    )

    test_group = str(
        test_df[
            "Group"
        ].astype(str).iloc[0]
    )

    feature_signals = [
        "Weight",
        "Status_Score",
        "Feeding",
        "Defecation",
    ]

    feature_train = make_features(
        train_df,
        train_groups,
        window,
        feature_signals,
    )

    feature_test = make_features(
        test_df,
        [test_group],
        window,
        feature_signals,
    )

    feature_cols = [
        c
        for c in feature_train.columns
        if c != "Group"
    ]

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(
            feature_train[
                feature_cols
            ]
        )
    )

    X_test = scaler.transform(
        imputer.transform(
            feature_test[
                feature_cols
            ]
        )
    )

    distances = np.linalg.norm(
        X_train
        - X_test[0],
        axis=1,
    )

    order = np.argsort(
        distances
    )[:min(k, len(distances))]

    weights = (
        1.0
        / (
            distances[order]
            + 1e-8
        )
    )

    weights = (
        weights
        / weights.sum()
    )

    predictions = {}

    for day in target_days:
        values = []
        used_weights = []

        for idx, weight in zip(
            order,
            weights,
        ):
            group = train_groups[
                idx
            ]

            q = _target_values(
                train_df[
                    train_df[
                        "Group"
                    ].astype(str)
                    == group
                ],
                signal,
                day,
            )

            if len(q):
                values.append(
                    float(
                        q.iloc[0][
                            signal
                        ]
                    )
                )

                used_weights.append(
                    float(weight)
                )

        if not values:
            predictions[day] = np.nan
            continue

        used_weights = np.asarray(
            used_weights
        )

        used_weights = (
            used_weights
            / used_weights.sum()
        )

        predictions[day] = float(
            np.sum(
                used_weights
                * np.asarray(
                    values
                )
            )
        )

    return predictions, {
        "Model": "euclidean_feature",
        "FitValid": True,
        "Converged": True,
        "FallbackUsed": False,
        "Optimizer": "",
        "WarningCount": 0,
        "WarningTypes": "",
        "Message": "",
    }


def simple_dtw(
    a,
    b,
    radius=1,
):
    a = np.asarray(
        a,
        float,
    )

    b = np.asarray(
        b,
        float,
    )

    n = len(a)
    m = len(b)

    if n == 0 or m == 0:
        return np.inf

    dp = np.full(
        (n + 1, m + 1),
        np.inf,
    )

    path_len = np.zeros(
        (n + 1, m + 1),
        int,
    )

    dp[0, 0] = 0.0

    for i in range(
        1,
        n + 1,
    ):
        j0 = max(
            1,
            i - radius,
        )

        j1 = min(
            m,
            i + radius,
        )

        for j in range(
            j0,
            j1 + 1,
        ):
            cost = float(
                np.linalg.norm(
                    a[i - 1]
                    - b[j - 1]
                )
            )

            choices = [
                (
                    dp[i - 1, j],
                    path_len[
                        i - 1,
                        j,
                    ],
                ),
                (
                    dp[i, j - 1],
                    path_len[
                        i,
                        j - 1,
                    ],
                ),
                (
                    dp[
                        i - 1,
                        j - 1,
                    ],
                    path_len[
                        i - 1,
                        j - 1,
                    ],
                ),
            ]

            best = min(
                choices,
                key=lambda x: x[0],
            )

            dp[i, j] = (
                cost
                + best[0]
            )

            path_len[i, j] = (
                best[1]
                + 1
            )

    if (
        not np.isfinite(
            dp[n, m]
        )
        or path_len[
            n,
            m,
        ] == 0
    ):
        return np.inf

    return float(
        dp[n, m]
        / path_len[
            n,
            m,
        ]
    )


def _core_stats(
    train_df,
    window,
):
    stats = {}

    for signal in [
        "Weight",
        "Status_Score",
        "Feeding",
    ]:
        q = train_df[
            train_df[
                "Day"
            ] <= window
        ].copy()

        observed_col = (
            f"{signal}_observed"
        )

        if observed_col in q.columns:
            q = q[
                q[
                    observed_col
                ] == 1
            ]

        values = q[
            signal
        ].dropna().to_numpy(
            float
        )

        mean = float(
            values.mean()
        )

        sd = float(
            values.std(
                ddof=1
            )
        )

        if sd <= 1e-12:
            sd = 1.0

        stats[
            signal
        ] = (
            mean,
            sd,
        )

    return stats


def _core_sequence(
    df,
    group,
    window,
    stats,
):
    rows = []

    for day in range(
        1,
        window + 1,
    ):
        q = df[
            (
                df[
                    "Group"
                ].astype(str)
                == str(group)
            )
            & (
                df[
                    "Day"
                ] == day
            )
        ]

        if q.empty:
            continue

        vector = []
        ok = True

        for signal in [
            "Weight",
            "Status_Score",
            "Feeding",
        ]:
            observed_col = (
                f"{signal}_observed"
            )

            if (
                observed_col
                in q.columns
                and int(
                    q.iloc[0][
                        observed_col
                    ]
                ) != 1
            ):
                ok = False
                break

            value = q.iloc[0][
                signal
            ]

            if pd.isna(
                value
            ):
                ok = False
                break

            mean, sd = stats[
                signal
            ]

            vector.append(
                (
                    float(value)
                    - mean
                )
                / sd
            )

        if ok:
            rows.append(
                vector
            )

    return np.asarray(
        rows,
        float,
    )


def dtw_predict(
    train_df,
    test_df,
    signal,
    window,
    target_days,
    all4=False,
    k=2,
):
    train_groups = sorted(
        train_df[
            "Group"
        ].astype(str).unique()
    )

    test_group = str(
        test_df[
            "Group"
        ].astype(str).iloc[0]
    )

    stats = _core_stats(
        train_df,
        window,
    )

    test_core = (
        _core_sequence(
            test_df,
            test_group,
            window,
            stats,
        )
    )

    defecation_train = train_df[
        (
            train_df[
                "Day"
            ] <= window
        )
        & (
            train_df[
                "Defecation_observed"
            ] == 1
        )
    ]

    defecation_values = (
        defecation_train[
            "Defecation"
        ]
        .dropna()
        .to_numpy(
            float
        )
    )

    defecation_mean = (
        float(
            defecation_values.mean()
        )
        if len(
            defecation_values
        )
        else 0.0
    )

    defecation_sd = (
        float(
            defecation_values.std(
                ddof=1
            )
        )
        if len(
            defecation_values
        ) > 1
        else 1.0
    )

    if defecation_sd <= 1e-12:
        defecation_sd = 1.0

    def sparse_sequence(
        data,
        group,
    ):
        q = data[
            (
                data[
                    "Group"
                ].astype(str)
                == str(group)
            )
            & (
                data[
                    "Day"
                ] <= window
            )
            & (
                data[
                    "Defecation_observed"
                ] == 1
            )
        ]

        return (
            (
                q[
                    "Defecation"
                ]
                .dropna()
                .to_numpy(
                    float
                )
                - defecation_mean
            )
            / defecation_sd
        ).reshape(
            -1,
            1,
        )

    test_def = sparse_sequence(
        test_df,
        test_group,
    )

    distances = []

    for group in train_groups:
        train_core = (
            _core_sequence(
                train_df,
                group,
                window,
                stats,
            )
        )

        core_distance = simple_dtw(
            test_core,
            train_core,
            radius=1,
        )

        if all4:
            train_def = (
                sparse_sequence(
                    train_df,
                    group,
                )
            )

            if (
                len(test_def) == 1
                and len(
                    train_def
                ) == 1
            ):
                def_distance = abs(
                    float(
                        test_def[
                            0,
                            0,
                        ]
                        - train_def[
                            0,
                            0,
                        ]
                    )
                )

            elif (
                len(test_def)
                and len(train_def)
            ):
                def_distance = (
                    simple_dtw(
                        test_def,
                        train_def,
                        radius=1,
                    )
                )

            else:
                def_distance = 0.0

            distance = math.sqrt(
                core_distance
                * core_distance
                + def_distance
                * def_distance
            )

        else:
            distance = (
                core_distance
            )

        distances.append(
            distance
        )

    distances = np.asarray(
        distances
    )

    order = np.argsort(
        distances
    )[:min(k, len(distances))]

    weights = (
        1.0
        / (
            distances[order]
            + 1e-8
        )
    )

    weights = (
        weights
        / weights.sum()
    )

    predictions = {}

    for day in target_days:
        values = []
        used_weights = []

        for idx, weight in zip(
            order,
            weights,
        ):
            group = (
                train_groups[
                    idx
                ]
            )

            q = _target_values(
                train_df[
                    train_df[
                        "Group"
                    ].astype(str)
                    == group
                ],
                signal,
                day,
            )

            if len(q):
                values.append(
                    float(
                        q.iloc[0][
                            signal
                        ]
                    )
                )

                used_weights.append(
                    float(
                        weight
                    )
                )

        if not values:
            predictions[
                day
            ] = np.nan
            continue

        used_weights = np.asarray(
            used_weights
        )

        used_weights = (
            used_weights
            / used_weights.sum()
        )

        predictions[
            day
        ] = float(
            np.sum(
                used_weights
                * np.asarray(
                    values
                )
            )
        )

    return predictions, {
        "Model": (
            "dtw_all4"
            if all4
            else "dtw_core3"
        ),
        "FitValid": True,
        "Converged": True,
        "FallbackUsed": False,
        "Optimizer": "",
        "WarningCount": 0,
        "WarningTypes": "",
        "Message": "",
    }


def mixed_predict(
    train_df,
    test_df,
    signal,
    window,
    target_days,
):
    import statsmodels.api as sm

    train = train_df.copy()

    observed_col = (
        f"{signal}_observed"
    )

    if observed_col in train.columns:
        train = train[
            train[
                observed_col
            ] == 1
        ]

    train = train[
        train[
            signal
        ].notna()
    ]

    if len(train) < 10:
        raise InvalidModelFit(
            "MixedLM training rows < 10"
        )

    t = train[
        "Day"
    ].to_numpy(
        float
    )

    X = np.column_stack(
        [
            np.ones(
                len(t)
            ),
            t,
            t * t,
        ]
    )

    Z = np.column_stack(
        [
            np.ones(
                len(t)
            ),
            t - 4.5,
        ]
    )

    y = train[
        signal
    ].to_numpy(
        float
    )

    groups = train[
        "Group"
    ].astype(
        str
    ).to_numpy()

    optimizers = [
        "lbfgs",
        "powell",
    ]

    attempts = []

    fit = None
    used_optimizer = None
    captured_warnings = []

    for optimizer in optimizers:
        try:
            with warnings.catch_warnings(
                record=True
            ) as warning_list:
                warnings.simplefilter(
                    "always"
                )

                model = sm.MixedLM(
                    y,
                    X,
                    groups=groups,
                    exog_re=Z,
                )

                candidate = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=300,
                    disp=False,
                )

            warning_names = [
                type(w.message).__name__
                for w in warning_list
            ]

            warning_text = [
                str(w.message)
                for w in warning_list
            ]

            attempts.append(
                {
                    "optimizer":
                        optimizer,
                    "converged":
                        bool(
                            getattr(
                                candidate,
                                "converged",
                                False,
                            )
                        ),
                    "warnings":
                        warning_names,
                }
            )

            bad_convergence_warning = any(
                isinstance(
                    w.message,
                    ConvergenceWarning,
                )
                for w in warning_list
            )

            if (
                bool(
                    getattr(
                        candidate,
                        "converged",
                        False,
                    )
                )
                and not bad_convergence_warning
            ):
                fit = candidate
                used_optimizer = optimizer
                captured_warnings = (
                    warning_text
                )
                break

        except Exception as exc:
            attempts.append(
                {
                    "optimizer":
                        optimizer,
                    "converged":
                        False,
                    "exception":
                        repr(exc),
                }
            )

    if fit is None:
        raise InvalidModelFit(
            "MixedLM invalid after "
            f"optimizers={attempts}"
        )

    beta = np.asarray(
        fit.fe_params,
        float,
    )

    covariance_re = np.asarray(
        fit.cov_re,
        float,
    )

    sigma2 = float(
        fit.scale
    )

    if (
        not np.all(
            np.isfinite(
                beta
            )
        )
        or not np.all(
            np.isfinite(
                covariance_re
            )
        )
        or not np.isfinite(
            sigma2
        )
        or sigma2 <= 0
    ):
        raise InvalidModelFit(
            "MixedLM returned non-finite "
            "or invalid variance estimates"
        )

    test = test_df[
        test_df[
            "Day"
        ] <= window
    ].copy()

    if observed_col in test.columns:
        test = test[
            test[
                observed_col
            ] == 1
        ]

    test = test[
        test[
            signal
        ].notna()
    ]

    if len(test):
        tt = test[
            "Day"
        ].to_numpy(
            float
        )

        X_test = np.column_stack(
            [
                np.ones(
                    len(tt)
                ),
                tt,
                tt * tt,
            ]
        )

        Z_test = np.column_stack(
            [
                np.ones(
                    len(tt)
                ),
                tt - 4.5,
            ]
        )

        residual = (
            test[
                signal
            ].to_numpy(
                float
            )
            - X_test @ beta
        )

        V = (
            Z_test
            @ covariance_re
            @ Z_test.T
            + sigma2
            * np.eye(
                len(test)
            )
        )

        random_effect = (
            covariance_re
            @ Z_test.T
            @ np.linalg.pinv(
                V
            )
            @ residual
        )

    else:
        random_effect = (
            np.zeros(2)
        )

    predictions = {}

    for day in target_days:
        x_future = np.asarray(
            [
                1.0,
                float(day),
                float(
                    day * day
                ),
            ]
        )

        z_future = np.asarray(
            [
                1.0,
                float(day)
                - 4.5,
            ]
        )

        predictions[
            day
        ] = float(
            x_future @ beta
            + z_future
            @ random_effect
        )

    return predictions, {
        "Model": "mixed_effects",
        "FitValid": True,
        "Converged": True,
        "FallbackUsed": (
            used_optimizer
            != optimizers[0]
        ),
        "Optimizer": (
            used_optimizer
            or ""
        ),
        "WarningCount": len(
            captured_warnings
        ),
        "WarningTypes": ";".join(
            sorted(
                set(
                    captured_warnings
                )
            )
        ),
        "Message": "",
    }


def predict_model(
    name,
    train_df,
    test_df,
    signal,
    window,
    target_days,
):
    if name == "ridge_direct":
        return ridge_predict(
            train_df,
            test_df,
            signal,
            window,
            target_days,
        )

    if name == "euclidean_feature":
        return euclidean_predict(
            train_df,
            test_df,
            signal,
            window,
            target_days,
        )

    if name == "dtw_core3":
        return dtw_predict(
            train_df,
            test_df,
            signal,
            window,
            target_days,
            all4=False,
        )

    if name == "dtw_all4":
        return dtw_predict(
            train_df,
            test_df,
            signal,
            window,
            target_days,
            all4=True,
        )

    if name == "mixed_effects":
        return mixed_predict(
            train_df,
            test_df,
            signal,
            window,
            target_days,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


def inner_cv_score(
    train_outer,
    signal,
    window,
    target_days,
    model,
):
    y_true_all = []
    y_pred_all = []
    diagnostics = []

    groups = sorted(
        train_outer[
            "Group"
        ].astype(str).unique()
    )

    for validation_group in groups:
        train_inner = train_outer[
            train_outer[
                "Group"
            ].astype(str)
            != validation_group
        ].copy()

        validation = train_outer[
            train_outer[
                "Group"
            ].astype(str)
            == validation_group
        ].copy()

        try:
            predictions, diagnostic = (
                predict_model(
                    model,
                    train_inner,
                    validation,
                    signal,
                    window,
                    target_days,
                )
            )

            diagnostic = dict(
                diagnostic
            )

            diagnostic.update(
                {
                    "InnerValidationGroup":
                        validation_group,
                    "Status":
                        "valid",
                }
            )

        except InvalidModelFit as exc:
            diagnostics.append(
                {
                    "Model":
                        model,
                    "InnerValidationGroup":
                        validation_group,
                    "FitValid":
                        False,
                    "Converged":
                        False,
                    "FallbackUsed":
                        False,
                    "Optimizer":
                        "",
                    "WarningCount":
                        0,
                    "WarningTypes":
                        "",
                    "Message":
                        str(exc),
                    "Status":
                        "invalid_fit",
                }
            )
            continue

        except Exception as exc:
            diagnostics.append(
                {
                    "Model":
                        model,
                    "InnerValidationGroup":
                        validation_group,
                    "FitValid":
                        False,
                    "Converged":
                        False,
                    "FallbackUsed":
                        False,
                    "Optimizer":
                        "",
                    "WarningCount":
                        0,
                    "WarningTypes":
                        "",
                    "Message":
                        repr(exc),
                    "Status":
                        "exception",
                }
            )
            continue

        diagnostics.append(
            diagnostic
        )

        truth = future_truth(
            validation,
            validation_group,
            signal,
            target_days,
        )

        for _, row in truth.iterrows():
            day = int(
                row["Day"]
            )

            if (
                day in predictions
                and np.isfinite(
                    predictions[
                        day
                    ]
                )
            ):
                y_true_all.append(
                    float(
                        row[
                            "y_true"
                        ]
                    )
                )

                y_pred_all.append(
                    float(
                        predictions[
                            day
                        ]
                    )
                )

    valid_folds = sum(
        1
        for d in diagnostics
        if d.get(
            "Status"
        ) == "valid"
    )

    total_folds = len(
        groups
    )

    # Formal rule:
    # candidate must produce a valid fit for every inner fold.
    # Otherwise its score is invalid for this outer fold.
    if (
        valid_folds
        != total_folds
    ):
        return (
            np.nan,
            diagnostics,
        )

    return (
        nrmse(
            y_true_all,
            y_pred_all,
        ),
        diagnostics,
    )


def choose_model(
    train_outer,
    signal,
    window,
    target_days,
    candidates,
):
    score_rows = []
    diagnostics = []

    for model in candidates:
        score, model_diagnostics = (
            inner_cv_score(
                train_outer,
                signal,
                window,
                target_days,
                model,
            )
        )

        score_rows.append(
            {
                "Model":
                    model,
                "Inner_NRMSE":
                    score,
                "ValidCandidate":
                    bool(
                        np.isfinite(
                            score
                        )
                    ),
            }
        )

        for diagnostic in (
            model_diagnostics
        ):
            row = dict(
                diagnostic
            )
            row[
                "CandidateModel"
            ] = model
            diagnostics.append(
                row
            )

    scores = pd.DataFrame(
        score_rows
    )

    valid = scores[
        np.isfinite(
            scores[
                "Inner_NRMSE"
            ]
        )
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No valid inner-CV candidate "
            f"for signal={signal}, "
            f"window={window}"
        )

    best_score = valid[
        "Inner_NRMSE"
    ].min()

    tied = valid[
        np.abs(
            valid[
                "Inner_NRMSE"
            ]
            - best_score
        )
        <= 1e-8
    ].copy()

    tied[
        "Priority"
    ] = (
        tied[
            "Model"
        ]
        .map(
            MODEL_PRIORITY
        )
        .fillna(
            999
        )
    )

    chosen = (
        tied
        .sort_values(
            [
                "Priority",
                "Model",
            ]
        )
        .iloc[0][
            "Model"
        ]
    )

    return (
        chosen,
        scores,
        diagnostics,
    )
