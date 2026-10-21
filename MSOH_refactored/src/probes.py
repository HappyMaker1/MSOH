"""Alternative prediction probes for MSOH model-family robustness.

The original pipeline uses trajectory similarity + nearest-neighbour pooling.
This module adds two deliberately different prediction families:

1) Ridge Direct
   Early multivariate trajectory features -> each future signal/day directly.
   Preprocessing and alpha selection are performed strictly inside each LOOG
   training fold.

2) Mixed Effects
   A signal-specific longitudinal mixed model is fitted on training groups.
   For the held-out group, its Day1..Day-k measurements update a random
   intercept/slope through the Gaussian BLUP formula, then Day7..Day9 are
   forecast.  This is a population-growth + individual/group-update probe,
   not a nearest-neighbour method.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALL_SIGNALS, TARGET_DAYS, WINDOWS
from .common import probe_model_name
from .prediction import get_truth

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
except Exception:  # pragma: no cover - handled with an explicit runtime error
    MixedLM = None


DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TIME_CENTER = 5.0


def _append(rows, group, window, model, truth, pred, target_days=TARGET_DAYS):
    for signal in ALL_SIGNALS:
        for day in target_days:
            yt = truth[(signal, day)]
            yp = pred.get((signal, day), np.nan)
            err = yp - yt if not pd.isna(yt) and not pd.isna(yp) else np.nan
            rows.append(
                {
                    "Group": str(group),
                    "Window": int(window),
                    "Model": model,
                    "Signal": signal,
                    "TargetDay": int(day),
                    "y_true": yt,
                    "y_pred": yp,
                    "error": err,
                    "abs_error": abs(err) if not pd.isna(err) else np.nan,
                }
            )


# =====================================================================
# Ridge direct prediction
# =====================================================================

def _ridge_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    # Keep the same compact representation used by Euclidean Feature so that
    # the diagnostic asks whether the *prediction family*, rather than a much
    # richer feature engineering recipe, changes the result.
    cols = [
        c
        for c in feature_df.columns
        if c.endswith("_last") or c.endswith("_slope")
    ]
    if not cols:
        raise ValueError("feature table 中未找到 *_last / *_slope 特征")
    return cols


def _ridge_fit_predict(X_train, y_train, X_test, alphas):
    """Train-fold-only imputation/scaling + inner-LOO alpha selection."""
    n = len(y_train)
    if n < 2:
        return float(np.nanmean(y_train)) if n else np.nan, np.nan

    # With very tiny training sets RidgeCV/LOO can be unstable.  The ordinary
    # project setting has ~11 training groups; fallback keeps the function safe.
    if n >= 4:
        ridge = RidgeCV(
            alphas=np.asarray(alphas, dtype=float),
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
    else:
        ridge = Ridge(alpha=1.0)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", ridge),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = float(pipe.predict(X_test)[0])
    fitted = pipe.named_steps["ridge"]
    alpha = float(getattr(fitted, "alpha_", getattr(fitted, "alpha", np.nan)))
    return pred, alpha


def run_ridge_loog(
    data_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    windows=WINDOWS,
    target_days=TARGET_DAYS,
    alphas=DEFAULT_RIDGE_ALPHAS,
):
    """LOOG direct Ridge forecast for each signal x future day.

    One multivariate early feature vector (last+slope across all four signals)
    is used to predict each future response separately.  The held-out group is
    never used for imputation, scaling, or alpha selection.
    """
    model = probe_model_name("ridge")
    groups = sorted(data_df["Group"].astype(str).unique())
    feature_df = feature_df.copy()
    feature_df["Group"] = feature_df["Group"].astype(str)
    cols = _ridge_feature_columns(feature_df)
    rows, diagnostics = [], []

    for test in groups:
        train_groups = [g for g in groups if g != test]
        truth = get_truth(data_df, test, target_days)

        for window in windows:
            fw = feature_df[feature_df["Window"] == window].copy()
            test_x = fw[fw["Group"] == test]
            if len(test_x) != 1:
                raise ValueError(f"Ridge: Group={test}, Window={window} feature row != 1")

            pred = {}
            for signal in ALL_SIGNALS:
                for day in target_days:
                    target = data_df[
                        data_df["Group"].astype(str).isin(train_groups)
                        & (data_df["Day"] == day)
                        & data_df[signal].notna()
                    ][["Group", signal]].copy()
                    target["Group"] = target["Group"].astype(str)

                    train_x = fw[fw["Group"].isin(train_groups)][["Group", *cols]]
                    merged = train_x.merge(target, on="Group", how="inner")
                    if merged.empty:
                        yp, alpha = np.nan, np.nan
                    else:
                        yp, alpha = _ridge_fit_predict(
                            merged[cols].to_numpy(float),
                            merged[signal].to_numpy(float),
                            test_x[cols].to_numpy(float),
                            alphas,
                        )
                    pred[(signal, day)] = yp
                    diagnostics.append(
                        {
                            "Group": test,
                            "Window": int(window),
                            "Signal": signal,
                            "TargetDay": int(day),
                            "N_Train": int(len(merged)),
                            "Selected_Alpha": alpha,
                        }
                    )

            _append(rows, test, window, model, truth, pred, target_days)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


# =====================================================================
# Mixed-effects longitudinal prediction
# =====================================================================

def _fixed_design(day) -> np.ndarray:
    d = np.asarray(day, dtype=float)
    tc = d - TIME_CENTER
    return np.column_stack([np.ones(len(d)), tc, tc**2])


def _random_design(day, random_slope=True) -> np.ndarray:
    d = np.asarray(day, dtype=float)
    tc = d - TIME_CENTER
    if random_slope:
        return np.column_stack([np.ones(len(d)), tc])
    return np.ones((len(d), 1), dtype=float)


def _fit_mixed_one_signal(train: pd.DataFrame, signal: str):
    """Fit quadratic fixed time + random intercept/slope; fallback to intercept.

    Returns a compact dictionary that is independent of statsmodels' prediction
    API and can be used to obtain the BLUP for an unseen held-out group.
    """
    if MixedLM is None:
        raise ImportError(
            "Mixed Effects 需要 statsmodels。请运行: python -m pip install statsmodels"
        )

    p = train[["Group", "Day", signal]].dropna().copy()
    p["Group"] = p["Group"].astype(str)
    if p["Group"].nunique() < 3 or len(p) < 8:
        return _fit_ols_fallback(p, signal, reason="too_few_training_observations")

    y = p[signal].to_numpy(float)
    X = _fixed_design(p["Day"].to_numpy(float))

    last_error = None
    for random_slope in (True, False):
        Z = _random_design(p["Day"].to_numpy(float), random_slope=random_slope)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = MixedLM(y, X, groups=p["Group"].to_numpy(), exog_re=Z)
                # Multiple optimizers improve robustness on small group-level data.
                res = md.fit(reml=True, method=["lbfgs", "powell"], disp=False)
            fe = np.asarray(res.fe_params, dtype=float)
            G = np.asarray(res.cov_re, dtype=float)
            sigma2 = float(res.scale)
            if not np.all(np.isfinite(fe)) or not np.all(np.isfinite(G)) or not np.isfinite(sigma2):
                raise FloatingPointError("non-finite MixedLM parameters")
            return {
                "fit_type": "random_intercept_slope" if random_slope else "random_intercept",
                "fe": fe,
                "G": G,
                "sigma2": max(sigma2, 1e-12),
                "random_slope": random_slope,
                "converged": bool(getattr(res, "converged", True)),
                "llf": float(getattr(res, "llf", np.nan)),
            }
        except Exception as exc:
            last_error = repr(exc)

    return _fit_ols_fallback(p, signal, reason=last_error or "mixedlm_failed")


def _fit_ols_fallback(p: pd.DataFrame, signal: str, reason: str):
    """Deterministic safety fallback; diagnostics make its use explicit."""
    if p.empty:
        return {
            "fit_type": "no_data",
            "fe": np.array([np.nan, np.nan, np.nan]),
            "G": np.zeros((1, 1)),
            "sigma2": np.nan,
            "random_slope": False,
            "converged": False,
            "llf": np.nan,
            "fallback_reason": reason,
        }
    y = p[signal].to_numpy(float)
    X = _fixed_design(p["Day"].to_numpy(float))
    fe = np.linalg.pinv(X) @ y
    resid = y - X @ fe
    sigma2 = float(np.mean(resid**2)) if len(resid) else 1.0
    # Empirical intercept variance across training-group mean residuals.  This
    # gives the held-out early observations a conservative shrinkage update.
    tmp = p.copy()
    tmp["resid"] = resid
    means = tmp.groupby("Group")["resid"].mean().to_numpy(float)
    var_b = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
    return {
        "fit_type": "ols_plus_random_intercept_fallback",
        "fe": np.asarray(fe, float),
        "G": np.array([[max(var_b, 1e-12)]], dtype=float),
        "sigma2": max(sigma2, 1e-12),
        "random_slope": False,
        "converged": False,
        "llf": np.nan,
        "fallback_reason": reason,
    }


def _heldout_blup(fit: dict, early: pd.DataFrame, signal: str) -> np.ndarray:
    q = 2 if fit["random_slope"] else 1
    if early.empty or not np.all(np.isfinite(fit["fe"])):
        return np.zeros(q, dtype=float)

    y = early[signal].to_numpy(float)
    day = early["Day"].to_numpy(float)
    X = _fixed_design(day)
    Z = _random_design(day, random_slope=fit["random_slope"])
    r = y - X @ fit["fe"]
    G = np.asarray(fit["G"], float)
    sigma2 = float(fit["sigma2"])
    V = Z @ G @ Z.T + sigma2 * np.eye(len(y))
    try:
        Vinv_r = np.linalg.solve(V, r)
    except np.linalg.LinAlgError:
        Vinv_r = np.linalg.pinv(V) @ r
    return G @ Z.T @ Vinv_r


def _mixed_predict_targets(fit: dict, early: pd.DataFrame, signal: str, target_days):
    if not np.all(np.isfinite(fit["fe"])):
        return {d: np.nan for d in target_days}, np.zeros(1)
    b = _heldout_blup(fit, early, signal)
    Xf = _fixed_design(target_days)
    Zf = _random_design(target_days, random_slope=fit["random_slope"])
    yhat = Xf @ fit["fe"] + Zf @ b
    return {int(d): float(v) for d, v in zip(target_days, yhat)}, b


def run_mixed_effects_loog(
    data_df: pd.DataFrame,
    windows=WINDOWS,
    target_days=TARGET_DAYS,
):
    """LOOG mixed-effects forecast with held-out-group BLUP updating.

    Population model for each signal is fitted only on the training groups, but
    may use all their observed Day1..Day9 values (standard supervised training).
    The held-out group's random effect is estimated only from Day1..Day-k.
    """
    model = probe_model_name("mixed_effects")
    df = data_df.copy()
    df["Group"] = df["Group"].astype(str)
    groups = sorted(df["Group"].unique())
    rows, diagnostics = [], []

    for test in groups:
        train_groups = [g for g in groups if g != test]
        train = df[df["Group"].isin(train_groups)].copy()
        test_df = df[df["Group"] == test].copy()
        truth = get_truth(df, test, target_days)

        # Fit once per held-out fold/signal.  The population model does not
        # depend on k; only the held-out BLUP update changes with k.
        fits = {signal: _fit_mixed_one_signal(train, signal) for signal in ALL_SIGNALS}

        for window in windows:
            pred = {}
            for signal in ALL_SIGNALS:
                early = test_df[
                    (test_df["Day"] <= window) & test_df[signal].notna()
                ][["Day", signal]].sort_values("Day")
                vals, b = _mixed_predict_targets(fits[signal], early, signal, target_days)
                for day in target_days:
                    pred[(signal, day)] = vals[day]

                f = fits[signal]
                diagnostics.append(
                    {
                        "Group": test,
                        "Window": int(window),
                        "Signal": signal,
                        "N_Early": int(len(early)),
                        "Fit_Type": f["fit_type"],
                        "Converged": bool(f.get("converged", False)),
                        "RandomEffect_Intercept": float(b[0]) if len(b) else np.nan,
                        "RandomEffect_Slope": float(b[1]) if len(b) > 1 else np.nan,
                        "Residual_Variance": float(f.get("sigma2", np.nan)),
                        "Fallback_Reason": f.get("fallback_reason", ""),
                    }
                )

            _append(rows, test, window, model, truth, pred, target_days)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
