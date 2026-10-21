"""
Schedule-aware handling for planned sparse longitudinal signals.

No interpolation is performed here. The observed flag is the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SparseSchedule:
    signal: str
    observed_col: str
    candidate_windows: tuple[int, ...]
    target_days: tuple[int, ...]
    design_observed_days: tuple[int, ...]
    effective_windows: tuple[int, ...]
    observed_target_days: tuple[int, ...]
    schedule_consistent_across_groups: bool


def observed_day_summary(
    df: pd.DataFrame,
    signal: str,
    observed_col: str,
) -> pd.DataFrame:
    rows = []
    groups = sorted(df["Group"].astype(str).unique().tolist())
    n_groups = len(groups)

    for day, part in df.groupby("Day", sort=True):
        flags = pd.to_numeric(part[observed_col], errors="coerce").fillna(0)
        values = pd.to_numeric(part[signal], errors="coerce")
        n_obs = int((flags == 1).sum())
        n_nonmissing = int(values.notna().sum())

        rows.append({
            "Signal": signal,
            "Day": int(day),
            "N_Groups": int(n_groups),
            "N_Observed_Flag1": n_obs,
            "Observed_Fraction": float(n_obs / n_groups) if n_groups else np.nan,
            "N_Nonmissing_Value": n_nonmissing,
            "Any_Observed": bool(n_obs > 0),
            "All_Groups_Observed": bool(n_obs == n_groups),
        })

    return pd.DataFrame(rows)


def infer_sparse_schedule(
    df: pd.DataFrame,
    signal: str,
    observed_col: str,
    candidate_windows: list[int] | tuple[int, ...],
    target_days: list[int] | tuple[int, ...],
) -> tuple[SparseSchedule, pd.DataFrame, pd.DataFrame]:
    summary = observed_day_summary(df, signal, observed_col)
    design_days = tuple(
        summary.loc[summary["Any_Observed"], "Day"].astype(int).tolist()
    )

    candidate_windows = tuple(sorted(int(k) for k in candidate_windows))
    target_days = tuple(sorted(int(d) for d in target_days))

    # A candidate window is "effective" only when it contains a newly available
    # real sparse observation relative to the previous candidate window.
    effective = []
    previous_seen: set[int] = set()
    for k in candidate_windows:
        seen = {d for d in design_days if d <= k}
        if seen - previous_seen:
            effective.append(k)
        previous_seen = seen

    observed_targets = tuple(d for d in target_days if d in design_days)

    # Consistency is descriptive: every design-observed day should either be
    # observed in all groups or not. We do not silently force a consensus.
    obs_rows = summary[summary["Any_Observed"]]
    consistent = bool(
        len(obs_rows) == 0 or obs_rows["All_Groups_Observed"].all()
    )

    target_table = summary[summary["Day"].isin(target_days)].copy()
    target_table["Is_TargetDay"] = True
    target_table["Usable_As_Observed_Target"] = target_table["Any_Observed"]

    schedule = SparseSchedule(
        signal=signal,
        observed_col=observed_col,
        candidate_windows=candidate_windows,
        target_days=target_days,
        design_observed_days=design_days,
        effective_windows=tuple(effective),
        observed_target_days=observed_targets,
        schedule_consistent_across_groups=consistent,
    )

    return schedule, summary, target_table


def sparse_information_count_by_window(
    schedule: SparseSchedule,
) -> pd.DataFrame:
    rows = []
    for k in schedule.candidate_windows:
        n = sum(d <= k for d in schedule.design_observed_days)
        rows.append({
            "Signal": schedule.signal,
            "Window": int(k),
            "N_Real_Observed_Days_Available": int(n),
            "Introduces_New_Sparse_Information": bool(
                k in schedule.effective_windows
            ),
        })
    return pd.DataFrame(rows)
