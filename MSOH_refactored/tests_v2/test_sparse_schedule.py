from __future__ import annotations

import pandas as pd

from src_v2.sparse_schedule import infer_sparse_schedule


def fixture_df():
    rows = []
    for g in ["A", "B", "C"]:
        for d in range(1, 10):
            observed = d in [2, 4, 6, 8]
            rows.append({
                "Group": g,
                "Day": d,
                "Defecation": float(d) if observed else None,
                "Defecation_observed": int(observed),
            })
    return pd.DataFrame(rows)


def test_effective_windows_are_schedule_aware():
    schedule, summary, targets = infer_sparse_schedule(
        fixture_df(),
        signal="Defecation",
        observed_col="Defecation_observed",
        candidate_windows=[2, 3, 4, 5, 6],
        target_days=[7, 8, 9],
    )
    assert schedule.design_observed_days == (2, 4, 6, 8)
    assert schedule.effective_windows == (2, 4, 6)
    assert schedule.observed_target_days == (8,)
    assert schedule.schedule_consistent_across_groups is True


def test_no_fake_daily_information():
    schedule, _, _ = infer_sparse_schedule(
        fixture_df(),
        signal="Defecation",
        observed_col="Defecation_observed",
        candidate_windows=[2, 3, 4, 5, 6],
        target_days=[7, 8, 9],
    )
    assert 3 not in schedule.effective_windows
    assert 5 not in schedule.effective_windows
