from __future__ import annotations

import pandas as pd
import pytest

from config_v2 import DEFAULT_THRESHOLDS, SIGNAL_SPECS
from src_v2.signal_registry import build_registry, validate_registry_against_data


def fixture_df():
    rows = []
    for g in ["A", "B", "C"]:
        for d in range(1, 10):
            rows.append({
                "Group": g,
                "Day": d,
                "Weight": 100 + d,
                "Weight_observed": 1,
                "Status_Score": 5 + 0.1 * d,
                "Status_Score_observed": 1,
                "Feeding": 20 + d,
                "Feeding_observed": 1,
                "Defecation": float(d) if d in [2, 4, 6, 8] else None,
                "Defecation_observed": 1 if d in [2, 4, 6, 8] else 0,
            })
    return pd.DataFrame(rows)


def test_registry_has_four_signals():
    reg = build_registry(SIGNAL_SPECS, DEFAULT_THRESHOLDS)
    assert set(reg) == {"Weight", "Status_Score", "Feeding", "Defecation"}
    assert reg["Defecation"].is_sparse
    assert not reg["Weight"].is_sparse


def test_registry_validates_clean_fixture():
    reg = build_registry(SIGNAL_SPECS, DEFAULT_THRESHOLDS)
    validate_registry_against_data(fixture_df(), reg)


def test_observed_flag_requires_value():
    df = fixture_df()
    df.loc[(df["Group"] == "A") & (df["Day"] == 2), "Defecation"] = None
    reg = build_registry(SIGNAL_SPECS, DEFAULT_THRESHOLDS)
    with pytest.raises(ValueError):
        validate_registry_against_data(df, reg)
