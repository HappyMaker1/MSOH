import numpy as np
import pandas as pd
from config import ALL_SIGNALS, WINDOWS, EXPECTED_DAYS


def validate_clean_data(df: pd.DataFrame):
    required = ["Group", "Day"] + [x for s in ALL_SIGNALS for x in (s, f"{s}_observed")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少字段：{missing}")
    if df.duplicated(["Group", "Day"]).any():
        raise ValueError("输入仍存在重复 Group-Day")
    expected = set(EXPECTED_DAYS)
    for g, p in df.groupby("Group"):
        m = expected - set(p["Day"].astype(int))
        if m:
            raise ValueError(f"Group={g} 缺少时间点：{sorted(m)}")


def observed_points(group_df, signal, k):
    return group_df[
        (group_df["Day"] <= k)
        & (group_df[f"{signal}_observed"] == 1)
        & group_df[signal].notna()
    ].sort_values("Day")


def build_feature_table(df: pd.DataFrame, windows=WINDOWS, use_delta=False) -> pd.DataFrame:
    rows = []
    for g, gp in df.groupby("Group", sort=True):
        gp = gp.sort_values("Day")
        for k in windows:
            row = {"Group": str(g), "Window": k, "ObservedUntilDay": k}
            for s in ALL_SIGNALS:
                p = observed_points(gp, s, k)
                row[f"{s}_n_observed"] = len(p)
                row[f"{s}_last"] = float(p.iloc[-1][s]) if len(p) else np.nan
                row[f"{s}_slope"] = float(np.polyfit(p["Day"].to_numpy(float), p[s].to_numpy(float), 1)[0]) if len(p)>=2 else np.nan
                if use_delta:
                    row[f"{s}_delta"] = float(p.iloc[-1][s]-p.iloc[0][s]) if len(p)>=2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def feature_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in features.columns if c not in {"Group","Window","ObservedUntilDay"}]
    for k, p in features.groupby("Window"):
        for c in feature_cols:
            rows.append({"Window":int(k), "Feature":c, "Missing_N":int(p[c].isna().sum()), "NonMissing_N":int(p[c].notna().sum())})
    return pd.DataFrame(rows)
