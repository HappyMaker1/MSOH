import itertools
import numpy as np
import pandas as pd


def _add_persistent_saturation(trace, entity_col="Entity"):
    """Add local and persistent saturation diagnostics.

    Local saturation requires both:
      1) low probability that the next-window gain exceeds the meaningful-gain delta; and
      2) sufficient group-level stability.

    Persistent saturation at window k requires every *evaluable* transition from
    k onward to satisfy local saturation. This prevents a single small/negative
    gain followed by a later large gain from being called saturation.
    """
    out = trace.copy()
    out["Condition_LocalSaturation"] = (
        out["Condition_Saturation"] & out["Condition_Stability"]
    )
    out["Condition_PersistentSaturation"] = False
    out["N_Future_Transitions_Evaluable"] = 0
    out["N_Future_Transitions_Saturated"] = 0

    for entity, idx in out.groupby(entity_col, sort=False).groups.items():
        p = out.loc[idx].sort_values(["FromWindow", "ToWindow"])
        ordered_idx = list(p.index)
        local = p["Condition_LocalSaturation"].fillna(False).to_numpy(bool)
        for pos, ridx in enumerate(ordered_idx):
            future_local = local[pos:]
            out.at[ridx, "N_Future_Transitions_Evaluable"] = int(len(future_local))
            out.at[ridx, "N_Future_Transitions_Saturated"] = int(np.sum(future_local))
            out.at[ridx, "Condition_PersistentSaturation"] = bool(
                len(future_local) > 0 and np.all(future_local)
            )
    return out


def decide_msoh(
    error_table,
    gain_table,
    stability,
    entity_col="Entity",
    error_threshold=0.40,
    gain_delta=0.03,
    alpha=0.10,
    stability_threshold=0.75,
):
    """Separate predictive adequacy from persistent information saturation.

    Final MSOH = earliest evaluable window satisfying BOTH:
      - predictive adequacy: Error(k) <= error_threshold
      - persistent saturation: every evaluable transition k->k+1,... has
        P(Gain > gain_delta) < alpha and adequate group-level stability.

    The function also reports adequacy onset and saturation onset independently.
    """
    trace = gain_table.merge(
        stability[
            [
                entity_col,
                "FromWindow",
                "ToWindow",
                "Stable_Group_Proportion",
                "Improved_Group_Proportion",
                "Worsened_Group_Proportion",
            ]
        ],
        on=[entity_col, "FromWindow", "ToWindow"],
        how="left",
    )
    cur = error_table[
        [entity_col, "Window", "Error_Point", "Error_CI_Low", "Error_CI_High"]
    ].rename(columns={"Window": "FromWindow"})
    trace = trace.merge(cur, on=[entity_col, "FromWindow"], how="left")

    trace["Error_Threshold"] = error_threshold
    trace["Gain_Delta"] = gain_delta
    trace["Alpha"] = alpha
    trace["Stability_Threshold"] = stability_threshold

    trace["Condition_Adequacy"] = trace["Error_Point"] <= error_threshold
    # Backward-compatible alias.
    trace["Condition_Error"] = trace["Condition_Adequacy"]
    trace["Condition_Saturation"] = trace["P_Gain_GT_Delta"] < alpha
    trace["Condition_Stability"] = (
        trace["Stable_Group_Proportion"] >= stability_threshold
    )
    trace = _add_persistent_saturation(trace, entity_col)
    trace["All_Conditions"] = (
        trace["Condition_Adequacy"] & trace["Condition_PersistentSaturation"]
    )

    rows = []
    all_entities = sorted(error_table[entity_col].dropna().unique())
    for entity in all_entities:
        ep = error_table[error_table[entity_col] == entity].sort_values("Window")
        tp = trace[trace[entity_col] == entity].sort_values("FromWindow")

        adequate = ep[ep["Error_Point"] <= error_threshold]
        adequacy_onset = np.nan if adequate.empty else int(adequate.iloc[0]["Window"])

        saturated = tp[tp["Condition_PersistentSaturation"]]
        saturation_onset = (
            np.nan if saturated.empty else int(saturated.iloc[0]["FromWindow"])
        )
        saturation_n_future = (
            np.nan
            if saturated.empty
            else int(saturated.iloc[0]["N_Future_Transitions_Evaluable"])
        )

        final = tp[tp["All_Conditions"]]
        candidate = np.nan if final.empty else int(final.iloc[0]["FromWindow"])

        has_adequacy = not adequate.empty
        has_saturation = not saturated.empty
        if not final.empty:
            status = "MSOH identified: adequate + persistently saturated"
        elif has_saturation and not has_adequacy:
            status = "Persistent saturation detected, but predictive adequacy not reached"
        elif has_adequacy and not has_saturation:
            status = "Predictive adequacy reached, but persistent saturation not established"
        else:
            status = "Neither predictive adequacy nor persistent saturation established"

        last_window = int(ep["Window"].max()) if not ep.empty else np.nan
        last_error = (
            float(ep.loc[ep["Window"] == last_window, "Error_Point"].iloc[0])
            if not ep.empty
            else np.nan
        )

        rows.append(
            {
                entity_col: entity,
                "Candidate_MSOH": candidate,
                "Adequacy_Onset": adequacy_onset,
                "Persistent_Saturation_Onset": saturation_onset,
                "Saturation_Evidence_N_Future_Transitions": saturation_n_future,
                "Last_Evaluable_Window": last_window,
                "Last_Window_Error": last_error,
                "Adequacy_Reached_Within_Horizon": bool(has_adequacy),
                "Persistent_Saturation_Established_Within_Horizon": bool(has_saturation),
                "Decision_Status": status,
                "Error_Threshold": error_threshold,
                "Gain_Delta": gain_delta,
                "Alpha": alpha,
                "Stability_Threshold": stability_threshold,
            }
        )

    return trace, pd.DataFrame(rows)


def gain_table_for_delta(store, delta, entity_col="Entity"):
    rows = []
    for entity, info in store.items():
        for i, (fw, tw) in enumerate(zip(info["windows"][:-1], info["windows"][1:])):
            vals = info["gain_boot"][:, i]
            vals = vals[~np.isnan(vals)]
            rows.append(
                {
                    entity_col: entity,
                    "FromWindow": fw,
                    "ToWindow": tw,
                    "Gain_Point": info["point_errors"][i] - info["point_errors"][i + 1],
                    "Gain_Delta": delta,
                    "P_Gain_GT_Delta": float(np.mean(vals > delta)),
                    "P_Gain_LE_Delta": float(np.mean(vals <= delta)),
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    error_table,
    store,
    stability,
    entity_col="Entity",
    error_grid=(0.35, 0.4, 0.45, 0.5),
    delta_grid=(0.01, 0.02, 0.03, 0.05),
    alpha_grid=(0.05, 0.1, 0.2),
    stability_grid=(0.5, 0.67, 0.75, 0.8),
):
    rows = []
    cache = {d: gain_table_for_delta(store, d, entity_col) for d in delta_grid}
    for te, d, a, ts in itertools.product(
        error_grid, delta_grid, alpha_grid, stability_grid
    ):
        _, summary = decide_msoh(
            error_table, cache[d], stability, entity_col, te, d, a, ts
        )
        for _, r in summary.iterrows():
            rows.append(
                {
                    entity_col: r[entity_col],
                    "Error_Threshold": te,
                    "Gain_Delta": d,
                    "Alpha": a,
                    "Stability_Threshold": ts,
                    "Candidate_MSOH": r["Candidate_MSOH"],
                    "Adequacy_Onset": r["Adequacy_Onset"],
                    "Persistent_Saturation_Onset": r[
                        "Persistent_Saturation_Onset"
                    ],
                    "Decision_Status": r["Decision_Status"],
                }
            )
    return pd.DataFrame(rows)


def summarize_sensitivity(df, entity_col="Entity"):
    rows = []
    for entity, p in df.groupby(entity_col):
        total = len(p)
        valid = p.dropna(subset=["Candidate_MSOH"])
        no = total - len(valid)
        if valid.empty:
            mode = prop = mn = mx = np.nan
        else:
            c = valid["Candidate_MSOH"].value_counts()
            mode = int(c.index[0])
            prop = float(c.iloc[0] / total)
            mn = int(valid["Candidate_MSOH"].min())
            mx = int(valid["Candidate_MSOH"].max())

        sat = p.dropna(subset=["Persistent_Saturation_Onset"])
        sat_prop = float(len(sat) / total)
        sat_mode = np.nan
        if not sat.empty:
            sat_mode = int(sat["Persistent_Saturation_Onset"].value_counts().index[0])

        adequacy = p.dropna(subset=["Adequacy_Onset"])
        adequacy_prop = float(len(adequacy) / total)

        rows.append(
            {
                entity_col: entity,
                "N_Threshold_Combinations": total,
                "N_No_MSOH": no,
                "P_No_MSOH": no / total,
                "Most_Common_MSOH": mode,
                "Most_Common_MSOH_Proportion": prop,
                "MSOH_Min": mn,
                "MSOH_Max": mx,
                "P_Adequacy_Reached": adequacy_prop,
                "P_Persistent_Saturation_Established": sat_prop,
                "Most_Common_Persistent_Saturation_Onset": sat_mode,
            }
        )
    return pd.DataFrame(rows)
