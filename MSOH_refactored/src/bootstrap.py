import numpy as np
import pandas as pd
from config import CORE_SIGNALS
from .common import similarity_label


# =====================================================================
# Shared aggregate bootstrap used by 06/07 and future analyses
# =====================================================================

def build_group_stats(pred, entity_col="Entity"):
    stores = {}
    for entity, pred_entity in pred.groupby(entity_col, sort=True):
        groups = sorted(pred_entity["Group"].astype(str).unique())
        windows = sorted(int(x) for x in pred_entity["Window"].unique())
        gidx = {g: i for i, g in enumerate(groups)}
        stats = {}
        for w in windows:
            stats[w] = {}
            for s in CORE_SIGNALS:
                p = pred_entity[
                    (pred_entity["Window"] == w)
                    & (pred_entity["Signal"] == s)
                    & pred_entity["y_true"].notna()
                    & pred_entity["y_pred"].notna()
                ]
                n = np.zeros(len(groups))
                sy = np.zeros(len(groups))
                sy2 = np.zeros(len(groups))
                sse = np.zeros(len(groups))
                for g, gp in p.groupby("Group"):
                    i = gidx[str(g)]
                    y = gp["y_true"].to_numpy(float)
                    yp = gp["y_pred"].to_numpy(float)
                    n[i] = len(y)
                    sy[i] = y.sum()
                    sy2[i] = (y * y).sum()
                    sse[i] = ((yp - y) ** 2).sum()
                stats[w][s] = {"n": n, "sy": sy, "sy2": sy2, "sse": sse}
        stores[entity] = {"groups": groups, "windows": windows, "stats": stats}
    return stores


def _nrmse(counts, s):
    n = float(np.dot(counts, s["n"]))
    if n <= 1:
        return np.nan
    sy = float(np.dot(counts, s["sy"]))
    sy2 = float(np.dot(counts, s["sy2"]))
    sse = float(np.dot(counts, s["sse"]))
    rmse = np.sqrt(sse / n)
    var = max(float((sy2 - sy * sy / n) / (n - 1)), 0.0)
    sd = np.sqrt(var)
    return np.nan if np.isclose(sd, 0) else float(rmse / sd)


def composite_error(counts, w, stats):
    vals = [_nrmse(counts, stats[w][s]) for s in CORE_SIGNALS]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_entities(pred, entity_col="Entity", b=5000, seed=20260924, shared_counts=True):
    """Generic/paired aggregate bootstrap.

    This function is intentionally retained unchanged for 07, where All4 and
    leave-one-signal-out variants must share the same bootstrap resamples.
    """
    stores = build_group_stats(pred, entity_col)
    out = {}
    base_groups = None
    shared = None
    for idx, (entity, info) in enumerate(stores.items()):
        groups, windows, stats = info["groups"], info["windows"], info["stats"]
        if shared_counts:
            if base_groups is None:
                base_groups = groups
                rng = np.random.default_rng(seed)
                shared = rng.multinomial(
                    len(groups), np.repeat(1 / len(groups), len(groups)), size=b
                )
            elif groups != base_groups:
                raise ValueError("配对 bootstrap 要求各实体 Group 集合一致")
            counts_mat = shared
        else:
            rng = np.random.default_rng(seed + idx * 1009)
            counts_mat = rng.multinomial(
                len(groups), np.repeat(1 / len(groups), len(groups)), size=b
            )
        full = np.ones(len(groups))
        point = np.array([composite_error(full, w, stats) for w in windows])
        boot = np.empty((b, len(windows)))
        for bi, c in enumerate(counts_mat):
            for wi, w in enumerate(windows):
                boot[bi, wi] = composite_error(c, w, stats)
        out[entity] = {
            **info,
            "point_errors": point,
            "error_boot": boot,
            "gain_boot": boot[:, :-1] - boot[:, 1:],
        }
    return out


def error_gain_tables(store, gain_delta=0.03, entity_name="Entity"):
    erows, grows = [], []
    for entity, info in store.items():
        ws = info["windows"]
        point = info["point_errors"]
        for i, w in enumerate(ws):
            vals = info["error_boot"][:, i]
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            erows.append(
                {
                    entity_name: entity,
                    "Window": w,
                    "Error_Point": point[i],
                    "Error_CI_Low": lo,
                    "Error_CI_High": hi,
                }
            )
        for i, (fw, tw) in enumerate(zip(ws[:-1], ws[1:])):
            vals = info["gain_boot"][:, i]
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            grows.append(
                {
                    entity_name: entity,
                    "FromWindow": fw,
                    "ToWindow": tw,
                    "Gain_Point": point[i] - point[i + 1],
                    "Gain_CI_Low": lo,
                    "Gain_CI_High": hi,
                    "P_Gain_GT_0": float(np.mean(vals > 0)),
                    "Gain_Delta": gain_delta,
                    "P_Gain_GT_Delta": float(np.mean(vals > gain_delta)),
                }
            )
    return pd.DataFrame(erows), pd.DataFrame(grows)


# =====================================================================
# 05 compatibility bootstrap: exact RNG semantics of original 05
# =====================================================================

def _core3_error_rows(rows: pd.DataFrame) -> float:
    vals = []
    for signal in CORE_SIGNALS:
        part = rows[
            (rows["Signal"] == signal)
            & rows["y_true"].notna()
            & rows["y_pred"].notna()
        ]
        if len(part) < 2:
            continue
        y = part["y_true"].to_numpy(float)
        yp = part["y_pred"].to_numpy(float)
        sd = float(np.std(y, ddof=1))
        if np.isclose(sd, 0.0):
            continue
        vals.append(float(np.sqrt(np.mean((yp - y) ** 2)) / sd))
    return float(np.mean(vals)) if vals else np.nan


def _method_table(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim[["SourceMethod", "Model"]]
        .drop_duplicates()
        .sort_values(["SourceMethod", "Model"])
        .reset_index(drop=True)
    )


def _resample_groups_as_rows(part: pd.DataFrame, sampled_groups) -> pd.DataFrame:
    blocks = []
    for bootstrap_id, group in enumerate(sampled_groups):
        block = part[part["Group"].astype(str) == str(group)].copy()
        if block.empty:
            continue
        block["_BootstrapUnit"] = bootstrap_id
        blocks.append(block)
    if not blocks:
        return part.iloc[0:0].copy()
    return pd.concat(blocks, ignore_index=True)


def legacy05_bootstrap_error_ci(sim: pd.DataFrame, bootstrap_b: int, rng) -> pd.DataFrame:
    rows = []
    for _, mrow in _method_table(sim).iterrows():
        source = mrow["SourceMethod"]
        model = mrow["Model"]
        method_part = sim[(sim["SourceMethod"] == source) & (sim["Model"] == model)].copy()
        groups = sorted(method_part["Group"].astype(str).unique().tolist())
        n_groups = len(groups)
        for window in sorted(method_part["Window"].unique()):
            part = method_part[method_part["Window"] == window]
            point = _core3_error_rows(part)
            boot = np.empty(bootstrap_b)
            for bi in range(bootstrap_b):
                sampled = rng.choice(groups, size=n_groups, replace=True)
                boot[bi] = _core3_error_rows(_resample_groups_as_rows(part, sampled))
            boot = boot[~np.isnan(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "SourceMethod": source,
                    "Model": model,
                    "SimilarityLabel": similarity_label(model),
                    "Window": int(window),
                    "N_Groups": n_groups,
                    "Error_Point": float(point),
                    "Error_CI_Low": float(lo),
                    "Error_CI_High": float(hi),
                    "Bootstrap_B": bootstrap_b,
                }
            )
    return pd.DataFrame(rows)


def legacy05_bootstrap_gain_ci(sim: pd.DataFrame, bootstrap_b: int, rng) -> pd.DataFrame:
    rows = []
    for _, mrow in _method_table(sim).iterrows():
        source = mrow["SourceMethod"]
        model = mrow["Model"]
        method_part = sim[(sim["SourceMethod"] == source) & (sim["Model"] == model)].copy()
        groups = sorted(method_part["Group"].astype(str).unique().tolist())
        n_groups = len(groups)
        windows = sorted(method_part["Window"].unique())
        for prev_w, curr_w in zip(windows[:-1], windows[1:]):
            prev_part = method_part[method_part["Window"] == prev_w]
            curr_part = method_part[method_part["Window"] == curr_w]
            prev_point = _core3_error_rows(prev_part)
            curr_point = _core3_error_rows(curr_part)
            boot = np.empty(bootstrap_b)
            for bi in range(bootstrap_b):
                sampled = rng.choice(groups, size=n_groups, replace=True)
                boot[bi] = _core3_error_rows(_resample_groups_as_rows(prev_part, sampled)) - _core3_error_rows(_resample_groups_as_rows(curr_part, sampled))
            boot = boot[~np.isnan(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "SourceMethod": source,
                    "Model": model,
                    "SimilarityLabel": similarity_label(model),
                    "FromWindow": int(prev_w),
                    "ToWindow": int(curr_w),
                    "Error_From": float(prev_point),
                    "Error_To": float(curr_point),
                    "Gain_Point": float(prev_point - curr_point),
                    "Gain_CI_Low": float(lo),
                    "Gain_CI_High": float(hi),
                    "P_Gain_GT_0": float(np.mean(boot > 0)),
                    "P_Gain_LT_0": float(np.mean(boot < 0)),
                    "N_Groups": n_groups,
                    "Bootstrap_B": bootstrap_b,
                }
            )
    return pd.DataFrame(rows)


def legacy05_paired_method_difference(sim: pd.DataFrame, bootstrap_b: int, rng) -> pd.DataFrame:
    rows = []
    methods = _method_table(sim).to_dict("records")
    windows = sorted(sim["Window"].unique())
    for window in windows:
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                ma, mb = methods[i], methods[j]
                pa = sim[(sim["SourceMethod"] == ma["SourceMethod"]) & (sim["Model"] == ma["Model"]) & (sim["Window"] == window)]
                pb = sim[(sim["SourceMethod"] == mb["SourceMethod"]) & (sim["Model"] == mb["Model"]) & (sim["Window"] == window)]
                common = sorted(set(pa["Group"].astype(str)) & set(pb["Group"].astype(str)))
                if not common:
                    continue
                pa = pa[pa["Group"].astype(str).isin(common)]
                pb = pb[pb["Group"].astype(str).isin(common)]
                point = _core3_error_rows(pa) - _core3_error_rows(pb)
                boot = np.empty(bootstrap_b)
                for bi in range(bootstrap_b):
                    sampled = rng.choice(common, size=len(common), replace=True)
                    boot[bi] = _core3_error_rows(_resample_groups_as_rows(pa, sampled)) - _core3_error_rows(_resample_groups_as_rows(pb, sampled))
                boot = boot[~np.isnan(boot)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(
                    {
                        "Window": int(window),
                        "Method_A": similarity_label(ma["Model"]),
                        "Method_B": similarity_label(mb["Model"]),
                        "Difference_A_minus_B": float(point),
                        "CI_Low": float(lo),
                        "CI_High": float(hi),
                        "P_A_Error_GT_B": float(np.mean(boot > 0)),
                        "P_A_Error_LT_B": float(np.mean(boot < 0)),
                        "N_Groups": len(common),
                        "Bootstrap_B": bootstrap_b,
                    }
                )
    return pd.DataFrame(rows)


# =====================================================================
# 06 compatibility bootstrap: exact method-specific RNG streams
# =====================================================================

def legacy06_method_bootstrap(pred: pd.DataFrame, b=5000, seed=20260924):
    """Reproduce original 06 build_method_uncertainty RNG behavior exactly.

    Methods are ordered by SourceMethod, Model. Each method receives its own
    multinomial stream: seed + method_index * 1009. The same count matrix is
    reused across all windows within that method.
    """
    out = {}
    methods = _method_table(pred)
    for mi, mrow in methods.iterrows():
        source = mrow["SourceMethod"]
        model = mrow["Model"]
        label = similarity_label(model)
        part = pred[(pred["SourceMethod"] == source) & (pred["Model"] == model)].copy()
        part["Entity"] = label
        info = build_group_stats(part, "Entity")[label]
        groups, windows, stats = info["groups"], info["windows"], info["stats"]
        rng = np.random.default_rng(seed + mi * 1009)
        counts = rng.multinomial(
            len(groups), np.repeat(1.0 / len(groups), len(groups)), size=b
        )
        full = np.ones(len(groups))
        point = np.array([composite_error(full, w, stats) for w in windows])
        boot = np.empty((b, len(windows)))
        for bi, c in enumerate(counts):
            for wi, w in enumerate(windows):
                boot[bi, wi] = composite_error(c, w, stats)
        out[label] = {
            **info,
            "SourceMethod": source,
            "Model": model,
            "point_errors": point,
            "error_boot": boot,
            "gain_boot": boot[:, :-1] - boot[:, 1:],
        }
    return out


def legacy06_error_gain_tables(store, gain_delta=0.03):
    erows, grows = [], []
    for label, info in store.items():
        ws = info["windows"]
        point = info["point_errors"]
        for i, w in enumerate(ws):
            vals = info["error_boot"][:, i]
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            erows.append(
                {
                    "SimilarityLabel": label,
                    "SourceMethod": info["SourceMethod"],
                    "Model": info["Model"],
                    "Window": int(w),
                    "Error_Point": float(point[i]),
                    "Error_CI_Low": float(lo),
                    "Error_CI_High": float(hi),
                    "N_Groups": len(info["groups"]),
                    "Bootstrap_B": len(info["error_boot"]),
                }
            )
        for i, (fw, tw) in enumerate(zip(ws[:-1], ws[1:])):
            vals = info["gain_boot"][:, i]
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            grows.append(
                {
                    "SimilarityLabel": label,
                    "SourceMethod": info["SourceMethod"],
                    "Model": info["Model"],
                    "FromWindow": int(fw),
                    "ToWindow": int(tw),
                    "Gain_Point": float(point[i] - point[i + 1]),
                    "Gain_CI_Low": float(lo),
                    "Gain_CI_High": float(hi),
                    "P_Gain_GT_0": float(np.mean(vals > 0)),
                    "Gain_Delta": float(gain_delta),
                    "P_Gain_GT_Delta": float(np.mean(vals > gain_delta)),
                    "P_Gain_LE_Delta": float(np.mean(vals <= gain_delta)),
                    "N_Groups": len(info["groups"]),
                    "Bootstrap_B": len(info["error_boot"]),
                }
            )
    return pd.DataFrame(erows), pd.DataFrame(grows)


# =====================================================================
# Shared descriptive group stability / paired differences
# =====================================================================

def group_errors(pred, entity_col="Entity"):
    scales = {}
    for (entity, w, s), p in pred.groupby([entity_col, "Window", "Signal"]):
        y = p["y_true"].dropna().to_numpy(float)
        scales[(entity, int(w), s)] = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    rows = []
    for (entity, g, w), p in pred.groupby([entity_col, "Group", "Window"]):
        vals = []
        for s in CORE_SIGNALS:
            q = p[(p["Signal"] == s) & p["y_true"].notna() & p["y_pred"].notna()]
            sd = scales[(entity, int(w), s)]
            if q.empty or pd.isna(sd) or np.isclose(sd, 0):
                continue
            vals.append(
                float(
                    np.sqrt(
                        np.mean(
                            (q["y_pred"].to_numpy(float) - q["y_true"].to_numpy(float)) ** 2
                        )
                    )
                    / sd
                )
            )
        rows.append(
            {
                entity_col: entity,
                "Group": str(g),
                "Window": int(w),
                "GroupCore3Error": float(np.mean(vals)) if vals else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stability_table(group_error, entity_col="Entity", epsilon=0.03):
    rows = []
    for entity, p in group_error.groupby(entity_col):
        pivot = p.pivot(index="Group", columns="Window", values="GroupCore3Error")
        ws = sorted(pivot.columns)
        for fw, tw in zip(ws[:-1], ws[1:]):
            valid = pivot[fw].notna() & pivot[tw].notna()
            d = (pivot.loc[valid, fw] - pivot.loc[valid, tw]).to_numpy(float)
            rows.append(
                {
                    entity_col: entity,
                    "FromWindow": int(fw),
                    "ToWindow": int(tw),
                    "Group_Epsilon": epsilon,
                    "Stable_Group_Proportion": float(np.mean(np.abs(d) <= epsilon)) if len(d) else np.nan,
                    "Improved_Group_Proportion": float(np.mean(d > epsilon)) if len(d) else np.nan,
                    "Worsened_Group_Proportion": float(np.mean(d < -epsilon)) if len(d) else np.nan,
                    "N_Groups": len(d),
                }
            )
    return pd.DataFrame(rows)


def paired_entity_difference(store, entity_a, entity_b):
    a, b = store[entity_a], store[entity_b]
    if a["windows"] != b["windows"]:
        raise ValueError("窗口不一致")
    rows = []
    for i, w in enumerate(a["windows"]):
        d = a["error_boot"][:, i] - b["error_boot"][:, i]
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(
            {
                "Window": w,
                "Entity_A": entity_a,
                "Entity_B": entity_b,
                "Difference_A_minus_B": a["point_errors"][i] - b["point_errors"][i],
                "CI_Low": lo,
                "CI_High": hi,
                "P_A_Error_GT_B": float(np.mean(d > 0)),
                "P_A_Error_LT_B": float(np.mean(d < 0)),
            }
        )
    return pd.DataFrame(rows)
