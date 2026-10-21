import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config import ALL_SIGNALS, CORE_SIGNALS


def dtw_distance(a, b, radius=1):
    """Single source of truth: exact DTW definition used by the original 03."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    if radius is None:
        radius = max(n, m)
    radius = max(int(radius), abs(n-m))
    cost = np.full((n+1, m+1), np.inf)
    plen = np.zeros((n+1, m+1), dtype=int)
    cost[0,0] = 0.0
    for i in range(1,n+1):
        for j in range(max(1,i-radius), min(m,i+radius)+1):
            ai, bj = a[i-1], b[j-1]
            local = abs(float(ai)-float(bj)) if np.ndim(ai)==0 else np.linalg.norm(ai-bj)
            prev = min([(cost[i-1,j],plen[i-1,j]),(cost[i,j-1],plen[i,j-1]),(cost[i-1,j-1],plen[i-1,j-1])], key=lambda x:x[0])
            if np.isinf(prev[0]):
                continue
            cost[i,j] = prev[0] + local
            plen[i,j] = prev[1] + 1
    return float(cost[n,m]/plen[n,m]) if plen[n,m] else np.nan


def fit_fold_scaler(df, train_groups, window, signals=ALL_SIGNALS):
    train = df[df["Group"].astype(str).isin(train_groups) & (df["Day"] <= window)]
    stats = {}
    for s in signals:
        obs = f"{s}_observed"
        vals = train.loc[(train[obs]==1) & train[s].notna(), s].to_numpy(float) if obs in train.columns else train[s].dropna().to_numpy(float)
        mean = float(np.mean(vals)) if len(vals) else 0.0
        sd = float(np.std(vals)) if len(vals) else 1.0
        stats[s] = (mean, 1.0 if np.isclose(sd,0) else sd)
    return stats


def dense_sequence(df, group, window, scaler, signals):
    p = df[(df["Group"].astype(str)==str(group)) & (df["Day"]<=window)].sort_values("Day")
    cols = [(p[s].to_numpy(float)-scaler[s][0])/scaler[s][1] for s in signals]
    return np.column_stack(cols) if cols else np.empty((len(p),0))


def def_sequence(df, group, window, scaler):
    p = df[(df["Group"].astype(str)==str(group)) & (df["Day"]<=window)].copy()
    if "Defecation_observed" in p.columns:
        p = p[(p["Defecation_observed"]==1) & p["Defecation"].notna()]
    else:
        p = p[p["Defecation"].notna()]
    p = p.sort_values("Day")
    mean, sd = scaler["Defecation"]
    return (p["Defecation"].to_numpy(float)-mean)/sd


def euclidean_feature_neighbors(feature_df, test_group, train_groups, window):
    cols = [f"{s}_{suffix}" for s in ALL_SIGNALS for suffix in ("last","slope")]
    p = feature_df[feature_df["Window"]==window].copy(); p["Group"] = p["Group"].astype(str)
    train = p[p["Group"].isin(train_groups)].set_index("Group").loc[train_groups].reset_index()
    test = p[p["Group"]==str(test_group)]
    imp, sc = SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    xtr = sc.fit_transform(imp.fit_transform(train[cols])); xte = sc.transform(imp.transform(test[cols]))[0]
    d = np.linalg.norm(xtr-xte, axis=1)
    return pd.DataFrame({"NeighborGroup":train_groups,"Distance":d,"Core3_DTW_Distance":np.nan,"Defecation_Distance":np.nan,"Defecation_Method":"feature_euclidean"}).sort_values("Distance").reset_index(drop=True)


def dtw_neighbors(df, test_group, train_groups, window, included_signals, radius=1, def_weight=1.0):
    dense = [s for s in included_signals if s != "Defecation"]
    scaler = fit_fold_scaler(df, train_groups, window, included_signals)
    test_dense = dense_sequence(df, test_group, window, scaler, dense) if dense else None
    test_def = def_sequence(df, test_group, window, scaler) if "Defecation" in included_signals else None
    rows = []
    for g in train_groups:
        core_d = dtw_distance(test_dense, dense_sequence(df,g,window,scaler,dense), radius) if dense else np.nan
        def_d, def_method = np.nan, "not_used"
        if "Defecation" in included_signals:
            train_def = def_sequence(df,g,window,scaler)
            if len(test_def)==1 and len(train_def)==1:
                def_d, def_method = float(abs(test_def[0]-train_def[0])), "single_point"
            elif len(test_def) and len(train_def):
                def_d, def_method = dtw_distance(test_def,train_def,None), "dtw"
            else:
                def_method = "missing"
        if dense and "Defecation" in included_signals:
            total = core_d if pd.isna(def_d) else float(np.sqrt(core_d**2 + def_weight*def_d**2))
        elif dense:
            total = core_d
        else:
            total = def_d
        rows.append({"NeighborGroup":g,"Distance":total,"Core3_DTW_Distance":core_d,"Defecation_Distance":def_d,"Defecation_Method":def_method})
    return pd.DataFrame(rows).sort_values("Distance").reset_index(drop=True)
