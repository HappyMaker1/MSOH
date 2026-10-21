import numpy as np
import pandas as pd
from config import ALL_SIGNALS, CORE_SIGNALS, TARGET_DAYS, WINDOWS, EPS
from .common import model_name
from .distance import euclidean_feature_neighbors, dtw_neighbors


def get_truth(df, group, target_days=TARGET_DAYS):
    out = {}
    p = df[(df["Group"].astype(str)==str(group)) & df["Day"].isin(target_days)]
    for s in ALL_SIGNALS:
        for day in target_days:
            row = p[p["Day"]==day]
            out[(s,day)] = float(row.iloc[0][s]) if len(row) and not pd.isna(row.iloc[0][s]) else np.nan
    return out


def predict_locf(df, group, window, target_days=TARGET_DAYS):
    p = df[(df["Group"].astype(str)==str(group)) & (df["Day"]<=window)].sort_values("Day")
    out = {}
    for s in ALL_SIGNALS:
        valid = p.dropna(subset=[s]); value = float(valid.iloc[-1][s]) if len(valid) else np.nan
        for day in target_days: out[(s,day)] = value
    return out


def predict_linear(df, group, window, target_days=TARGET_DAYS):
    p = df[(df["Group"].astype(str)==str(group)) & (df["Day"]<=window)].sort_values("Day")
    out = {}
    for s in ALL_SIGNALS:
        valid = p.dropna(subset=[s])
        if len(valid)>=2:
            slope, intercept = np.polyfit(valid["Day"].to_numpy(float), valid[s].to_numpy(float),1)
            vals = {d:float(intercept+slope*d) for d in target_days}
        elif len(valid)==1:
            vals = {d:float(valid.iloc[0][s]) for d in target_days}
        else:
            vals = {d:np.nan for d in target_days}
        out.update({(s,d):v for d,v in vals.items()})
    return out


def predict_global_mean(df, train_groups, target_days=TARGET_DAYS):
    train = df[df["Group"].astype(str).isin(train_groups) & df["Day"].isin(target_days)]
    out = {}
    for s in ALL_SIGNALS:
        for d in target_days:
            v = train.loc[train["Day"]==d,s].dropna(); out[(s,d)] = float(v.mean()) if len(v) else np.nan
    return out


def predict_from_neighbors(df, neighbors, k=2, target_days=TARGET_DAYS):
    selected = neighbors.head(k).copy()
    selected["RawWeight"] = 1.0/(selected["Distance"]+EPS)
    selected["Weight"] = selected["RawWeight"]/selected["RawWeight"].sum()
    out = {}
    for s in ALL_SIGNALS:
        for day in target_days:
            vals, weights = [], []
            for _, r in selected.iterrows():
                target = df[(df["Group"].astype(str)==str(r["NeighborGroup"])) & (df["Day"]==day)]
                if not len(target) or pd.isna(target.iloc[0][s]): continue
                vals.append(float(target.iloc[0][s])); weights.append(float(r["Weight"]))
            if vals:
                w=np.asarray(weights,float); w/=w.sum(); out[(s,day)] = float(np.sum(np.asarray(vals)*w))
            else: out[(s,day)] = np.nan
    return out, selected


def _append(rows, group, window, model, truth, pred, target_days=TARGET_DAYS):
    for s in ALL_SIGNALS:
        for day in target_days:
            yt, yp = truth[(s,day)], pred[(s,day)]
            err = yp-yt if not pd.isna(yt) and not pd.isna(yp) else np.nan
            rows.append({"Group":group,"Window":window,"Model":model,"Signal":s,"TargetDay":day,"y_true":yt,"y_pred":yp,"error":err,"abs_error":abs(err) if not pd.isna(err) else np.nan})


def run_loog(data_df, feature_df, method, k=2, radius=1, def_weight=1.0, windows=WINDOWS, target_days=TARGET_DAYS):
    groups = sorted(data_df["Group"].astype(str).unique()); rows, neighbor_rows = [], []
    sim_model = model_name(method,k)
    for test in groups:
        train = [g for g in groups if g!=test]; truth = get_truth(data_df,test,target_days)
        global_mean = predict_global_mean(data_df,train,target_days)
        for window in windows:
            if method=="euclidean_feature":
                neighbors = euclidean_feature_neighbors(feature_df,test,train,window)
            else:
                included = CORE_SIGNALS if method=="dtw_core3" else ALL_SIGNALS
                neighbors = dtw_neighbors(data_df,test,train,window,included,radius,def_weight)
            sim_pred, selected = predict_from_neighbors(data_df,neighbors,k,target_days)
            _append(rows,test,window,sim_model,truth,sim_pred,target_days)
            _append(rows,test,window,"LOCF",truth,predict_locf(data_df,test,window,target_days),target_days)
            _append(rows,test,window,"Linear",truth,predict_linear(data_df,test,window,target_days),target_days)
            _append(rows,test,window,"GlobalMean",truth,global_mean,target_days)
            for rank, (_, r) in enumerate(selected.iterrows(),1):
                neighbor_rows.append({"Group":test,"Window":window,"Model":sim_model,"Rank":rank,**r.to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(neighbor_rows)


def run_signal_variant(data_df, included_signals, variant, k=2, radius=1, def_weight=1.0, windows=WINDOWS, target_days=TARGET_DAYS):
    groups = sorted(data_df["Group"].astype(str).unique()); pred_rows, neighbor_rows = [], []
    for window in windows:
        for test in groups:
            train=[g for g in groups if g!=test]
            neighbors = dtw_neighbors(data_df,test,train,window,included_signals,radius,def_weight)
            pred, selected = predict_from_neighbors(data_df,neighbors,k,target_days)
            truth = get_truth(data_df,test,target_days)
            for s in CORE_SIGNALS:
                for day in target_days:
                    pred_rows.append({"Variant":variant,"IncludedSignals":"|".join(included_signals),"Group":test,"Window":window,"Signal":s,"TargetDay":day,"y_true":truth[(s,day)],"y_pred":pred[(s,day)]})
            for rank,(_,r) in enumerate(selected.iterrows(),1):
                neighbor_rows.append({"Variant":variant,"Window":window,"HeldOutGroup":test,"NeighborRank":rank,"NeighborGroup":r["NeighborGroup"],"Distance":r["Distance"]})
    return pd.DataFrame(pred_rows), pd.DataFrame(neighbor_rows)
