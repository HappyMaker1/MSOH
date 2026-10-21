from __future__ import annotations
import numpy as np
import pandas as pd

REQ={"Signal","Window","OuterGroup","TargetDay","SelectedModel","y_true","y_pred"}

def validate_predictions(df):
    miss=sorted(REQ-set(df.columns))
    if miss: raise ValueError(f"缺少字段：{miss}")
    if df.empty: raise ValueError("prediction file 为空")
    keys=["Signal","Window","OuterGroup","TargetDay"]
    if df.duplicated(keys).any(): raise ValueError("发现重复 signal/window/group/target")
    if df[["y_true","y_pred"]].isna().any().any(): raise ValueError("y_true/y_pred 存在缺失")

def metric(part):
    y=part["y_true"].to_numpy(float); p=part["y_pred"].to_numpy(float)
    e=p-y
    rmse=float(np.sqrt(np.mean(e*e)))
    mae=float(np.mean(np.abs(e)))
    bias=float(np.mean(e))
    sd=float(np.std(y,ddof=1)) if len(y)>=2 else np.nan
    nrmse=float(rmse/sd) if np.isfinite(sd) and not np.isclose(sd,0) else np.nan
    return {"N":len(y),"RMSE":rmse,"MAE":mae,"Bias":bias,"Target_SD":sd,"NRMSE":nrmse}

def build_signal_metrics(df):
    rows=[]
    for (s,w),p in df.groupby(["Signal","Window"],sort=True):
        m=metric(p)
        vc=(p[["OuterGroup","SelectedModel"]].drop_duplicates()["SelectedModel"].value_counts())
        rows.append({"Signal":s,"Window":int(w),"N_Predictions":len(p),
                     "N_Groups":p["OuterGroup"].nunique(),"N_TargetDays":p["TargetDay"].nunique(),
                     **m,"SelectedModelDistribution":";".join(f"{k}:{int(v)}" for k,v in vc.items())})
    return pd.DataFrame(rows).sort_values(["Signal","Window"]).reset_index(drop=True)

def build_gain_table(metrics):
    rows=[]
    for s,p in metrics.groupby("Signal",sort=True):
        p=p.sort_values("Window").reset_index(drop=True)
        for i in range(len(p)-1):
            a,b=p.iloc[i],p.iloc[i+1]
            fw,tw=int(a.Window),int(b.Window)
            rows.append({"Signal":s,"FromWindow":fw,"ToWindow":tw,
                         "WindowIncrement":tw-fw,
                         "NRMSE_From":a.NRMSE,"NRMSE_To":b.NRMSE,
                         "Gain_NRMSE":float(a.NRMSE-b.NRMSE),
                         "RMSE_From":a.RMSE,"RMSE_To":b.RMSE,
                         "Gain_RMSE":float(a.RMSE-b.RMSE),
                         "MAE_From":a.MAE,"MAE_To":b.MAE,
                         "Gain_MAE":float(a.MAE-b.MAE),
                         "TransitionType":"adjacent_day" if tw-fw==1 else "sparse_observation_update"})
    return pd.DataFrame(rows)

def build_group_error_table(df):
    rows=[]
    for (s,w),p in df.groupby(["Signal","Window"],sort=True):
        yall=p["y_true"].to_numpy(float)
        sd=float(np.std(yall,ddof=1)) if len(yall)>=2 else np.nan
        for g,q in p.groupby("OuterGroup",sort=True):
            e=q["y_pred"].to_numpy(float)-q["y_true"].to_numpy(float)
            rmse=float(np.sqrt(np.mean(e*e)))
            gn=float(rmse/sd) if np.isfinite(sd) and not np.isclose(sd,0) else np.nan
            rows.append({"Signal":s,"Window":int(w),"OuterGroup":str(g),
                         "N_Targets":len(q),"Group_RMSE":rmse,"Target_SD":sd,"Group_NRMSE":gn})
    return pd.DataFrame(rows)

def build_target_day_metrics(df):
    rows=[]
    for (s,w,d),p in df.groupby(["Signal","Window","TargetDay"],sort=True):
        rows.append({"Signal":s,"Window":int(w),"TargetDay":int(d),**metric(p)})
    return pd.DataFrame(rows)
