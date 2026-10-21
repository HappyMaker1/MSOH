from pathlib import Path
import numpy as np
import pandas as pd
from config import ALL_SIGNALS, CORE_SIGNALS
from .prediction import run_signal_variant
from .bootstrap import bootstrap_entities,error_gain_tables,group_errors,stability_table
from .msoh import decide_msoh

VARIANTS={
    "All4":ALL_SIGNALS,
    "Minus_Weight":["Status_Score","Feeding","Defecation"],
    "Minus_Status_Score":["Weight","Feeding","Defecation"],
    "Minus_Feeding":["Weight","Status_Score","Defecation"],
    "Minus_Defecation":["Weight","Status_Score","Feeding"],
}


def validate_all4_baseline(all4,reference_path:Path,tol=1e-6):
    if not reference_path.exists(): raise FileNotFoundError(reference_path)
    ref=pd.read_csv(reference_path); ref["Group"]=ref["Group"].astype(str)
    ref=ref[(ref["Model"]=="Similarity_DTW_All4_K2")&ref["Signal"].isin(CORE_SIGNALS)]
    keys=["Group","Window","Signal","TargetDay"]
    a=all4[keys+["y_true","y_pred"]].rename(columns={"y_true":"yt_a","y_pred":"yp_a"}); b=ref[keys+["y_true","y_pred"]].drop_duplicates(keys).rename(columns={"y_true":"yt_b","y_pred":"yp_b"})
    c=a.merge(b,on=keys,how="outer",indicator=True,validate="one_to_one")
    if not (c["_merge"]=="both").all(): raise RuntimeError("All4 baseline 行不一致")
    dy=float(np.nanmax(np.abs(c["yt_a"]-c["yt_b"]))); dp=float(np.nanmax(np.abs(c["yp_a"]-c["yp_b"]))); passed=dy<=tol and dp<=tol
    audit=pd.DataFrame([{"ReferenceFile":str(reference_path),"N_Rows_Compared":len(c),"Tolerance":tol,"MaxAbsDiff_y_true":dy,"MaxAbsDiff_y_pred":dp,"Baseline_PASS":passed}])
    if not passed: raise RuntimeError(f"Baseline validation FAIL: max |Δy_pred|={dp}")
    return audit


def run_loso(df,k=2,radius=1,def_weight=1.0,windows=None,target_days=None):
    preds=[]; neigh=[]
    kwargs={}
    if windows is not None: kwargs["windows"]=windows
    if target_days is not None: kwargs["target_days"]=target_days
    for name,signals in VARIANTS.items():
        p,n=run_signal_variant(df,signals,name,k,radius,def_weight,**kwargs); preds.append(p); neigh.append(n)
    return pd.concat(preds,ignore_index=True),pd.concat(neigh,ignore_index=True)


def contribution_table(store,delta=.03):
    base=store["All4"]; rows=[]
    for variant,info in store.items():
        if variant=="All4": continue
        removed=variant.replace("Minus_","")
        for i,w in enumerate(base["windows"]):
            boot=info["error_boot"][:,i]-base["error_boot"][:,i]; boot=boot[~np.isnan(boot)]; lo,hi=np.percentile(boot,[2.5,97.5]); point=info["point_errors"][i]-base["point_errors"][i]
            rows.append({"RemovedSignal":removed,"Variant":variant,"Window":w,"Error_Without_Signal":info["point_errors"][i],"Error_All4":base["point_errors"][i],"Contribution_C_s":point,"Contribution_CI_Low":lo,"Contribution_CI_High":hi,"P_C_GT_0":float(np.mean(boot>0)),"Contribution_Delta":delta,"P_C_GT_Delta":float(np.mean(boot>delta))})
    return pd.DataFrame(rows)


def msoh_shift(msoh):
    base=msoh.loc[msoh["Variant"]=="All4","Candidate_MSOH"].iloc[0]; rows=[]
    for _,r in msoh[msoh["Variant"]!="All4"].iterrows():
        cur=r["Candidate_MSOH"]; shift=np.nan if pd.isna(base) or pd.isna(cur) else float(cur-base)
        rows.append({"RemovedSignal":r["Variant"].replace("Minus_",""),"Variant":r["Variant"],"MSOH_All4":base,"MSOH_Without_Signal":cur,"Delta_MSOH_s":shift})
    return pd.DataFrame(rows)


def contribution_summary(contrib,shift):
    rows=[]
    for s,p in contrib.groupby("RemovedSignal"):
        peak=p.loc[p["Contribution_C_s"].idxmax()]; sh=shift.loc[shift["RemovedSignal"]==s,"Delta_MSOH_s"]
        rows.append({"Signal":s,"Mean_Contribution_C_s":float(p["Contribution_C_s"].mean()),"Max_Positive_Contribution_C_s":float(peak["Contribution_C_s"]),"Max_Positive_Window":int(peak["Window"]),"P_C_GT_0_At_Max_Positive":float(peak["P_C_GT_0"]),"CI_Low_At_Max_Positive":float(peak["Contribution_CI_Low"]),"CI_High_At_Max_Positive":float(peak["Contribution_CI_High"]),"Delta_MSOH_s":sh.iloc[0] if len(sh) else np.nan})
    return pd.DataFrame(rows)
