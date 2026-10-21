
from __future__ import annotations
import numpy as np
import pandas as pd

def persistent_at(k, windows, local):
    i=windows.index(int(k))
    if i>=len(windows)-1: return False
    req=[]
    for j in range(i,len(windows)-1):
        key=(int(windows[j]),int(windows[j+1]))
        if key not in local: return False
        req.append(bool(local[key]))
    return bool(req and all(req))

def build_signal_decisions(error_table,gain_table,stability_table,registry,tau_s=0.75):
    trans=[]; trace=[]; summary=[]
    for _,spec in registry.iterrows():
        s=str(spec["name"])
        e=error_table[error_table.Signal.astype(str)==s].sort_values("Window").copy()
        g=gain_table[gain_table.Signal.astype(str)==s].copy()
        st=stability_table[stability_table.Signal.astype(str)==s].copy()
        if e.empty: raise ValueError(f"{s}: missing error table")
        tau_e=float(spec["error_threshold"]); delta=float(spec["gain_delta"]); alpha=float(spec["alpha"])
        windows=e.Window.astype(int).tolist()
        m=g.merge(st[["Signal","FromWindow","ToWindow","Stable_Group_Proportion"]],
                  on=["Signal","FromWindow","ToWindow"],how="left",validate="one_to_one")
        local={}
        for _,r in m.iterrows():
            a,b=int(r.FromWindow),int(r.ToWindow)
            p=float(r.P_Gain_GT_Delta); stab=float(r.Stable_Group_Proportion)
            pc=bool(np.isfinite(p) and p<alpha)
            sc=bool(np.isfinite(stab) and stab>=tau_s)
            ls=bool(pc and sc); local[(a,b)]=ls
            trans.append({"Signal":s,"FromWindow":a,"ToWindow":b,
                          "P_Gain_GT_Delta":p,"Gain_Delta":delta,"Alpha":alpha,
                          "Gain_Probability_Condition":pc,
                          "Stable_Group_Proportion":stab,"Stability_Threshold":tau_s,
                          "Stability_Condition":sc,"Local_Saturation":ls})
        adequacy=[]; ps=[]; cand=[]
        for _,r in e.iterrows():
            w=int(r.Window); ep=float(r.Error_Point)
            ad=bool(ep<=tau_e); per=persistent_at(w,windows,local)
            eligible=(w!=windows[-1])
            if ad: adequacy.append(w)
            if per: ps.append(w)
            if ad and per and eligible: cand.append(w)
            trace.append({"Signal":s,"Window":w,"Error_Point":ep,
                          "Error_CI_Low":float(r.Error_CI_Low),"Error_CI_High":float(r.Error_CI_High),
                          "Error_Threshold":tau_e,"Adequate":ad,
                          "Persistent_Saturation":per,"Eligible_For_MSOH":eligible,
                          "Candidate_At_Window":bool(ad and per and eligible)})
        ao=min(adequacy) if adequacy else np.nan
        po=min(ps) if ps else np.nan
        dc=min(cand) if cand else np.nan
        sparse=(str(spec["density"])=="planned_sparse" or s=="Defecation")
        if sparse:
            cm=np.nan; status="Indeterminate_due_to_sparse_schedule"
            reason=("Sparse design: only one real future target (Day9), and no later sparse update "
                    "after the final effective window; finite Defecation MSOH is not assigned.")
        else:
            cm=dc
            if np.isfinite(cm):
                status="Identified"; reason="Adequacy and persistent saturation coincide."
            elif np.isfinite(ao) and np.isfinite(po):
                status="Adequate_and_saturated_but_no_common_window"; reason="Adequacy and persistent saturation do not coincide."
            elif np.isfinite(ao):
                status="Adequate_not_saturated"; reason="Adequacy reached; persistent saturation not established."
            elif np.isfinite(po):
                status="Saturated_inadequate"; reason="Persistent saturation established; adequacy not reached."
            else:
                status="Not_reached"; reason="Neither adequacy nor persistent saturation established within the evaluated horizon."
        summary.append({"Signal":s,"Role":str(spec["role"]),"Density":str(spec["density"]),
                        "Error_Threshold":tau_e,"Gain_Delta":delta,"Alpha":alpha,
                        "Stability_Threshold":tau_s,"Adequacy_Onset":ao,
                        "Persistent_Saturation_Onset":po,"Diagnostic_Candidate_MSOH":dc,
                        "Candidate_MSOH":cm,"Last_Window":int(windows[-1]),
                        "Last_Window_Error":float(e.iloc[-1].Error_Point),
                        "Decision_Status":status,"Reason":reason})
    return pd.DataFrame(trans),pd.DataFrame(trace),pd.DataFrame(summary)
