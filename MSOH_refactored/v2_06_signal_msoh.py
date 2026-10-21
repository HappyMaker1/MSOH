
from __future__ import annotations
from pathlib import Path
import argparse, pandas as pd
from config_v2 import RESULTS_V2
from src_v2.decision import build_signal_decisions

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--error",type=Path,default=RESULTS_V2/"05_uncertainty"/"signal_error_bootstrap.csv")
    p.add_argument("--gain",type=Path,default=RESULTS_V2/"05_uncertainty"/"signal_gain_bootstrap.csv")
    p.add_argument("--stability",type=Path,default=RESULTS_V2/"05_uncertainty"/"signal_stability.csv")
    p.add_argument("--registry",type=Path,default=RESULTS_V2/"01_registry"/"signal_registry.csv")
    p.add_argument("--output-dir",type=Path,default=RESULTS_V2/"06_signal_decisions")
    p.add_argument("--stability-threshold",type=float,default=0.75)
    a=p.parse_args()
    for f in [a.error,a.gain,a.stability,a.registry]:
        if not f.exists(): raise FileNotFoundError(f)
    e=pd.read_csv(a.error); g=pd.read_csv(a.gain); st=pd.read_csv(a.stability); reg=pd.read_csv(a.registry)
    tr,trace,s=build_signal_decisions(e,g,st,reg,a.stability_threshold)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    tr.to_csv(a.output_dir/"signal_local_saturation.csv",index=False,encoding="utf-8-sig")
    trace.to_csv(a.output_dir/"signal_decision_trace.csv",index=False,encoding="utf-8-sig")
    s.to_csv(a.output_dir/"signal_decision_summary.csv",index=False,encoding="utf-8-sig")
    d=s[s.Signal=="Defecation"]
    if len(d)!=1 or d.iloc[0].Decision_Status!="Indeterminate_due_to_sparse_schedule" or pd.notna(d.iloc[0].Candidate_MSOH):
        raise RuntimeError("Gate B audit failed")
    if tr.Local_Saturation.isna().any(): raise RuntimeError("Local saturation contains NaN")
    print("="*112)
    print("MSOH V2 Phase 5 完成：signal-specific decisions")
    print("="*112)
    print("\nLocal saturation:")
    print(tr[["Signal","FromWindow","ToWindow","P_Gain_GT_Delta","Stable_Group_Proportion","Local_Saturation"]].to_string(index=False))
    print("\nSignal decision summary:")
    print(s[["Signal","Adequacy_Onset","Persistent_Saturation_Onset","Candidate_MSOH","Decision_Status","Last_Window_Error"]].to_string(index=False))
    print("\nGate B: Defecation remains Indeterminate_due_to_sparse_schedule.")
    print("\nPhase 5 checks: PASS")
    print(f"输出目录: {a.output_dir}")
    print("="*112)

if __name__=="__main__": main()
