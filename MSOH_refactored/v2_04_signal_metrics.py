from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from config_v2 import RESULTS_V2
from src_v2.signal_metrics import validate_predictions,build_signal_metrics,build_gain_table,build_group_error_table,build_target_day_metrics

DEFAULT_INPUT=RESULTS_V2/"03_predictions"/"nested_outer_predictions.csv"
DEFAULT_OUTPUT=RESULTS_V2/"04_metrics"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args()
    if not a.input.exists(): raise FileNotFoundError(f"找不到 Phase 2 predictions: {a.input}")
    df=pd.read_csv(a.input); validate_predictions(df)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    metrics=build_signal_metrics(df)
    gains=build_gain_table(metrics)
    groups=build_group_error_table(df)
    targets=build_target_day_metrics(df)
    metrics.to_csv(a.output_dir/"signal_metrics_by_window.csv",index=False,encoding="utf-8-sig")
    gains.to_csv(a.output_dir/"signal_gain_by_transition.csv",index=False,encoding="utf-8-sig")
    groups.to_csv(a.output_dir/"group_error_by_window.csv",index=False,encoding="utf-8-sig")
    targets.to_csv(a.output_dir/"target_day_metrics.csv",index=False,encoding="utf-8-sig")

    dense={"Weight","Status_Score","Feeding"}
    for s in dense:
        q=metrics[metrics.Signal==s]
        if set(q.Window.astype(int))!={2,3,4,5,6}: raise RuntimeError(f"{s} window audit failed")
        if not (q.N_Groups==12).all() or not (q.N_TargetDays==3).all(): raise RuntimeError(f"{s} count audit failed")
    q=metrics[metrics.Signal=="Defecation"]
    if set(q.Window.astype(int))!={2,4,6}: raise RuntimeError("Defecation window audit failed")
    if not (q.N_Groups==12).all() or not (q.N_TargetDays==1).all(): raise RuntimeError("Defecation count audit failed")
    if metrics[["NRMSE","RMSE","MAE"]].isna().any().any(): raise RuntimeError("formal metrics contain NaN")

    print("="*104)
    print("MSOH V2 Phase 3 完成：signal-specific point metrics")
    print("="*104)
    print("\nError metrics:")
    print(metrics[["Signal","Window","N_Predictions","NRMSE","RMSE","MAE","Bias"]].to_string(index=False))
    print("\nPoint gains:")
    print(gains[["Signal","FromWindow","ToWindow","WindowIncrement","Gain_NRMSE","TransitionType"]].to_string(index=False))
    print("\n注意：Gain 此处仅为点估计；CI、P(Gain>delta)、stability 在 Phase 4 做 paired group bootstrap。")
    print("Defecation D2→D4 / D4→D6 是 sparse observation update，不是逐日 Gain。")
    print("\nPhase 3 checks: PASS")
    print(f"输出目录: {a.output_dir}")
    print("="*104)

if __name__=="__main__": main()
