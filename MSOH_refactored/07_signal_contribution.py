import argparse
from pathlib import Path
import pandas as pd
from config import CLEAN_PATH,ANALYSIS_DIR,ATTRIBUTION_DIR,K_NEIGHBORS,DTW_RADIUS,DEFECATION_WEIGHT,BOOTSTRAP_B,RANDOM_SEED,BASELINE_TOL,GAIN_DELTA,ERROR_THRESHOLD,ALPHA,STABILITY_THRESHOLD,GROUP_EPSILON,CONTRIBUTION_DELTA
from src.common import ensure_dirs,save_csv
from src.attribution import run_loso,validate_all4_baseline,contribution_table,msoh_shift,contribution_summary,VARIANTS
from src.bootstrap import bootstrap_entities,error_gain_tables,group_errors,stability_table
from src.msoh import decide_msoh
from src.plots import contribution_plots

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=CLEAN_PATH); p.add_argument("--reference-predictions",type=Path,default=ANALYSIS_DIR/"table/combined_predictions.csv"); p.add_argument("--output",type=Path,default=ATTRIBUTION_DIR); p.add_argument("--baseline-tol",type=float,default=BASELINE_TOL); p.add_argument("--k",type=int,default=K_NEIGHBORS); p.add_argument("--dtw-radius",type=int,default=DTW_RADIUS); p.add_argument("--defecation-weight",type=float,default=DEFECATION_WEIGHT); p.add_argument("--bootstrap",type=int,default=BOOTSTRAP_B); p.add_argument("--seed",type=int,default=RANDOM_SEED); p.add_argument("--error-threshold",type=float,default=ERROR_THRESHOLD); p.add_argument("--gain-delta",type=float,default=GAIN_DELTA); p.add_argument("--alpha",type=float,default=ALPHA); p.add_argument("--stability-threshold",type=float,default=STABILITY_THRESHOLD); p.add_argument("--group-epsilon",type=float,default=GROUP_EPSILON); p.add_argument("--contribution-delta",type=float,default=CONTRIBUTION_DELTA); a=p.parse_args()
    print("[1/6] 读取数据"); df=pd.read_csv(a.input); print("[2/6] 运行 All4 + LOSO"); pred,nei=run_loso(df,a.k,a.dtw_radius,a.defecation_weight); print("[3/6] All4 baseline validation"); audit=validate_all4_baseline(pred[pred["Variant"]=="All4"],a.reference_predictions,a.baseline_tol); print(f"PASS: max |Δy_pred|={audit.iloc[0]['MaxAbsDiff_y_pred']:.3e}")
    print("[4/6] paired Group bootstrap"); store=bootstrap_entities(pred,"Variant",a.bootstrap,a.seed,shared_counts=True); error,gain=error_gain_tables(store,a.gain_delta,"Variant"); ge=group_errors(pred,"Variant"); stable=stability_table(ge,"Variant",a.group_epsilon); trace,msoh=decide_msoh(error,gain,stable,"Variant",a.error_threshold,a.gain_delta,a.alpha,a.stability_threshold)
    print("[5/6] 信号贡献"); contrib=contribution_table(store,a.contribution_delta); shift=msoh_shift(msoh); summary=contribution_summary(contrib,shift); print("[6/6] 保存")
    table,fig=ensure_dirs(a.output); files={"baseline_validation":audit,"signal_variant_predictions":pred,"signal_variant_neighbors":nei,"signal_variant_error":error,"signal_variant_gain":gain,"signal_variant_stability":stable,"signal_variant_msoh_trace":trace,"signal_variant_msoh":msoh,"signal_contribution_by_window_bootstrap":contrib,"signal_msoh_shift":shift,"signal_contribution_summary_bootstrap":summary}
    for n,d in files.items(): save_csv(d,table/f"{n}.csv")
    contribution_plots(contrib,fig); print("\n各变体候选 MSOH："); print(msoh[["Variant","Candidate_MSOH","Decision_Status"]].to_string(index=False)); print("\nBootstrap 信号贡献摘要："); print(summary.to_string(index=False))
if __name__=="__main__": main()
