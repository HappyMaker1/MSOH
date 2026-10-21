from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .common import is_prediction_probe


def savefig(fig,out:Path,stem):
    out.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(out/f"{stem}.png",dpi=200,bbox_inches="tight"); fig.savefig(out/f"{stem}.pdf",bbox_inches="tight"); plt.close(fig)


def analysis_plots(tables,out):
    comp=tables["similarity_method_comparison"]; gain=tables["window_error_gain"]; overall=tables["overall_model_metrics"]; pred=tables["combined_predictions"]
    fig,ax=plt.subplots(figsize=(9,5))
    for label,p in comp.groupby("SimilarityLabel"): ax.plot(p["Window"],p["Core3_NRMSE_Mean"],marker="o",label=label)
    ax.set(xlabel="Observation Window End Day",ylabel="Core3 Mean NRMSE",title="Prediction Probes: Error(k)"); ax.legend(); savefig(fig,out,"fig01_prediction_probe_error")
    fig,ax=plt.subplots(figsize=(9,5))
    for label,p in gain.groupby("SimilarityLabel"): ax.plot(p["Window"],p["Gain_Core3"],marker="o",label=label)
    ax.axhline(0,linewidth=1); ax.set(xlabel="Observation Window End Day",ylabel="Gain(k)",title="Prediction Probes: Gain(k)"); ax.legend(); savefig(fig,out,"fig02_prediction_probe_gain")
    fig,ax=plt.subplots(figsize=(9,5))
    for label,p in comp.groupby("SimilarityLabel"): ax.plot(p["Window"],p["Core3_TrendAccuracy"],marker="o",label=label)
    ax.set(xlabel="Observation Window End Day",ylabel="Core3 Trend Accuracy",title="Prediction Probes: Trend Accuracy"); ax.legend(); savefig(fig,out,"fig03_prediction_probe_trend")
    fig,ax=plt.subplots(figsize=(10,5))
    for model,p in overall.groupby("Model"): ax.plot(p["Window"],p["Core3_NRMSE_Mean"],marker="o",label=model)
    ax.set(xlabel="Observation Window End Day",ylabel="Core3 Mean NRMSE",title="All Models"); ax.legend(fontsize=8); savefig(fig,out,"fig04_all_models")
    mask=pred["Model"].map(is_prediction_probe)
    for signal in ["Weight","Status_Score","Feeding"]:
        p=pred[(pred["Window"]==4)&(pred["Signal"]==signal)&mask&pred["y_true"].notna()&pred["y_pred"].notna()]
        if p.empty: continue
        fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(p["y_true"],p["y_pred"]); lo=min(p["y_true"].min(),p["y_pred"].min()); hi=max(p["y_true"].max(),p["y_pred"].max()); ax.plot([lo,hi],[lo,hi]); ax.set(xlabel="True",ylabel="Predicted",title=f"True vs Predicted: {signal}, Day4"); savefig(fig,out,f"fig05_true_vs_pred_day4_{signal.lower()}")


def robustness_plots(error,gain,paired,summary,out):
    fig,ax=plt.subplots(figsize=(9,5))
    for label,p in error.groupby("SimilarityLabel"):
        y=p["Error_Point"].to_numpy(float); ax.errorbar(p["Window"],y,yerr=np.vstack([y-p["Error_CI_Low"],p["Error_CI_High"]-y]),marker="o",capsize=3,label=label)
    ax.set(xlabel="Observation Window End Day",ylabel="Core3 Mean NRMSE",title="Bootstrap 95% CI for Error(k)"); ax.legend(); savefig(fig,out,"fig01_error_ci")
    fig,ax=plt.subplots(figsize=(9,5))
    for label,p in gain.groupby("SimilarityLabel"):
        y=p["Gain_Point"].to_numpy(float); ax.errorbar(p["ToWindow"],y,yerr=np.vstack([y-p["Gain_CI_Low"],p["Gain_CI_High"]-y]),marker="o",capsize=3,label=label)
    ax.axhline(0,linewidth=1); ax.set(xlabel="To Window",ylabel="Gain",title="Bootstrap 95% CI for Gain"); ax.legend(); savefig(fig,out,"fig02_gain_ci")


def msoh_plots(error,gain,summary,out,alpha=.1):
    fig,ax=plt.subplots(figsize=(9,5))
    for entity,p in error.groupby("SimilarityLabel"): ax.plot(p["Window"],p["Error_Point"],marker="o",label=entity)
    ax.set(xlabel="Window",ylabel="Core3 Mean NRMSE",title="MSOH Error Curves Across Prediction Probes"); ax.legend(); savefig(fig,out,"fig01_decision_trace")
    fig,ax=plt.subplots(figsize=(9,5))
    for entity,p in gain.groupby("SimilarityLabel"): ax.plot(p["ToWindow"],p["P_Gain_GT_Delta"],marker="o",label=entity)
    ax.axhline(alpha,linewidth=1); ax.set_ylim(0,1); ax.set(xlabel="Next Window",ylabel="P(Gain > delta)",title="Meaningful Future Gain Probability"); ax.legend(); savefig(fig,out,"fig02_gain_probability")


def contribution_plots(contrib,out):
    for signal,p in contrib.groupby("RemovedSignal"):
        y=p["Contribution_C_s"].to_numpy(float); fig,ax=plt.subplots(figsize=(8,5)); ax.errorbar(p["Window"],y,yerr=np.vstack([y-p["Contribution_CI_Low"],p["Contribution_CI_High"]-y]),marker="o",capsize=3); ax.axhline(0,linewidth=1); ax.set(xlabel="Window",ylabel="C_s(k)",title=f"Signal Contribution: {signal}"); savefig(fig,out,f"fig05_contribution_ci_{signal.lower()}")
    fig,ax=plt.subplots(figsize=(8,5))
    for signal,p in contrib.groupby("RemovedSignal"): ax.plot(p["Window"],p["P_C_GT_0"],marker="o",label=signal)
    ax.set_ylim(0,1); ax.set(xlabel="Window",ylabel="P(C_s(k)>0)",title="Positive Signal Contribution Probability"); ax.legend(); savefig(fig,out,"fig06_contribution_probability")
