import argparse
from pathlib import Path
from config import PREDICTION_DIR,ANALYSIS_DIR
from src.analysis import load_all_predictions,run_analysis
from src.common import ensure_dirs,save_csv
from src.plots import analysis_plots

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",type=Path,default=PREDICTION_DIR); p.add_argument("--output-root",type=Path,default=ANALYSIS_DIR); a=p.parse_args()
    table,fig=ensure_dirs(a.output_root); tables=run_analysis(load_all_predictions(a.input_dir))
    for name,df in tables.items(): save_csv(df,table/f"{name}.csv")
    analysis_plots(tables,fig)
    comp=tables["similarity_method_comparison"]; gain=tables["window_error_gain"]
    print("="*84); print("统一 04 分析完成"); print(comp[["Window","SimilarityLabel","Core3_NRMSE_Mean","Core3_TrendAccuracy"]].to_string(index=False)); print("\nError / Gain:"); print(gain[["Window","SimilarityLabel","Error_Core3","Gain_Core3","RelativeGain_Core3"]].to_string(index=False)); print("="*84)
if __name__=="__main__": main()
