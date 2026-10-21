import argparse
from pathlib import Path
import pandas as pd
from config import CLEAN_PATH,FEATURE_PATH,PREDICTION_DIR,K_NEIGHBORS,DTW_RADIUS,DEFECATION_WEIGHT
from src.common import SIMILARITY_METHODS,save_csv
from src.prediction import run_loog

def basic_metrics(pred):
    rows=[]
    for (w,m,s),p in pred.groupby(["Window","Model","Signal"]):
        v=p[p["y_true"].notna()&p["y_pred"].notna()]
        if v.empty: continue
        import numpy as np
        e=(v["y_pred"]-v["y_true"]).to_numpy(float); rows.append({"Window":w,"Model":m,"Signal":s,"N":len(v),"MAE":float(np.mean(abs(e))),"RMSE":float(np.sqrt(np.mean(e**2)))})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=CLEAN_PATH); p.add_argument("--features",type=Path,default=FEATURE_PATH); p.add_argument("--output-dir",type=Path,default=PREDICTION_DIR); p.add_argument("--similarity-method",choices=SIMILARITY_METHODS,default="euclidean_feature"); p.add_argument("--k",type=int,default=K_NEIGHBORS); p.add_argument("--dtw-radius",type=int,default=DTW_RADIUS); p.add_argument("--defecation-weight",type=float,default=DEFECATION_WEIGHT); a=p.parse_args()
    data=pd.read_csv(a.data); feat=pd.read_csv(a.features); pred,nei=run_loog(data,feat,a.similarity_method,a.k,a.dtw_radius,a.defecation_weight); a.output_dir.mkdir(parents=True,exist_ok=True); suf=a.similarity_method
    save_csv(pred,a.output_dir/f"predictions_loog_{suf}.csv"); save_csv(nei,a.output_dir/f"similarity_neighbors_{suf}.csv"); save_csv(basic_metrics(pred),a.output_dir/f"basic_metrics_{suf}.csv"); print(f"03 完成：{suf}")
if __name__=="__main__": main()
