import argparse
from pathlib import Path
import pandas as pd
from config import CLEAN_PATH,WINDOWS
from src.features import validate_clean_data,build_feature_table,feature_audit
from src.common import save_csv

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=CLEAN_PATH); p.add_argument("--output-dir",type=Path,default=Path("results/02_output")); p.add_argument("--use-delta",action="store_true"); a=p.parse_args()
    df=pd.read_csv(a.input); validate_clean_data(df); feat=build_feature_table(df,WINDOWS,a.use_delta); save_csv(feat,a.output_dir/"early_features.csv"); save_csv(feature_audit(feat),a.output_dir/"feature_audit.csv"); print(f"02 完成：{len(feat)} group-window rows")
if __name__=="__main__": main()
