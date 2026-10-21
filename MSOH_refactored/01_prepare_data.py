import argparse
from pathlib import Path
from config import DATA_PATH
from src.preprocessing import load_raw,prepare_data,group_day_audit,missingness_by_group,audit_summary
from src.common import save_csv

def main():
    p=argparse.ArgumentParser(); 
    p.add_argument("--input",type=Path,default=DATA_PATH); 
    p.add_argument("--sheet",default=0); 
    p.add_argument("--output-dir",type=Path,default=Path("results/01_output")); a=p.parse_args()
    df=prepare_data(load_raw(a.input,a.sheet)); 
    a.output_dir.mkdir(parents=True,exist_ok=True)
    save_csv(df,a.output_dir/"animal_data_clean.csv"); save_csv(audit_summary(df),a.output_dir/"data_audit_summary.csv"); 
    save_csv(group_day_audit(df),a.output_dir/"group_day_audit.csv"); 
    save_csv(missingness_by_group(df),a.output_dir/"missingness_by_group.csv")
    print(f"01 完成：{df['Group'].nunique()} groups, Day {df['Day'].min()}-{df['Day'].max()}, {len(df)} rows")
if __name__=="__main__": main()
