import argparse, subprocess, sys

STAGES = {
    1: ["01_prepare_data.py"],
    2: ["02_features.py"],
    3: [
        "03_models_cv.py --similarity-method euclidean_feature",
        "03_models_cv.py --similarity-method dtw_core3",
        "03_models_cv.py --similarity-method dtw_all4",
        "03_models_cv_probes.py --method ridge",
        "03_models_cv_probes.py --method mixed_effects",
    ],
    4: ["04_analysis.py"],
    5: ["05_robustness.py"],
    6: ["06_msoh.py"],
    7: ["07_signal_contribution.py"],
}


def main():
    p=argparse.ArgumentParser(description="Run MSOH pipeline")
    p.add_argument("--stage",type=int,choices=range(1,8),help="只运行指定 stage；默认运行 1-7")
    a=p.parse_args(); stages=[a.stage] if a.stage else range(1,8)
    for s in stages:
        for cmd in STAGES[s]:
            args=[sys.executable,*cmd.split()]
            print("\n>"," ".join(args),flush=True)
            subprocess.run(args,check=True)


if __name__=="__main__": main()
