from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(name):
    cmd = [sys.executable, str(ROOT / name)]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--through-phase",
        type=int,
        default=1,
        choices=range(0, 8),
    )
    args = p.parse_args()

    if args.through_phase >= 0:
        run("v2_00_freeze_baseline.py")
    if args.through_phase >= 1:
        run("v2_01_signal_registry.py")
    if args.through_phase >= 2:
        run("v2_03_signal_predictions.py")
    if args.through_phase >= 3:
        run("v2_04_signal_metrics.py")
    if args.through_phase >= 4:
        run("v2_05_uncertainty.py")
    if args.through_phase >= 5:
        run("v2_06_signal_msoh.py")
    if args.through_phase >= 6:
        run("v2_07_overall_msoh.py")
    if args.through_phase >= 7:
        run("v2_08_validation.py")


if __name__ == "__main__":
    main()
