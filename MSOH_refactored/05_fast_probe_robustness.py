from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config import GAIN_DELTA, RANDOM_SEED
from src.bootstrap import bootstrap_entities, error_gain_tables


DEFAULT_FILES = {
    "Ridge Direct": Path("results/03_output/predictions_loog_ridge.csv"),
    "Mixed Effects": Path("results/03_output/predictions_loog_mixed_effects.csv"),
}


def load_probe_predictions() -> pd.DataFrame:
    frames = []
    missing = []
    for label, path in DEFAULT_FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        required = {"Group", "Window", "Signal", "y_true", "y_pred"}
        absent = required.difference(df.columns)
        if absent:
            raise ValueError(f"{path} 缺少必要列: {sorted(absent)}")
        df = df.copy()
        df["Probe"] = label
        frames.append(df)

    if missing:
        raise FileNotFoundError(
            "找不到以下 probe 预测文件，请先运行 03_models_cv_probes.py:\n- "
            + "\n- ".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast group-level bootstrap for Ridge and Mixed Effects probes only."
    )
    parser.add_argument("--b", type=int, default=2000, help="bootstrap 次数，默认 2000")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    parser.add_argument(
        "--gain-delta", type=float, default=GAIN_DELTA,
        help="有意义的最小 Gain 阈值，默认读取 config.GAIN_DELTA",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/05_fast_probe_output")
    )
    args = parser.parse_args()

    if args.b < 100:
        raise ValueError("--b 建议至少 100；诊断推荐 2000，最终分析可用 5000+。")

    t0 = time.perf_counter()
    pred = load_probe_predictions()

    # Aggregate group-level bootstrap. The same sampled group counts are reused
    # across windows within each probe so Gain(k) is paired across adjacent windows.
    store = bootstrap_entities(
        pred,
        entity_col="Probe",
        b=args.b,
        seed=args.seed,
        shared_counts=False,
    )
    err, gain = error_gain_tables(
        store,
        gain_delta=args.gain_delta,
        entity_name="Probe",
    )

    # A concise diagnostic flag. This is deliberately NOT a full MSOH decision:
    # it only asks whether the probability of a meaningful additional gain is low.
    gain["P_Gain_LE_Delta"] = 1.0 - gain["P_Gain_GT_Delta"]
    gain["Saturation_Signal_alpha0.10"] = gain["P_Gain_GT_Delta"] < 0.10

    args.output_dir.mkdir(parents=True, exist_ok=True)
    err_path = args.output_dir / "probe_error_bootstrap_fast.csv"
    gain_path = args.output_dir / "probe_gain_bootstrap_fast.csv"
    err.to_csv(err_path, index=False, encoding="utf-8-sig")
    gain.to_csv(gain_path, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - t0
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("=" * 100)
    print(f"Fast probe bootstrap 完成 | B={args.b} | seed={args.seed} | 用时 {elapsed:.2f} 秒")
    print("仅分析 Ridge Direct 和 Mixed Effects；不会覆盖正式 05 输出。")
    print("\nError bootstrap:")
    print(err.to_string(index=False))
    print("\nGain bootstrap:")
    cols = [
        "Probe", "FromWindow", "ToWindow", "Gain_Point",
        "Gain_CI_Low", "Gain_CI_High", "P_Gain_GT_0",
        "Gain_Delta", "P_Gain_GT_Delta", "Saturation_Signal_alpha0.10",
    ]
    print(gain[cols].to_string(index=False))
    print("\n输出:")
    print(f"  {err_path}")
    print(f"  {gain_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
