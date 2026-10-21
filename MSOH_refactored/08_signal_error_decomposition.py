from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CORE_SIGNALS, RANDOM_SEED
from src.common import is_prediction_probe, similarity_label


def parse_args():
    p = argparse.ArgumentParser(
        description="Signal-specific prediction error decomposition for MSOH probes."
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/04_output/table/combined_predictions.csv"),
        help="04 combined predictions CSV.",
    )
    p.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("results/03_output"),
        help="Fallback directory containing predictions_loog_*.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/08_signal_diagnostics"),
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def load_predictions(path: Path, prediction_dir: Path) -> pd.DataFrame:
    required = {"Group", "Window", "Model", "Signal", "TargetDay", "y_true", "y_pred"}
    if path.exists():
        df = pd.read_csv(path)
    else:
        files = sorted(prediction_dir.glob("predictions_loog_*.csv"))
        if not files:
            raise FileNotFoundError(
                f"找不到 {path}，且 {prediction_dir} 中没有 predictions_loog_*.csv"
            )
        frames = []
        for f in files:
            x = pd.read_csv(f)
            x["SourceFile"] = f.name
            frames.append(x)
        df = pd.concat(frames, ignore_index=True).drop_duplicates()

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"预测文件缺少字段：{sorted(missing)}")

    df = df[df["Model"].map(is_prediction_probe)].copy()
    df = df[df["Signal"].isin(CORE_SIGNALS)].copy()
    df["SimilarityLabel"] = df["Model"].map(similarity_label)
    return df


def signal_nrmse(rows: pd.DataFrame) -> float:
    v = rows[rows["y_true"].notna() & rows["y_pred"].notna()]
    if len(v) < 2:
        return np.nan
    y = v["y_true"].to_numpy(float)
    pred = v["y_pred"].to_numpy(float)
    sd = float(np.std(y, ddof=1))
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return np.nan
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return rmse / sd


def point_tables(pred: pd.DataFrame):
    rows = []
    for (label, model, signal, window), p in pred.groupby(
        ["SimilarityLabel", "Model", "Signal", "Window"], sort=True
    ):
        nrmse = signal_nrmse(p)
        n = int((p["y_true"].notna() & p["y_pred"].notna()).sum())
        rows.append(
            {
                "SimilarityLabel": label,
                "Model": model,
                "Signal": signal,
                "Window": int(window),
                "N": n,
                "NRMSE": nrmse,
            }
        )
    err = pd.DataFrame(rows).sort_values(["SimilarityLabel", "Signal", "Window"])

    gain_rows = []
    for (label, model, signal), p in err.groupby(
        ["SimilarityLabel", "Model", "Signal"], sort=True
    ):
        p = p.sort_values("Window")
        prev = None
        for _, r in p.iterrows():
            e = float(r["NRMSE"]) if pd.notna(r["NRMSE"]) else np.nan
            g = prev - e if prev is not None and np.isfinite(e) and np.isfinite(prev) else np.nan
            gain_rows.append(
                {
                    "SimilarityLabel": label,
                    "Model": model,
                    "Signal": signal,
                    "Window": int(r["Window"]),
                    "NRMSE": e,
                    "Gain": g,
                }
            )
            prev = e
    gain = pd.DataFrame(gain_rows)
    return err, gain


def bootstrap_signal_errors(pred: pd.DataFrame, B: int, seed: int):
    """Group-level paired bootstrap across all windows for each model/signal.

    The same sampled group-count vector is reused across windows within a
    model/signal replicate, preserving longitudinal pairing.
    """
    rng = np.random.default_rng(seed)
    out_error = []
    out_gain = []

    for (label, model, signal), d in pred.groupby(
        ["SimilarityLabel", "Model", "Signal"], sort=True
    ):
        groups = sorted(d["Group"].dropna().unique().tolist())
        if len(groups) < 2:
            continue
        windows = sorted(int(x) for x in d["Window"].unique())
        index = {g: i for i, g in enumerate(groups)}

        # Per-group sufficient statistics for each window:
        # SSE, n, sum(y), sum(y^2) to reconstruct pooled RMSE and pooled sample SD.
        stats = {}
        for w in windows:
            a = np.zeros((len(groups), 4), dtype=float)
            q = d[(d["Window"] == w) & d["y_true"].notna() & d["y_pred"].notna()]
            for g, pg in q.groupby("Group"):
                y = pg["y_true"].to_numpy(float)
                yp = pg["y_pred"].to_numpy(float)
                i = index[g]
                a[i, 0] = np.sum((yp - y) ** 2)
                a[i, 1] = len(y)
                a[i, 2] = np.sum(y)
                a[i, 3] = np.sum(y * y)
            stats[w] = a

        counts = rng.multinomial(len(groups), [1.0 / len(groups)] * len(groups), size=B)
        boot = {w: np.full(B, np.nan) for w in windows}

        for w in windows:
            a = stats[w]
            sse = counts @ a[:, 0]
            n = counts @ a[:, 1]
            sy = counts @ a[:, 2]
            sy2 = counts @ a[:, 3]
            valid = n > 1
            var_num = sy2 - (sy * sy) / np.where(n == 0, 1, n)
            var = np.full(B, np.nan)
            var[valid] = var_num[valid] / (n[valid] - 1)
            rmse = np.full(B, np.nan)
            rmse[valid] = np.sqrt(sse[valid] / n[valid])
            sd = np.sqrt(np.maximum(var, 0))
            good = valid & np.isfinite(sd) & (sd > 1e-12)
            boot[w][good] = rmse[good] / sd[good]

            vals = boot[w][np.isfinite(boot[w])]
            out_error.append(
                {
                    "SimilarityLabel": label,
                    "Model": model,
                    "Signal": signal,
                    "Window": w,
                    "Bootstrap_B": int(len(vals)),
                    "NRMSE_Boot_Median": float(np.median(vals)) if len(vals) else np.nan,
                    "NRMSE_CI_Low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                    "NRMSE_CI_High": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                }
            )

        for w0, w1 in zip(windows[:-1], windows[1:]):
            g = boot[w0] - boot[w1]
            vals = g[np.isfinite(g)]
            out_gain.append(
                {
                    "SimilarityLabel": label,
                    "Model": model,
                    "Signal": signal,
                    "FromWindow": w0,
                    "ToWindow": w1,
                    "Bootstrap_B": int(len(vals)),
                    "Gain_Boot_Median": float(np.median(vals)) if len(vals) else np.nan,
                    "Gain_CI_Low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                    "Gain_CI_High": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                    "P_Gain_GT_0": float(np.mean(vals > 0)) if len(vals) else np.nan,
                }
            )

    return pd.DataFrame(out_error), pd.DataFrame(out_gain)


def bottleneck_summary(err: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, model, window), p in err.groupby(
        ["SimilarityLabel", "Model", "Window"], sort=True
    ):
        q = p.dropna(subset=["NRMSE"])
        if q.empty:
            continue
        worst = q.loc[q["NRMSE"].idxmax()]
        best = q.loc[q["NRMSE"].idxmin()]
        rows.append(
            {
                "SimilarityLabel": label,
                "Model": model,
                "Window": int(window),
                "Worst_Signal": worst["Signal"],
                "Worst_NRMSE": float(worst["NRMSE"]),
                "Best_Signal": best["Signal"],
                "Best_NRMSE": float(best["NRMSE"]),
                "Signal_NRMSE_Range": float(worst["NRMSE"] - best["NRMSE"]),
            }
        )
    return pd.DataFrame(rows)


def make_plots(err: pd.DataFrame, gain: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # One figure per signal; all models shown together.
    for signal in CORE_SIGNALS:
        d = err[err["Signal"] == signal]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for label, p in d.groupby("SimilarityLabel", sort=True):
            p = p.sort_values("Window")
            ax.plot(p["Window"], p["NRMSE"], marker="o", label=label)
        ax.set_xlabel("Observation window (Day k)")
        ax.set_ylabel("NRMSE (SD-normalized)")
        ax.set_title(f"Signal-specific prediction error: {signal}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"signal_error_{signal.lower()}.png", dpi=180)
        fig.savefig(fig_dir / f"signal_error_{signal.lower()}.pdf")
        plt.close(fig)

    for signal in CORE_SIGNALS:
        d = gain[(gain["Signal"] == signal) & gain["Gain"].notna()]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for label, p in d.groupby("SimilarityLabel", sort=True):
            p = p.sort_values("Window")
            ax.plot(p["Window"], p["Gain"], marker="o", label=label)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("To window (Day k)")
        ax.set_ylabel("Gain = NRMSE(k-1) - NRMSE(k)")
        ax.set_title(f"Signal-specific marginal gain: {signal}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"signal_gain_{signal.lower()}.png", dpi=180)
        fig.savefig(fig_dir / f"signal_gain_{signal.lower()}.pdf")
        plt.close(fig)


def main():
    args = parse_args()
    t0 = time.time()
    table_dir = args.output_dir / "table"
    fig_dir = args.output_dir / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.predictions, args.prediction_dir)
    err, gain = point_tables(pred)
    boot_err, boot_gain = bootstrap_signal_errors(pred, args.bootstrap, args.seed)
    bottle = bottleneck_summary(err)

    if not boot_err.empty:
        err = err.merge(
            boot_err,
            on=["SimilarityLabel", "Model", "Signal", "Window"],
            how="left",
        )
    if not boot_gain.empty:
        gain = gain.merge(
            boot_gain,
            left_on=["SimilarityLabel", "Model", "Signal", "Window"],
            right_on=["SimilarityLabel", "Model", "Signal", "ToWindow"],
            how="left",
        )

    err.to_csv(table_dir / "signal_error_by_window.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(table_dir / "signal_gain_by_window.csv", index=False, encoding="utf-8-sig")
    bottle.to_csv(table_dir / "signal_bottleneck_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(err, gain, fig_dir)

    latest = err[err["Window"] == err["Window"].max()].copy()
    latest = latest.sort_values(["SimilarityLabel", "NRMSE"], ascending=[True, False])

    print("=" * 100)
    print(f"08 单指标误差分解完成 | bootstrap B={args.bootstrap} | 用时 {time.time()-t0:.2f} 秒")
    print("\n各模型在最后一个观察窗口的单指标 NRMSE（高 = 更难预测）：")
    cols = ["SimilarityLabel", "Signal", "Window", "NRMSE", "NRMSE_CI_Low", "NRMSE_CI_High"]
    cols = [c for c in cols if c in latest.columns]
    print(latest[cols].to_string(index=False))

    print("\n每个模型/窗口的当前瓶颈指标：")
    print(bottle.to_string(index=False))
    print("\n输出：")
    print(f"  {table_dir / 'signal_error_by_window.csv'}")
    print(f"  {table_dir / 'signal_gain_by_window.csv'}")
    print(f"  {table_dir / 'signal_bottleneck_summary.csv'}")
    print(f"  {fig_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
