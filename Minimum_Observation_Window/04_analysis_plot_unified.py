"""
04_analysis_plot.py

统一分析版：
自动扫描 results/03_output/predictions_loog_*.csv
并比较不同 Similarity 方法。

兼容当前统一 03_models_cv.py 输出：
- predictions_loog_euclidean_feature.csv
- predictions_loog_dtw_core3.csv
- predictions_loog_dtw_all4.csv

输出：
results/04_output/
├─ table/
└─ figure/

主要分析：
1. 各方法、各窗口、各信号的 MAE / RMSE / NRMSE
2. Core3 / All4 综合误差
3. Error(k) 与 Gain(k)
4. Trend Accuracy
5. Similarity 方法横向比较
6. 基线模型比较
7. True vs Predicted 散点图

说明：
- 不对 y_true / y_pred 的缺失值做强制填补
- Defecation 在 Day7/Day9 若无真实值，会自然被排除
- 不自动宣布 MSOH；阈值需预先定义
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("results/03_output")
DEFAULT_OUTPUT_ROOT = Path("results/04_output")

CORE_SIGNALS = [
    "Weight",
    "Status_Score",
    "Feeding",
]

ALL_SIGNALS = [
    "Weight",
    "Status_Score",
    "Feeding",
    "Defecation",
]

TARGET_DAYS = [7, 8, 9]

ERROR_TAU = None
GAIN_GAMMA = None

SIMILARITY_PREFIX = "Similarity_"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return p.parse_args()


# ============================================================
# 1. 读取与合并所有 prediction 文件
# ============================================================

def detect_method_from_filename(path: Path) -> str:
    name = path.stem

    prefix = "predictions_loog_"

    if name.startswith(prefix):
        return name[len(prefix):]

    return name


def load_all_predictions(input_dir: Path) -> pd.DataFrame:
    files = sorted(
        input_dir.glob(
            "predictions_loog_*.csv"
        )
    )

    if not files:
        # 兼容旧版
        legacy = input_dir / "predictions_loog.csv"

        if legacy.exists():
            files = [legacy]
        else:
            raise FileNotFoundError(
                f"在 {input_dir} 中没有找到 predictions_loog_*.csv"
            )

    frames = []

    required = {
        "Group",
        "Window",
        "Model",
        "Signal",
        "TargetDay",
        "y_true",
        "y_pred",
    }

    for path in files:
        df = pd.read_csv(path)

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} 缺少字段：{sorted(missing)}"
            )

        source_method = detect_method_from_filename(
            path
        )

        df["SourceMethod"] = source_method
        df["SourceFile"] = path.name

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # 去除同一来源内可能的完全重复
    combined = combined.drop_duplicates()

    return combined


# ============================================================
# 2. 模型类型
# ============================================================

def is_similarity_model(model: str) -> bool:
    return str(model).startswith(
        SIMILARITY_PREFIX
    )


def similarity_label(model: str) -> str:
    model = str(model)

    replacements = {
        "Similarity_EuclideanFeature_K2":
            "Euclidean Feature",
        "Similarity_DTW_Core3_K2":
            "DTW Core3",
        "Similarity_DTW_All4_K2":
            "DTW All4",
    }

    return replacements.get(
        model,
        model,
    )


# ============================================================
# 3. 基础指标
# ============================================================

def calculate_model_metrics(
    pred: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "SourceMethod",
        "Window",
        "Model",
        "Signal",
    ]

    for keys, part in pred.groupby(
        group_cols,
        sort=True,
    ):

        (
            source_method,
            window,
            model,
            signal,
        ) = keys

        valid = part[
            part["y_true"].notna()
            & part["y_pred"].notna()
        ].copy()

        n = len(valid)

        if n == 0:
            mae = np.nan
            rmse = np.nan
            true_sd = np.nan
            nrmse_sd = np.nan

        else:
            err = (
                valid["y_pred"]
                - valid["y_true"]
            ).to_numpy(
                dtype=float
            )

            mae = float(
                np.mean(
                    np.abs(err)
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            )

            true_sd = float(
                np.std(
                    valid["y_true"].to_numpy(
                        dtype=float
                    ),
                    ddof=1,
                )
            ) if n > 1 else np.nan

            nrmse_sd = (
                rmse / true_sd
                if (
                    not pd.isna(true_sd)
                    and not np.isclose(
                        true_sd,
                        0.0,
                    )
                )
                else np.nan
            )

        rows.append(
            {
                "SourceMethod":
                    source_method,
                "Window":
                    int(window),
                "Model":
                    model,
                "Signal":
                    signal,
                "N":
                    int(n),
                "MAE":
                    mae,
                "RMSE":
                    rmse,
                "True_SD":
                    true_sd,
                "NRMSE_SD":
                    nrmse_sd,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_overall_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        source_method,
        window,
        model,
    ), part in metrics.groupby(
        [
            "SourceMethod",
            "Window",
            "Model",
        ],
        sort=True,
    ):

        core = part[
            part["Signal"].isin(
                CORE_SIGNALS
            )
        ]

        all4 = part[
            part["Signal"].isin(
                ALL_SIGNALS
            )
        ]

        rows.append(
            {
                "SourceMethod":
                    source_method,
                "Window":
                    int(window),
                "Model":
                    model,
                "Core3_NRMSE_Mean":
                    float(
                        core["NRMSE_SD"].mean()
                    ),
                "All4_NRMSE_Mean":
                    float(
                        all4["NRMSE_SD"].mean()
                    ),
                "Core3_Signals_N":
                    int(
                        core["NRMSE_SD"]
                        .notna()
                        .sum()
                    ),
                "All4_Signals_N":
                    int(
                        all4["NRMSE_SD"]
                        .notna()
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 4. 组级误差
# ============================================================

def build_group_level_errors(
    pred: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "SourceMethod",
        "Group",
        "Window",
        "Model",
    ]

    for keys, part in pred.groupby(
        group_cols,
        sort=True,
    ):

        (
            source_method,
            group,
            window,
            model,
        ) = keys

        for scope_name, signals in [
            ("Core3", CORE_SIGNALS),
            ("All4", ALL_SIGNALS),
        ]:

            sub = part[
                part["Signal"].isin(
                    signals
                )
                & part["y_true"].notna()
                & part["y_pred"].notna()
            ].copy()

            if sub.empty:
                mae = np.nan
                rmse = np.nan
                n = 0
            else:
                err = (
                    sub["y_pred"]
                    - sub["y_true"]
                ).to_numpy(
                    dtype=float
                )

                mae = float(
                    np.mean(
                        np.abs(err)
                    )
                )

                rmse = float(
                    np.sqrt(
                        np.mean(
                            err ** 2
                        )
                    )
                )

                n = len(sub)

            rows.append(
                {
                    "SourceMethod":
                        source_method,
                    "Group":
                        group,
                    "Window":
                        int(window),
                    "Model":
                        model,
                    "Scope":
                        scope_name,
                    "N":
                        int(n),
                    "MAE":
                        mae,
                    "RMSE":
                        rmse,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. Error(k) / Gain(k)
# ============================================================

def build_window_error_gain(
    overall: pd.DataFrame,
) -> pd.DataFrame:

    sim = overall[
        overall["Model"].map(
            is_similarity_model
        )
    ].copy()

    rows = []

    for (
        source_method,
        model,
    ), part in sim.groupby(
        [
            "SourceMethod",
            "Model",
        ],
        sort=True,
    ):

        part = part.sort_values(
            "Window"
        )

        prev_core = None
        prev_all4 = None

        for _, row in part.iterrows():

            core = row[
                "Core3_NRMSE_Mean"
            ]

            all4 = row[
                "All4_NRMSE_Mean"
            ]

            gain_core = (
                prev_core - core
                if (
                    prev_core is not None
                    and not pd.isna(
                        prev_core
                    )
                    and not pd.isna(
                        core
                    )
                )
                else np.nan
            )

            gain_all4 = (
                prev_all4 - all4
                if (
                    prev_all4 is not None
                    and not pd.isna(
                        prev_all4
                    )
                    and not pd.isna(
                        all4
                    )
                )
                else np.nan
            )

            rel_core = (
                gain_core / prev_core
                if (
                    prev_core is not None
                    and not pd.isna(
                        gain_core
                    )
                    and not np.isclose(
                        prev_core,
                        0.0,
                    )
                )
                else np.nan
            )

            rel_all4 = (
                gain_all4 / prev_all4
                if (
                    prev_all4 is not None
                    and not pd.isna(
                        gain_all4
                    )
                    and not np.isclose(
                        prev_all4,
                        0.0,
                    )
                )
                else np.nan
            )

            rows.append(
                {
                    "SourceMethod":
                        source_method,
                    "Model":
                        model,
                    "SimilarityLabel":
                        similarity_label(
                            model
                        ),
                    "Window":
                        int(
                            row["Window"]
                        ),
                    "Error_Core3":
                        core,
                    "Gain_Core3":
                        gain_core,
                    "RelativeGain_Core3":
                        rel_core,
                    "Error_All4":
                        all4,
                    "Gain_All4":
                        gain_all4,
                    "RelativeGain_All4":
                        rel_all4,
                }
            )

            prev_core = core
            prev_all4 = all4

    return pd.DataFrame(
        rows
    )


# ============================================================
# 6. 趋势准确率
# ============================================================

def sign_with_zero(
    x: float,
    tol: float = 1e-12,
) -> int:

    if pd.isna(x):
        return 99

    if x > tol:
        return 1

    if x < -tol:
        return -1

    return 0


def build_trend_by_group(
    pred: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "SourceMethod",
        "Group",
        "Window",
        "Model",
        "Signal",
    ]

    for keys, part in pred.groupby(
        group_cols,
        sort=True,
    ):

        (
            source_method,
            group,
            window,
            model,
            signal,
        ) = keys

        valid = part[
            part["y_true"].notna()
            & part["y_pred"].notna()
        ].sort_values(
            "TargetDay"
        )

        if len(valid) < 2:

            true_slope = np.nan
            pred_slope = np.nan
            correct = np.nan

        else:

            x = valid[
                "TargetDay"
            ].to_numpy(
                dtype=float
            )

            y_true = valid[
                "y_true"
            ].to_numpy(
                dtype=float
            )

            y_pred = valid[
                "y_pred"
            ].to_numpy(
                dtype=float
            )

            true_slope = float(
                np.polyfit(
                    x,
                    y_true,
                    1,
                )[0]
            )

            pred_slope = float(
                np.polyfit(
                    x,
                    y_pred,
                    1,
                )[0]
            )

            correct = int(
                sign_with_zero(
                    true_slope
                )
                ==
                sign_with_zero(
                    pred_slope
                )
            )

        rows.append(
            {
                "SourceMethod":
                    source_method,
                "Group":
                    group,
                "Window":
                    int(window),
                "Model":
                    model,
                "Signal":
                    signal,
                "TrueSlope":
                    true_slope,
                "PredSlope":
                    pred_slope,
                "TrendCorrect":
                    correct,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_trend_accuracy(
    trend_by_group: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        source_method,
        window,
        model,
        signal,
    ), part in trend_by_group.groupby(
        [
            "SourceMethod",
            "Window",
            "Model",
            "Signal",
        ],
        sort=True,
    ):

        valid = part[
            part["TrendCorrect"].notna()
        ]

        acc = (
            float(
                valid[
                    "TrendCorrect"
                ].mean()
            )
            if len(valid)
            else np.nan
        )

        rows.append(
            {
                "SourceMethod":
                    source_method,
                "Window":
                    int(window),
                "Model":
                    model,
                "Signal":
                    signal,
                "N":
                    int(len(valid)),
                "TrendAccuracy":
                    acc,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_core3_trend_summary(
    trend_accuracy: pd.DataFrame,
) -> pd.DataFrame:

    core = trend_accuracy[
        trend_accuracy["Signal"].isin(
            CORE_SIGNALS
        )
    ]

    return (
        core.groupby(
            [
                "SourceMethod",
                "Window",
                "Model",
            ],
            as_index=False,
        )
        .agg(
            Core3_TrendAccuracy=(
                "TrendAccuracy",
                "mean",
            ),
            Core3_TrendSignalsN=(
                "TrendAccuracy",
                lambda x: int(
                    x.notna().sum()
                ),
            ),
        )
    )


# ============================================================
# 7. 相似度方法比较
# ============================================================

def build_similarity_method_comparison(
    overall: pd.DataFrame,
    core3_trend: pd.DataFrame,
) -> pd.DataFrame:

    sim_overall = overall[
        overall["Model"].map(
            is_similarity_model
        )
    ].copy()

    sim_trend = core3_trend[
        core3_trend["Model"].map(
            is_similarity_model
        )
    ].copy()

    merged = sim_overall.merge(
        sim_trend[
            [
                "SourceMethod",
                "Window",
                "Model",
                "Core3_TrendAccuracy",
            ]
        ],
        on=[
            "SourceMethod",
            "Window",
            "Model",
        ],
        how="left",
    )

    merged["SimilarityLabel"] = (
        merged["Model"].map(
            similarity_label
        )
    )

    return merged.sort_values(
        [
            "Window",
            "Core3_NRMSE_Mean",
        ]
    )


# ============================================================
# 8. MSOH 分析表（不自动宣布）
# ============================================================

def build_msoh_analysis(
    window_gain: pd.DataFrame,
) -> pd.DataFrame:

    df = window_gain.copy()

    if ERROR_TAU is not None:
        df[
            "ErrorCriterion"
        ] = (
            df["Error_Core3"]
            <= ERROR_TAU
        )
    else:
        df[
            "ErrorCriterion"
        ] = np.nan

    if GAIN_GAMMA is not None:
        df[
            "GainCriterion"
        ] = (
            df["Gain_Core3"]
            <= GAIN_GAMMA
        )
    else:
        df[
            "GainCriterion"
        ] = np.nan

    if (
        ERROR_TAU is not None
        and GAIN_GAMMA is not None
    ):
        df[
            "MSOH_Candidate"
        ] = (
            df["ErrorCriterion"]
            & df["GainCriterion"]
        )
    else:
        df[
            "MSOH_Candidate"
        ] = np.nan

    return df


# ============================================================
# 9. 作图
# ============================================================

def savefig(
    fig,
    figure_dir: Path,
    stem: str,
):
    fig.tight_layout()

    fig.savefig(
        figure_dir / f"{stem}.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig.savefig(
        figure_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_similarity_error(
    comparison: pd.DataFrame,
    figure_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for model, part in comparison.groupby(
        "Model"
    ):
        part = part.sort_values(
            "Window"
        )

        ax.plot(
            part["Window"],
            part["Core3_NRMSE_Mean"],
            marker="o",
            label=similarity_label(
                model
            ),
        )

    ax.set_xlabel(
        "Observation Window End Day"
    )

    ax.set_ylabel(
        "Core3 Mean NRMSE (SD-normalized)"
    )

    ax.set_title(
        "Similarity Method Comparison: Error(k)"
    )

    ax.legend()

    savefig(
        fig,
        figure_dir,
        "fig01_similarity_method_error",
    )


def plot_similarity_gain(
    gain: pd.DataFrame,
    figure_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for model, part in gain.groupby(
        "Model"
    ):
        part = part.sort_values(
            "Window"
        )

        ax.plot(
            part["Window"],
            part["Gain_Core3"],
            marker="o",
            label=similarity_label(
                model
            ),
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Observation Window End Day"
    )

    ax.set_ylabel(
        "Gain(k) = Error(k-1) - Error(k)"
    )

    ax.set_title(
        "Marginal Predictive Gain by Similarity Method"
    )

    ax.legend()

    savefig(
        fig,
        figure_dir,
        "fig02_similarity_method_gain",
    )


def plot_similarity_trend(
    comparison: pd.DataFrame,
    figure_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for model, part in comparison.groupby(
        "Model"
    ):
        part = part.sort_values(
            "Window"
        )

        ax.plot(
            part["Window"],
            part["Core3_TrendAccuracy"],
            marker="o",
            label=similarity_label(
                model
            ),
        )

    ax.set_xlabel(
        "Observation Window End Day"
    )

    ax.set_ylabel(
        "Core3 Mean Trend Accuracy"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Trend Accuracy by Similarity Method"
    )

    ax.legend()

    savefig(
        fig,
        figure_dir,
        "fig03_similarity_method_trend",
    )


def plot_all_models(
    overall: pd.DataFrame,
    figure_dir: Path,
):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    # 避免重复基线：取每个 Window/Model 第一条
    temp = (
        overall.sort_values(
            "SourceMethod"
        )
        .drop_duplicates(
            subset=[
                "Window",
                "Model",
            ],
            keep="first",
        )
    )

    for model, part in temp.groupby(
        "Model"
    ):
        part = part.sort_values(
            "Window"
        )

        ax.plot(
            part["Window"],
            part["Core3_NRMSE_Mean"],
            marker="o",
            label=similarity_label(
                model
            ),
        )

    ax.set_xlabel(
        "Observation Window End Day"
    )

    ax.set_ylabel(
        "Core3 Mean NRMSE"
    )

    ax.set_title(
        "All Model Comparison"
    )

    ax.legend(
        fontsize=8,
    )

    savefig(
        fig,
        figure_dir,
        "fig04_all_model_comparison",
    )


def plot_true_vs_pred(
    pred: pd.DataFrame,
    figure_dir: Path,
    window: int = 4,
):
    sim = pred[
        pred["Model"].map(
            is_similarity_model
        )
        & (pred["Window"] == window)
    ]

    for signal in ALL_SIGNALS:

        part = sim[
            (sim["Signal"] == signal)
            & sim["y_true"].notna()
            & sim["y_pred"].notna()
        ]

        if part.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        for model, mpart in part.groupby(
            "Model"
        ):
            ax.scatter(
                mpart["y_true"],
                mpart["y_pred"],
                label=similarity_label(
                    model
                ),
                alpha=0.75,
            )

        low = min(
            part["y_true"].min(),
            part["y_pred"].min(),
        )

        high = max(
            part["y_true"].max(),
            part["y_pred"].max(),
        )

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
        )

        ax.set_xlabel(
            "Observed"
        )

        ax.set_ylabel(
            "Predicted"
        )

        ax.set_title(
            f"{signal}: True vs Predicted, Day1-{window}"
        )

        ax.legend(
            fontsize=8,
        )

        stem_signal = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            signal.lower(),
        )

        savefig(
            fig,
            figure_dir,
            f"fig05_true_vs_pred_day{window}_{stem_signal}",
        )


# ============================================================
# 10. 保存表
# ============================================================

def save_tables(
    table_dir: Path,
    **tables,
):
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in tables.items():
        df.to_csv(
            table_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# 11. 控制台摘要
# ============================================================

def print_summary(
    comparison: pd.DataFrame,
    gain: pd.DataFrame,
):
    print("=" * 84)
    print("统一 04 分析完成")
    print("=" * 84)

    models = (
        comparison[
            "Model"
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        "Similarity 方法：",
        [
            similarity_label(
                x
            )
            for x in models
        ],
    )

    print("\nCore3 NRMSE：")

    display_cols = [
        "Window",
        "SimilarityLabel",
        "Core3_NRMSE_Mean",
        "Core3_TrendAccuracy",
    ]

    print(
        comparison[
            display_cols
        ]
        .sort_values(
            [
                "Window",
                "Core3_NRMSE_Mean",
            ]
        )
        .to_string(
            index=False
        )
    )

    print("\nError / Gain：")

    print(
        gain[
            [
                "Window",
                "SimilarityLabel",
                "Error_Core3",
                "Gain_Core3",
                "RelativeGain_Core3",
            ]
        ]
        .sort_values(
            [
                "SimilarityLabel",
                "Window",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n注意：当前不根据结果自动选择 MSOH；"
        "ERROR_TAU / GAIN_GAMMA 仍为空。"
    )

    print("=" * 84)


# ============================================================
# 12. main
# ============================================================

def main():
    args = parse_args()

    table_dir = (
        args.output_root
        / "table"
    )

    figure_dir = (
        args.output_root
        / "figure"
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred = load_all_predictions(
        args.input_dir
    )

    metrics = calculate_model_metrics(
        pred
    )

    overall = build_overall_metrics(
        metrics
    )

    group_errors = build_group_level_errors(
        pred
    )

    gain = build_window_error_gain(
        overall
    )

    trend_by_group = build_trend_by_group(
        pred
    )

    trend_accuracy = build_trend_accuracy(
        trend_by_group
    )

    core3_trend = build_core3_trend_summary(
        trend_accuracy
    )

    comparison = build_similarity_method_comparison(
        overall,
        core3_trend,
    )

    msoh = build_msoh_analysis(
        gain
    )

    save_tables(
        table_dir,
        combined_predictions=pred,
        model_metrics=metrics,
        overall_model_metrics=overall,
        group_level_errors=group_errors,
        window_error_gain=gain,
        trend_by_group=trend_by_group,
        trend_accuracy=trend_accuracy,
        core3_trend_summary=core3_trend,
        similarity_method_comparison=comparison,
        msoh_analysis=msoh,
    )

    plot_similarity_error(
        comparison,
        figure_dir,
    )

    plot_similarity_gain(
        gain,
        figure_dir,
    )

    plot_similarity_trend(
        comparison,
        figure_dir,
    )

    plot_all_models(
        overall,
        figure_dir,
    )

    plot_true_vs_pred(
        pred,
        figure_dir,
        window=4,
    )

    print_summary(
        comparison,
        gain,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] 04 分析失败：{exc}",
            file=sys.stderr,
        )
        raise
