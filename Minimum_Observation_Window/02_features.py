"""
02_features.py

动物实验 MSOH 项目 - 第 2 步：早期观察窗口与轨迹特征构造

输入
----
results/animal_data_clean.csv

输出
----
results/early_features.csv
results/feature_audit.csv

主分析窗口
--------
Day1-Day2
Day1-Day3
Day1-Day4
Day1-Day5
Day1-Day6

主特征
--------
每个指标：
- *_last
- *_slope

Defecation 特殊规则
-------------------
已知 Day1、Day3、Day5 为实验设计上的结构性未测。

因此主分析：
1. 不做时间插值；
2. 仅使用窗口内 observed=1 的真实测量点；
3. 至少 1 个真实点时可计算 last；
4. 至少 2 个真实点时可计算 slope；
5. 少于 2 个真实点时 slope = NaN。

其他指标
--------
Weight / Status_Score / Feeding 当前无缺失，
仍统一只使用 observed=1 的真实观测点计算特征。

重要原则
--------
- 不使用 Day k 之后的数据。
- 不在本文件中做全局缺失填补。
- 不在本文件中做标准化。
- 后续缺失填补与标准化由 03_models_cv.py
  在每个 Leave-One-Group-Out 训练 fold 内完成。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1. 配置
# ============================================================

DEFAULT_INPUT = Path("results/animal_data_clean.csv")
DEFAULT_OUTPUT_DIR = Path("results/02_output")

WINDOWS = [2, 3, 4, 5, 6]

SIGNALS = [
    "Weight",
    "Status_Score",
    "Feeding",
    "Defecation",
]

ID_COLUMNS = ["Group", "Day"]


# ============================================================
# 2. 参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造早期观察窗口并提取纵向轨迹特征"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入清洗后 CSV，默认 {DEFAULT_INPUT}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--use-delta",
        action="store_true",
        help="可选：额外输出 *_delta 特征；第一版主分析默认关闭",
    )

    return parser.parse_args()


# ============================================================
# 3. 读取与验证
# ============================================================

def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)

    required = ["Group", "Day"]

    for signal in SIGNALS:
        required.append(signal)
        required.append(f"{signal}_observed")

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "输入文件缺少必要字段："
            + ", ".join(missing)
        )

    return df


def validate_clean_data(df: pd.DataFrame) -> None:
    """
    检查特征构造前的数据结构。
    """

    duplicated = df.duplicated(["Group", "Day"])

    if duplicated.any():
        raise ValueError(
            "输入数据仍存在重复 Group-Day，不能继续构造特征。"
        )

    expected_windows = set(range(1, 10))

    for group, part in df.groupby("Group"):
        days = set(part["Day"].astype(int).tolist())

        if not expected_windows.issubset(days):
            missing_days = sorted(expected_windows - days)
            raise ValueError(
                f"Group={group} 缺少时间点：{missing_days}"
            )


# ============================================================
# 4. 特征计算工具
# ============================================================

def get_observed_points(
    group_df: pd.DataFrame,
    signal: str,
    k: int,
) -> pd.DataFrame:
    """
    只保留：
    1. Day <= k
    2. *_observed == 1
    3. 指标值非缺失

    这样可以确保 Defecation 不会因为结构性未测
    被错误当成连续观测。
    """
    observed_col = f"{signal}_observed"

    part = group_df[
        (group_df["Day"] <= k)
        & (group_df[observed_col] == 1)
        & (group_df[signal].notna())
    ].copy()

    return part.sort_values("Day")


def calculate_last(
    observed_df: pd.DataFrame,
    signal: str,
) -> float:
    """
    窗口内最后一个真实观测值。
    """
    if observed_df.empty:
        return np.nan

    return float(observed_df.iloc[-1][signal])


def calculate_slope(
    observed_df: pd.DataFrame,
    signal: str,
) -> float:
    """
    使用真实观测点做一阶线性拟合。

    至少需要 2 个真实观测点。
    """
    if len(observed_df) < 2:
        return np.nan

    x = observed_df["Day"].to_numpy(dtype=float)
    y = observed_df[signal].to_numpy(dtype=float)

    slope = np.polyfit(x, y, deg=1)[0]

    return float(slope)


def calculate_delta(
    observed_df: pd.DataFrame,
    signal: str,
) -> float:
    """
    可选敏感性特征：
    最后真实观测 - 第一个真实观测

    至少 2 个真实点。
    """
    if len(observed_df) < 2:
        return np.nan

    return float(
        observed_df.iloc[-1][signal]
        - observed_df.iloc[0][signal]
    )


# ============================================================
# 5. 单组单窗口特征
# ============================================================

def extract_group_window_features(
    group_df: pd.DataFrame,
    k: int,
    use_delta: bool = False,
) -> dict:
    """
    提取某一个实验组在 Day1-Dayk 窗口下的特征。
    """

    group_name = str(group_df["Group"].iloc[0])

    row = {
        "Group": group_name,
        "Window": k,
        "ObservedUntilDay": k,
    }

    for signal in SIGNALS:
        observed = get_observed_points(
            group_df=group_df,
            signal=signal,
            k=k,
        )

        n_observed = len(observed)

        row[f"{signal}_n_observed"] = int(n_observed)
        row[f"{signal}_last"] = calculate_last(
            observed,
            signal,
        )
        row[f"{signal}_slope"] = calculate_slope(
            observed,
            signal,
        )

        if use_delta:
            row[f"{signal}_delta"] = calculate_delta(
                observed,
                signal,
            )

    return row


# ============================================================
# 6. 全部组与窗口
# ============================================================

def build_feature_table(
    df: pd.DataFrame,
    windows: list[int] = WINDOWS,
    use_delta: bool = False,
) -> pd.DataFrame:
    rows = []

    for group, group_df in df.groupby("Group", sort=True):
        group_df = (
            group_df
            .sort_values("Day")
            .reset_index(drop=True)
        )

        for k in windows:
            rows.append(
                extract_group_window_features(
                    group_df=group_df,
                    k=k,
                    use_delta=use_delta,
                )
            )

    features = pd.DataFrame(rows)

    return features.sort_values(
        ["Group", "Window"]
    ).reset_index(drop=True)


# ============================================================
# 7. 特征审计
# ============================================================

def build_feature_audit(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    检查每个窗口中：
    - 各特征缺失数量
    - 各特征缺失比例

    特别关注 Defecation_slope：
    Day1-2 / Day1-3 因真实观测点不足 2 个，
    预期应出现结构性 NaN。
    """

    feature_cols = [
        c for c in features.columns
        if c not in [
            "Group",
            "Window",
            "ObservedUntilDay",
        ]
    ]

    rows = []

    for k, part in features.groupby("Window"):
        for col in feature_cols:
            rows.append(
                {
                    "Window": int(k),
                    "Feature": col,
                    "n_rows": len(part),
                    "n_missing": int(
                        part[col].isna().sum()
                    ),
                    "missing_rate": float(
                        part[col].isna().mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 8. 输出
# ============================================================

def save_outputs(
    features: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        output_dir
        / "early_features.csv"
    )

    audit_path = (
        output_dir
        / "feature_audit.csv"
    )

    features.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit = build_feature_audit(features)

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "features": feature_path,
        "audit": audit_path,
    }


# ============================================================
# 9. 控制台摘要
# ============================================================

def print_console_report(
    features: pd.DataFrame,
) -> None:

    print("=" * 68)
    print("早期纵向轨迹特征构造完成")
    print("=" * 68)

    print(
        f"实验组数量       : "
        f"{features['Group'].nunique()}"
    )

    print(
        f"观察窗口         : "
        f"{sorted(features['Window'].unique())}"
    )

    print(
        f"特征记录总数     : "
        f"{len(features)}"
    )

    print("\nDefecation 各窗口真实观测点：")

    summary = (
        features
        .groupby("Window")[
            "Defecation_n_observed"
        ]
        .agg(["min", "max", "median"])
    )

    print(summary.to_string())

    print("\nDefecation_slope 缺失情况：")

    slope_missing = (
        features
        .groupby("Window")[
            "Defecation_slope"
        ]
        .apply(lambda s: s.isna().sum())
    )

    for k, n_missing in slope_missing.items():
        print(
            f"  Day1-Day{k}: "
            f"{n_missing}/"
            f"{features['Group'].nunique()} "
            f"缺失"
        )

    print("\n原则确认：")
    print("- 未进行任何时间插值。")
    print("- 所有特征只使用 Day<=当前窗口的数据。")
    print("- Defecation 仅使用真实 observed=1 的测量点。")
    print("- 未进行全局缺失填补。")
    print("- 未进行全局标准化。")
    print("=" * 68)


# ============================================================
# 10. main
# ============================================================

def main() -> None:
    args = parse_args()

    df = load_clean_data(
        args.input
    )

    validate_clean_data(df)

    features = build_feature_table(
        df=df,
        windows=WINDOWS,
        use_delta=args.use_delta,
    )

    paths = save_outputs(
        features=features,
        output_dir=args.output_dir,
    )

    print_console_report(
        features
    )

    print("\n输出文件：")

    for name, path in paths.items():
        print(
            f"  {name:<10}: {path}"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"\n[ERROR] 特征构造失败：{exc}",
            file=sys.stderr,
        )
        raise
