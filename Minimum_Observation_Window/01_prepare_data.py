"""
01_prepare_data.py

动物实验 MSOH 项目 - 第 1 步：数据读取、清洗与审计

针对当前真实数据结构：
- 12 个实验组
- Day 1-9
- 108 行
- 指标：
    Weight
    Status_Score
    Feeding
    Defecation
- 当前 Excel 中 Status_Score 列名末尾存在空格，脚本会自动清理。
- Defecation 存在计划性/实际缺失，脚本只做缺失标记，不进行插值。

核心原则
--------
1. 当前统计单位是实验组（Group），不是单只动物。
2. 不把缺失值填 0。
3. 不在这里做跨时间插值，避免后续观察窗口发生未来信息泄漏。
4. 不做全数据标准化。标准化必须在 LOOG 每个训练 fold 内执行。
5. 保留原始数值，仅增加 *_observed 缺失标记列。

运行示例
--------
python 01_prepare_data.py \
    --input "data/animal_data.xlsx" \
    --output-dir "results"

输出
----
results/
├── animal_data_clean.csv
├── data_audit_summary.csv
├── group_day_audit.csv
└── missingness_by_group.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


# ============================================================
# 1. 项目配置
# ============================================================

REQUIRED_COLUMNS = [
    "Group",
    "Day",
    "Weight",
    "Status_Score",
    "Feeding",
    "Defecation",
]

SIGNAL_COLUMNS = [
    "Weight",
    "Status_Score",
    "Feeding",
    "Defecation",
]

EXPECTED_DAYS = list(range(1, 10))


# ============================================================
# 2. 参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="动物实验纵向数据：读取、清洗与数据审计"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/animal_data.xlsx"),
        help="",
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel 工作表名称或序号，默认第 1 个工作表",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/01_output"),
        help="输出目录，默认 results/",
    )

    return parser.parse_args()


# ============================================================
# 3. 数据读取
# ============================================================

def load_data(path: Path, sheet=0) -> pd.DataFrame:
    """读取 Excel 或 CSV。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"暂不支持文件类型：{suffix}。请使用 .xlsx、.xls 或 .csv"
    )


# ============================================================
# 4. 字段名清洗
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理字段名。

    当前真实文件中：
        'Status_Score '
    末尾存在空格。

    统一执行 strip()，避免后续出现隐蔽 KeyError。
    """
    df = df.copy()

    cleaned = [str(col).strip() for col in df.columns]

    if len(cleaned) != len(set(cleaned)):
        raise ValueError(
            "字段名清理空格后出现重复列名，请检查原始文件。"
        )

    df.columns = cleaned
    return df


# ============================================================
# 5. 基础结构验证
# ============================================================

def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "缺少必需字段："
            + ", ".join(missing)
            + f"\n当前字段：{list(df.columns)}"
        )


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一关键字段类型。

    Group:
        字符串，并清除首尾空格

    Day:
        必须可转换为整数

    各指标:
        转换为数值；非法文本变 NaN，之后在审计报告中体现
    """
    df = df.copy()

    if df["Group"].isna().any():
        raise ValueError("Group 存在缺失值，无法确定实验组。")

    df["Group"] = df["Group"].astype(str).str.strip()

    if (df["Group"] == "").any():
        raise ValueError("Group 存在空字符串。")

    day_numeric = pd.to_numeric(df["Day"], errors="coerce")

    if day_numeric.isna().any():
        bad_rows = df.loc[day_numeric.isna(), ["Group", "Day"]]
        raise ValueError(
            "Day 中存在无法转换为数字的值：\n"
            + bad_rows.to_string(index=False)
        )

    if not ((day_numeric % 1) == 0).all():
        raise ValueError("Day 必须是整数时间点。")

    df["Day"] = day_numeric.astype(int)

    for col in SIGNAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 6. 数据完整性检查
# ============================================================

def validate_unique_group_day(df: pd.DataFrame) -> None:
    """每个 Group-Day 只能有一条记录。"""

    duplicated = df.duplicated(
        subset=["Group", "Day"],
        keep=False,
    )

    if duplicated.any():
        bad = (
            df.loc[duplicated, ["Group", "Day"]]
            .sort_values(["Group", "Day"])
        )

        raise ValueError(
            "发现重复 Group-Day 记录：\n"
            + bad.to_string(index=False)
        )


def validate_day_range(df: pd.DataFrame) -> None:
    """
    当前项目设计预期 Day1-Day9。

    如果出现范围外时间点，直接报错，
    防止意外把其他实验阶段混进来。
    """
    invalid = ~df["Day"].isin(EXPECTED_DAYS)

    if invalid.any():
        bad_days = sorted(df.loc[invalid, "Day"].unique())
        raise ValueError(
            f"发现预期范围 Day1-Day9 之外的时间点：{bad_days}"
        )


# ============================================================
# 7. 缺失标记
# ============================================================

def add_observed_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个纵向指标增加观测标记。

    例如：
        Defecation_observed = 1  -> 当日真实观测
        Defecation_observed = 0  -> 当日缺失

    注意：
    此处绝不插值，也不填 0。
    """
    df = df.copy()

    for col in SIGNAL_COLUMNS:
        df[f"{col}_observed"] = (
            df[col].notna().astype("int8")
        )

    return df


# ============================================================
# 8. 审计表
# ============================================================

def build_group_day_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    每组检查：
    - 实际记录数
    - 最小/最大 Day
    - 缺失了哪些 Day
    - 是否完整拥有 Day1-Day9
    """
    rows = []

    expected = set(EXPECTED_DAYS)

    for group, part in df.groupby("Group", sort=True):
        days = set(part["Day"].tolist())
        missing_days = sorted(expected - days)
        extra_days = sorted(days - expected)

        rows.append(
            {
                "Group": group,
                "n_rows": len(part),
                "day_min": part["Day"].min(),
                "day_max": part["Day"].max(),
                "missing_days": (
                    ",".join(map(str, missing_days))
                    if missing_days
                    else ""
                ),
                "extra_days": (
                    ",".join(map(str, extra_days))
                    if extra_days
                    else ""
                ),
                "complete_Day1_Day9": (
                    len(missing_days) == 0
                    and len(extra_days) == 0
                    and len(part) == len(EXPECTED_DAYS)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_missingness_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    每组每个指标的缺失数量和观测数量。
    """
    rows = []

    for group, part in df.groupby("Group", sort=True):
        row = {
            "Group": group,
            "n_rows": len(part),
        }

        for col in SIGNAL_COLUMNS:
            n_missing = int(part[col].isna().sum())
            n_observed = int(part[col].notna().sum())

            row[f"{col}_observed_n"] = n_observed
            row[f"{col}_missing_n"] = n_missing
            row[f"{col}_missing_rate"] = (
                n_missing / len(part)
                if len(part) > 0
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    整体审计摘要。
    """
    records = [
        ("n_rows", len(df)),
        ("n_groups", df["Group"].nunique()),
        ("day_min", df["Day"].min()),
        ("day_max", df["Day"].max()),
        (
            "duplicate_group_day_n",
            int(
                df.duplicated(
                    subset=["Group", "Day"]
                ).sum()
            ),
        ),
    ]

    for col in SIGNAL_COLUMNS:
        records.append(
            (f"{col}_missing_n", int(df[col].isna().sum()))
        )

        records.append(
            (
                f"{col}_missing_rate",
                float(df[col].isna().mean()),
            )
        )

    return pd.DataFrame(
        records,
        columns=["metric", "value"],
    )


# ============================================================
# 9. 主清洗函数
# ============================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    按固定顺序完成数据清洗。

    顺序很重要：
    1. 清理列名
    2. 检查必需字段
    3. 统一类型
    4. 检查 Group-Day 唯一性
    5. 检查时间范围
    6. 排序
    7. 增加 observed 标记
    """
    df = clean_column_names(df)

    validate_required_columns(df)

    # 只保留项目需要的字段。
    # 避免 Excel 中无关辅助列意外进入模型。
    df = df[REQUIRED_COLUMNS].copy()

    df = normalize_types(df)

    validate_unique_group_day(df)
    validate_day_range(df)

    df = (
        df.sort_values(["Group", "Day"])
        .reset_index(drop=True)
    )

    df = add_observed_flags(df)

    return df


# ============================================================
# 10. 输出
# ============================================================

def save_outputs(
    clean_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "animal_data_clean.csv"
    summary_path = output_dir / "data_audit_summary.csv"
    group_day_path = output_dir / "group_day_audit.csv"
    missing_path = output_dir / "missingness_by_group.csv"

    clean_df.to_csv(
        clean_path,
        index=False,
        encoding="utf-8-sig",
    )

    build_audit_summary(clean_df).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    build_group_day_audit(clean_df).to_csv(
        group_day_path,
        index=False,
        encoding="utf-8-sig",
    )

    build_missingness_by_group(clean_df).to_csv(
        missing_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "clean": clean_path,
        "summary": summary_path,
        "group_day": group_day_path,
        "missingness": missing_path,
    }


# ============================================================
# 11. 控制台报告
# ============================================================

def print_console_report(df: pd.DataFrame) -> None:
    group_audit = build_group_day_audit(df)

    print("=" * 60)
    print("动物实验纵向数据审计完成")
    print("=" * 60)

    print(f"记录总数       : {len(df)}")
    print(f"实验组数量     : {df['Group'].nunique()}")
    print(
        f"时间范围       : "
        f"Day{df['Day'].min()} - Day{df['Day'].max()}"
    )

    print(
        "各组 Day1-Day9 完整: "
        f"{bool(group_audit['complete_Day1_Day9'].all())}"
    )

    print("\n缺失值：")

    for col in SIGNAL_COLUMNS:
        n_missing = int(df[col].isna().sum())
        rate = df[col].isna().mean()

        print(
            f"  {col:<16} "
            f"{n_missing:>3} "
            f"({rate:.1%})"
        )

    print("\n重要说明：")
    print("- 缺失值未填 0。")
    print("- 未执行任何跨时间插值。")
    print("- 未执行全局标准化。")
    print(
        "- 缺失处理和标准化将在后续 "
        "Leave-One-Group-Out 训练 fold 内完成。"
    )
    print("=" * 60)


# ============================================================
# 12. main
# ============================================================

def main() -> None:
    args = parse_args()

    raw_df = load_data(
        path=args.input,
        sheet=args.sheet,
    )

    clean_df = prepare_data(raw_df)

    paths = save_outputs(
        clean_df=clean_df,
        output_dir=args.output_dir,
    )

    print_console_report(clean_df)

    print("\n输出文件：")
    for name, path in paths.items():
        print(f"  {name:<12}: {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] 数据准备失败：{exc}",
            file=sys.stderr,
        )
        raise
