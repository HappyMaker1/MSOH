from pathlib import Path

import pandas as pd

from config import ALL_SIGNALS, EXPECTED_DAYS


# ============================================================
# Defecation measurement schedule
# ============================================================

# 当前实验设计：Defecation 只在 Day2 / Day4 / Day6 / Day9 计划测量。
# 非计划测量日即使原始 Excel 中出现 0，也不应被解释为真实的“0 排便值”。
DEFECATION_MEASUREMENT_DAYS = [2, 4, 6, 9]


# ============================================================
# 1. 数据读取
# ============================================================

def load_raw(path: Path, sheet=0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("仅支持 .xlsx/.xls/.csv")


# ============================================================
# 2. 数据清洗
# ============================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------------------------
    # 2.1 字段名清理
    # --------------------------------------------------------
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Group", "Day", *ALL_SIGNALS]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"缺少必需字段：{missing}"
        )

    if len(df.columns) != len(set(df.columns)):
        raise ValueError(
            "字段名清理后出现重复列"
        )

    # --------------------------------------------------------
    # 2.2 Group
    # --------------------------------------------------------
    if df["Group"].isna().any():
        raise ValueError(
            "Group 存在缺失"
        )

    df["Group"] = (
        df["Group"]
        .astype(str)
        .str.strip()
    )

    if (df["Group"] == "").any():
        raise ValueError(
            "Group 存在空字符串"
        )

    # --------------------------------------------------------
    # 2.3 Day
    # --------------------------------------------------------
    day = pd.to_numeric(
        df["Day"],
        errors="coerce",
    )

    if (
        day.isna().any()
        or not ((day % 1) == 0).all()
    ):
        raise ValueError(
            "Day 必须是整数"
        )

    df["Day"] = day.astype(int)

    # --------------------------------------------------------
    # 2.4 Signal numeric conversion
    # --------------------------------------------------------
    for s in ALL_SIGNALS:
        df[s] = pd.to_numeric(
            df[s],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 2.5 Group-Day 唯一性
    # --------------------------------------------------------
    duplicated = df.duplicated(
        ["Group", "Day"],
        keep=False,
    )

    if duplicated.any():
        bad_rows = (
            df.loc[
                duplicated,
                ["Group", "Day"],
            ]
            .sort_values(
                ["Group", "Day"]
            )
        )

        raise ValueError(
            "发现重复 Group-Day：\n"
            + bad_rows.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 2.6 Day 范围
    # --------------------------------------------------------
    bad = sorted(
        set(df["Day"])
        - set(EXPECTED_DAYS)
    )

    if bad:
        raise ValueError(
            f"发现预期范围外 Day：{bad}"
        )

    # --------------------------------------------------------
    # 2.7 排序
    # --------------------------------------------------------
    df = (
        df.sort_values(
            ["Group", "Day"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 2.8 Dense signals observed flags
    # ========================================================

    dense_signals = [
        s
        for s in ALL_SIGNALS
        if s != "Defecation"
    ]

    for s in dense_signals:
        df[f"{s}_observed"] = (
            df[s]
            .notna()
            .astype("int8")
        )

    # ========================================================
    # 2.9 Defecation schedule-aware cleaning
    # ========================================================

    # 1 = 实验设计规定当天应该测
    # 0 = 实验设计规定当天不测
    df["Defecation_scheduled"] = (
        df["Day"]
        .isin(DEFECATION_MEASUREMENT_DAYS)
        .astype("int8")
    )

    # 保留原始 Defecation 数值用于审计。
    df["Defecation_raw"] = df["Defecation"]

    # 标记非计划测量日原始表中存在数值的记录。
    # 当前这类值是 0 占位，不能当作真实“0 排便”。
    df["Defecation_unscheduled_value"] = (
        (df["Defecation_scheduled"] == 0)
        & df["Defecation"].notna()
    ).astype("int8")

    # 非计划测量日属于 structural unobserved。
    # 清洗后统一转为 NaN。
    df.loc[
        df["Defecation_scheduled"] == 0,
        "Defecation",
    ] = pd.NA

    # 只有 scheduled=1 且有实际数值时才算真实观测。
    df["Defecation_observed"] = (
        (df["Defecation_scheduled"] == 1)
        & df["Defecation"].notna()
    ).astype("int8")

    return df


# ============================================================
# 3. Group-Day 审计
# ============================================================

def group_day_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:

    expected = set(
        EXPECTED_DAYS
    )

    rows = []

    for g, p in df.groupby(
        "Group",
        sort=True,
    ):

        days = set(
            p["Day"].tolist()
        )

        rows.append(
            {
                "Group": g,
                "N_Rows": len(p),
                "Min_Day": p["Day"].min(),
                "Max_Day": p["Day"].max(),
                "Missing_Days": ",".join(
                    map(
                        str,
                        sorted(
                            expected - days
                        ),
                    )
                ),
                "Extra_Days": ",".join(
                    map(
                        str,
                        sorted(
                            days - expected
                        ),
                    )
                ),
                "Complete_Day1_9": (
                    expected.issubset(days)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 4. Group-level missingness / observed audit
# ============================================================

def missingness_by_group(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for g, p in df.groupby(
        "Group",
        sort=True,
    ):

        row = {
            "Group": g
        }

        for s in ALL_SIGNALS:

            observed_col = (
                f"{s}_observed"
            )

            if s == "Defecation":

                scheduled_n = int(
                    p[
                        "Defecation_scheduled"
                    ].sum()
                )

                observed_n = int(
                    p[
                        "Defecation_observed"
                    ].sum()
                )

                missing_n = (
                    scheduled_n
                    - observed_n
                )

                unscheduled_value_n = int(
                    p[
                        "Defecation_unscheduled_value"
                    ].sum()
                )

                row[
                    "Defecation_Scheduled_N"
                ] = scheduled_n

                row[
                    "Defecation_Observed_N"
                ] = observed_n

                row[
                    "Defecation_Missing_N"
                ] = missing_n

                row[
                    "Defecation_UnscheduledValue_N"
                ] = unscheduled_value_n

            else:

                observed_n = int(
                    p[
                        observed_col
                    ].sum()
                )

                missing_n = int(
                    (
                        p[
                            observed_col
                        ]
                        == 0
                    ).sum()
                )

                row[
                    f"{s}_Observed_N"
                ] = observed_n

                row[
                    f"{s}_Missing_N"
                ] = missing_n

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. Overall audit summary
# ============================================================

def audit_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = [
        {
            "Item": "Rows",
            "Value": len(df),
        },
        {
            "Item": "Groups",
            "Value": (
                df["Group"]
                .nunique()
            ),
        },
        {
            "Item": "Min_Day",
            "Value": (
                df["Day"]
                .min()
            ),
        },
        {
            "Item": "Max_Day",
            "Value": (
                df["Day"]
                .max()
            ),
        },
    ]

    for s in ALL_SIGNALS:

        observed_col = (
            f"{s}_observed"
        )

        if s == "Defecation":

            scheduled_n = int(
                df[
                    "Defecation_scheduled"
                ].sum()
            )

            observed_n = int(
                df[
                    "Defecation_observed"
                ].sum()
            )

            missing_n = (
                scheduled_n
                - observed_n
            )

            unscheduled_value_n = int(
                df[
                    "Defecation_unscheduled_value"
                ].sum()
            )

            if scheduled_n > 0:
                missing_rate = (
                    missing_n
                    / scheduled_n
                )
            else:
                missing_rate = float(
                    "nan"
                )

            rows += [
                {
                    "Item":
                        "Defecation_Scheduled_N",
                    "Value":
                        scheduled_n,
                },
                {
                    "Item":
                        "Defecation_Observed_N",
                    "Value":
                        observed_n,
                },
                {
                    "Item":
                        "Defecation_Missing_N",
                    "Value":
                        missing_n,
                },
                {
                    "Item":
                        "Defecation_Missing_Rate",
                    "Value":
                        missing_rate,
                },
                {
                    "Item":
                        "Defecation_UnscheduledValue_N",
                    "Value":
                        unscheduled_value_n,
                },
            ]

        else:

            observed_n = int(
                df[
                    observed_col
                ].sum()
            )

            missing_n = int(
                (
                    df[
                        observed_col
                    ]
                    == 0
                ).sum()
            )

            missing_rate = (
                missing_n
                / len(df)
                if len(df) > 0
                else float("nan")
            )

            rows += [
                {
                    "Item":
                        f"{s}_Observed_N",
                    "Value":
                        observed_n,
                },
                {
                    "Item":
                        f"{s}_Missing_N",
                    "Value":
                        missing_n,
                },
                {
                    "Item":
                        f"{s}_Missing_Rate",
                    "Value":
                        missing_rate,
                },
            ]

    return pd.DataFrame(
        rows
    )
