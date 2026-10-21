from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path("results/04_output/table/combined_predictions.csv")
DEFAULT_OUTPUT = Path("results/05_output")

CORE_SIGNALS = ["Weight", "Status_Score", "Feeding"]
SIMILARITY_PREFIX = "Similarity_"

BOOTSTRAP_B = 5000
RANDOM_SEED = 20260924
CHANGE_TOL = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--change-tol", type=float, default=CHANGE_TOL)
    return p.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path)

    required = {
        "Group", "Window", "Model", "Signal",
        "TargetDay", "y_true", "y_pred", "SourceMethod"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少字段：{sorted(missing)}")

    df["Group"] = df["Group"].astype(str)
    return df


def is_similarity_model(model: str) -> bool:
    return str(model).startswith(SIMILARITY_PREFIX)


def similarity_label(model: str) -> str:
    mapping = {
        "Similarity_EuclideanFeature_K2": "Euclidean Feature",
        "Similarity_DTW_Core3_K2": "DTW Core3",
        "Similarity_DTW_All4_K2": "DTW All4",
    }
    return mapping.get(str(model), str(model))


def core3_error_from_rows(rows: pd.DataFrame) -> float:
    """
    严格复现 04 的 Core3_NRMSE_Mean：
    对每个 Core3 signal：
        RMSE / SD(y_true, ddof=1)
    再对三个 signal 做等权平均。
    """
    nrmse_values = []

    for signal in CORE_SIGNALS:
        part = rows[
            (rows["Signal"] == signal)
            & rows["y_true"].notna()
            & rows["y_pred"].notna()
        ]

        if len(part) < 2:
            continue

        y_true = part["y_true"].to_numpy(dtype=float)
        y_pred = part["y_pred"].to_numpy(dtype=float)

        err = y_pred - y_true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        true_sd = float(np.std(y_true, ddof=1))

        if np.isclose(true_sd, 0.0):
            continue

        nrmse_values.append(rmse / true_sd)

    if not nrmse_values:
        return np.nan

    return float(np.mean(nrmse_values))


def prepare_similarity_data(pred: pd.DataFrame) -> pd.DataFrame:
    sim = pred[
        pred["Model"].map(is_similarity_model)
        & pred["Signal"].isin(CORE_SIGNALS)
    ].copy()

    if sim.empty:
        raise ValueError("输入中没有找到 Similarity 模型记录。")

    return sim


def method_table(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim[["SourceMethod", "Model"]]
        .drop_duplicates()
        .sort_values(["SourceMethod", "Model"])
        .reset_index(drop=True)
    )


def resample_groups_as_rows(
    part: pd.DataFrame,
    sampled_groups: np.ndarray,
) -> pd.DataFrame:
    blocks = []

    for bootstrap_id, group in enumerate(sampled_groups):
        block = part[part["Group"] == str(group)].copy()

        if block.empty:
            continue

        block["_BootstrapUnit"] = bootstrap_id
        blocks.append(block)

    if not blocks:
        return part.iloc[0:0].copy()

    return pd.concat(blocks, ignore_index=True)


def bootstrap_error_ci(
    sim: pd.DataFrame,
    bootstrap_b: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for _, mrow in method_table(sim).iterrows():
        source_method = mrow["SourceMethod"]
        model = mrow["Model"]

        method_part = sim[
            (sim["SourceMethod"] == source_method)
            & (sim["Model"] == model)
        ].copy()

        groups = sorted(method_part["Group"].unique().tolist())
        n_groups = len(groups)

        for window in sorted(method_part["Window"].unique()):
            window_part = method_part[method_part["Window"] == window]

            point = core3_error_from_rows(window_part)

            boot = np.empty(bootstrap_b, dtype=float)

            for b in range(bootstrap_b):
                sampled_groups = rng.choice(
                    groups,
                    size=n_groups,
                    replace=True,
                )

                boot_rows = resample_groups_as_rows(
                    window_part,
                    sampled_groups,
                )

                boot[b] = core3_error_from_rows(boot_rows)

            boot = boot[~np.isnan(boot)]

            ci_low, ci_high = np.percentile(
                boot,
                [2.5, 97.5],
            )

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "Window": int(window),
                "N_Groups": int(n_groups),
                "Error_Point": float(point),
                "Error_CI_Low": float(ci_low),
                "Error_CI_High": float(ci_high),
                "Bootstrap_B": int(bootstrap_b),
            })

    return pd.DataFrame(rows)


def bootstrap_gain_ci(
    sim: pd.DataFrame,
    bootstrap_b: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for _, mrow in method_table(sim).iterrows():
        source_method = mrow["SourceMethod"]
        model = mrow["Model"]

        method_part = sim[
            (sim["SourceMethod"] == source_method)
            & (sim["Model"] == model)
        ].copy()

        groups = sorted(method_part["Group"].unique().tolist())
        n_groups = len(groups)

        windows = sorted(method_part["Window"].unique())

        for prev_w, curr_w in zip(windows[:-1], windows[1:]):
            prev_part = method_part[method_part["Window"] == prev_w]
            curr_part = method_part[method_part["Window"] == curr_w]

            prev_point = core3_error_from_rows(prev_part)
            curr_point = core3_error_from_rows(curr_part)
            gain_point = prev_point - curr_point

            boot_gain = np.empty(bootstrap_b, dtype=float)

            for b in range(bootstrap_b):
                sampled_groups = rng.choice(
                    groups,
                    size=n_groups,
                    replace=True,
                )

                prev_boot = resample_groups_as_rows(
                    prev_part,
                    sampled_groups,
                )

                curr_boot = resample_groups_as_rows(
                    curr_part,
                    sampled_groups,
                )

                prev_error = core3_error_from_rows(prev_boot)
                curr_error = core3_error_from_rows(curr_boot)

                boot_gain[b] = prev_error - curr_error

            boot_gain = boot_gain[~np.isnan(boot_gain)]

            ci_low, ci_high = np.percentile(
                boot_gain,
                [2.5, 97.5],
            )

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "FromWindow": int(prev_w),
                "ToWindow": int(curr_w),
                "Error_From": float(prev_point),
                "Error_To": float(curr_point),
                "Gain_Point": float(gain_point),
                "Gain_CI_Low": float(ci_low),
                "Gain_CI_High": float(ci_high),
                "P_Gain_GT_0": float(np.mean(boot_gain > 0)),
                "P_Gain_LT_0": float(np.mean(boot_gain < 0)),
                "N_Groups": int(n_groups),
                "Bootstrap_B": int(bootstrap_b),
            })

    return pd.DataFrame(rows)


def paired_method_difference(
    sim: pd.DataFrame,
    bootstrap_b: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    methods = method_table(sim).to_dict("records")
    windows = sorted(sim["Window"].unique())

    for window in windows:
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method_a = methods[i]
                method_b = methods[j]

                part_a = sim[
                    (sim["SourceMethod"] == method_a["SourceMethod"])
                    & (sim["Model"] == method_a["Model"])
                    & (sim["Window"] == window)
                ]

                part_b = sim[
                    (sim["SourceMethod"] == method_b["SourceMethod"])
                    & (sim["Model"] == method_b["Model"])
                    & (sim["Window"] == window)
                ]

                common_groups = sorted(
                    set(part_a["Group"])
                    & set(part_b["Group"])
                )

                if not common_groups:
                    continue

                part_a = part_a[part_a["Group"].isin(common_groups)]
                part_b = part_b[part_b["Group"].isin(common_groups)]

                error_a = core3_error_from_rows(part_a)
                error_b = core3_error_from_rows(part_b)
                point_diff = error_a - error_b

                n_groups = len(common_groups)
                boot_diff = np.empty(bootstrap_b, dtype=float)

                for b in range(bootstrap_b):
                    sampled_groups = rng.choice(
                        common_groups,
                        size=n_groups,
                        replace=True,
                    )

                    boot_a = resample_groups_as_rows(
                        part_a,
                        sampled_groups,
                    )

                    boot_b = resample_groups_as_rows(
                        part_b,
                        sampled_groups,
                    )

                    boot_diff[b] = (
                        core3_error_from_rows(boot_a)
                        - core3_error_from_rows(boot_b)
                    )

                boot_diff = boot_diff[~np.isnan(boot_diff)]

                ci_low, ci_high = np.percentile(
                    boot_diff,
                    [2.5, 97.5],
                )

                rows.append({
                    "Window": int(window),
                    "Method_A": similarity_label(method_a["Model"]),
                    "Method_B": similarity_label(method_b["Model"]),
                    "Error_A": float(error_a),
                    "Error_B": float(error_b),
                    "Difference_A_minus_B": float(point_diff),
                    "CI_Low": float(ci_low),
                    "CI_High": float(ci_high),
                    "P_A_Error_GT_B": float(np.mean(boot_diff > 0)),
                    "P_A_Error_LT_B": float(np.mean(boot_diff < 0)),
                    "N_Groups": int(n_groups),
                    "Bootstrap_B": int(bootstrap_b),
                })

    return pd.DataFrame(rows)


def build_group_core3_error(sim: pd.DataFrame) -> pd.DataFrame:
    """
    仅用于组级改善/恶化描述，不用于主 Error/Gain bootstrap。
    每个 signal 的分母使用该方法/窗口的全样本 true SD。
    """
    scale_rows = []

    for (
        source_method,
        model,
        window,
        signal,
    ), part in sim.groupby(
        ["SourceMethod", "Model", "Window", "Signal"],
        sort=True,
    ):
        values = part["y_true"].dropna().to_numpy(dtype=float)

        sd = (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else np.nan
        )

        scale_rows.append({
            "SourceMethod": source_method,
            "Model": model,
            "Window": int(window),
            "Signal": signal,
            "True_SD": sd,
        })

    scales = pd.DataFrame(scale_rows)

    temp = sim.merge(
        scales,
        on=["SourceMethod", "Model", "Window", "Signal"],
        how="left",
    )

    rows = []

    for (
        source_method,
        model,
        group,
        window,
    ), part in temp.groupby(
        ["SourceMethod", "Model", "Group", "Window"],
        sort=True,
    ):
        signal_errors = []

        for signal in CORE_SIGNALS:
            sub = part[
                (part["Signal"] == signal)
                & part["y_true"].notna()
                & part["y_pred"].notna()
            ]

            if sub.empty:
                continue

            true_sd = sub["True_SD"].iloc[0]

            if pd.isna(true_sd) or np.isclose(true_sd, 0.0):
                continue

            err = (
                sub["y_pred"].to_numpy(dtype=float)
                - sub["y_true"].to_numpy(dtype=float)
            )

            rmse = float(np.sqrt(np.mean(err ** 2)))
            signal_errors.append(rmse / true_sd)

        rows.append({
            "SourceMethod": source_method,
            "Model": model,
            "SimilarityLabel": similarity_label(model),
            "Group": str(group),
            "Window": int(window),
            "GroupCore3Error": (
                float(np.mean(signal_errors))
                if signal_errors
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def group_window_changes(
    group_errors: pd.DataFrame,
    tol: float,
) -> pd.DataFrame:
    rows = []

    for (
        source_method,
        model,
        group,
    ), part in group_errors.groupby(
        ["SourceMethod", "Model", "Group"],
        sort=True,
    ):
        part = part.sort_values("Window")

        windows = part["Window"].tolist()
        errors = part["GroupCore3Error"].tolist()

        for prev_w, curr_w, prev_e, curr_e in zip(
            windows[:-1],
            windows[1:],
            errors[:-1],
            errors[1:],
        ):
            if pd.isna(prev_e) or pd.isna(curr_e):
                gain = np.nan
                category = "missing"
            else:
                gain = float(prev_e - curr_e)

                if gain > tol:
                    category = "improved"
                elif gain < -tol:
                    category = "worsened"
                else:
                    category = "stable"

            rows.append({
                "SourceMethod": source_method,
                "Model": model,
                "SimilarityLabel": similarity_label(model),
                "Group": group,
                "FromWindow": int(prev_w),
                "ToWindow": int(curr_w),
                "Gain_Group": gain,
                "Category": category,
                "Tolerance": float(tol),
            })

    return pd.DataFrame(rows)


def build_robustness_summary(
    changes: pd.DataFrame,
) -> pd.DataFrame:
    valid = changes[
        changes["Category"].isin(
            ["improved", "stable", "worsened"]
        )
    ].copy()

    summary = (
        valid.groupby(
            [
                "SourceMethod",
                "Model",
                "SimilarityLabel",
                "FromWindow",
                "ToWindow",
                "Category",
            ]
        )
        .size()
        .rename("N")
        .reset_index()
    )

    totals = (
        summary.groupby(
            [
                "SourceMethod",
                "Model",
                "SimilarityLabel",
                "FromWindow",
                "ToWindow",
            ]
        )["N"]
        .transform("sum")
    )

    summary["Proportion"] = summary["N"] / totals

    return summary


def savefig(fig, figure_dir: Path, stem: str):
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


def plot_error_ci(error_ci: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, part in error_ci.groupby("SimilarityLabel"):
        part = part.sort_values("Window")

        y = part["Error_Point"].to_numpy(dtype=float)

        low = y - part["Error_CI_Low"].to_numpy(dtype=float)
        high = part["Error_CI_High"].to_numpy(dtype=float) - y

        ax.errorbar(
            part["Window"],
            y,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=3,
            label=label,
        )

    ax.set_xlabel("Observation Window End Day")
    ax.set_ylabel("Core3 Mean NRMSE")
    ax.set_title("Bootstrap 95% CI for Error(k)")
    ax.legend()

    savefig(fig, figure_dir, "fig01_error_ci")


def plot_gain_ci(gain_ci: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, part in gain_ci.groupby("SimilarityLabel"):
        part = part.sort_values("ToWindow")

        y = part["Gain_Point"].to_numpy(dtype=float)

        low = y - part["Gain_CI_Low"].to_numpy(dtype=float)
        high = part["Gain_CI_High"].to_numpy(dtype=float) - y

        ax.errorbar(
            part["ToWindow"],
            y,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=3,
            label=label,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("To Window End Day")
    ax.set_ylabel("Gain(k) = Error(k-1) - Error(k)")
    ax.set_title("Bootstrap 95% CI for Marginal Gain")
    ax.legend()

    savefig(fig, figure_dir, "fig02_gain_ci")


def plot_method_difference(
    paired: pd.DataFrame,
    figure_dir: Path,
):
    if paired.empty:
        return

    pairs = paired[
        ["Method_A", "Method_B"]
    ].drop_duplicates()

    for _, pair in pairs.iterrows():
        method_a = pair["Method_A"]
        method_b = pair["Method_B"]

        part = paired[
            (paired["Method_A"] == method_a)
            & (paired["Method_B"] == method_b)
        ].sort_values("Window")

        fig, ax = plt.subplots(figsize=(8, 5))

        y = part["Difference_A_minus_B"].to_numpy(dtype=float)
        low = y - part["CI_Low"].to_numpy(dtype=float)
        high = part["CI_High"].to_numpy(dtype=float) - y

        ax.errorbar(
            part["Window"],
            y,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=3,
        )

        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Observation Window End Day")
        ax.set_ylabel(f"Error difference: {method_a} - {method_b}")
        ax.set_title("Paired Method Difference with Bootstrap 95% CI")

        safe = (
            f"{method_a}_vs_{method_b}"
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        savefig(
            fig,
            figure_dir,
            f"fig03_method_difference_{safe}",
        )


def plot_group_change_distribution(
    summary: pd.DataFrame,
    figure_dir: Path,
):
    if summary.empty:
        return

    for label, part in summary.groupby("SimilarityLabel"):
        pivot = part.pivot_table(
            index="ToWindow",
            columns="Category",
            values="Proportion",
            fill_value=0,
        )

        for col in ["improved", "stable", "worsened"]:
            if col not in pivot.columns:
                pivot[col] = 0.0

        pivot = pivot[
            ["improved", "stable", "worsened"]
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        bottom = np.zeros(len(pivot))

        for col in pivot.columns:
            values = pivot[col].to_numpy(dtype=float)

            ax.bar(
                pivot.index.astype(int),
                values,
                bottom=bottom,
                label=col,
            )

            bottom += values

        ax.set_xlabel("To Window End Day")
        ax.set_ylabel("Proportion of Groups")
        ax.set_ylim(0, 1)
        ax.set_title(f"Group-level Window Change: {label}")
        ax.legend()

        safe = label.lower().replace(" ", "_")

        savefig(
            fig,
            figure_dir,
            f"fig04_group_change_distribution_{safe}",
        )


def save_table(df: pd.DataFrame, path: Path):
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    error_ci: pd.DataFrame,
    gain_ci: pd.DataFrame,
    paired: pd.DataFrame,
):
    print("=" * 92)
    print("05 稳健性分析完成（已与 04 的 Core3 NRMSE 定义统一）")
    print("=" * 92)

    print("\nBootstrap Error(k):")

    print(
        error_ci[
            [
                "Window",
                "SimilarityLabel",
                "Error_Point",
                "Error_CI_Low",
                "Error_CI_High",
            ]
        ]
        .sort_values(
            ["SimilarityLabel", "Window"]
        )
        .to_string(index=False)
    )

    print("\nBootstrap Gain(k):")

    print(
        gain_ci[
            [
                "FromWindow",
                "ToWindow",
                "SimilarityLabel",
                "Gain_Point",
                "Gain_CI_Low",
                "Gain_CI_High",
                "P_Gain_GT_0",
            ]
        ]
        .sort_values(
            ["SimilarityLabel", "ToWindow"]
        )
        .to_string(index=False)
    )

    if not paired.empty:
        print("\n方法间配对 Error 差：")

        print(
            paired[
                [
                    "Window",
                    "Method_A",
                    "Method_B",
                    "Difference_A_minus_B",
                    "CI_Low",
                    "CI_High",
                ]
            ]
            .sort_values(
                ["Window", "Method_A", "Method_B"]
            )
            .to_string(index=False)
        )

    print(
        "\n检查点：05 的 Error_Point 应与 04 的 Core3_NRMSE_Mean 对应一致。"
    )
    print(
        "Bootstrap 单位为 Group，未把 Group-Day 行当作独立样本。"
    )
    print("=" * 92)


def main():
    args = parse_args()

    if args.bootstrap < 100:
        raise ValueError(
            "--bootstrap 建议至少 100；正式分析建议 5000。"
        )

    table_dir = args.output / "table"
    figure_dir = args.output / "figure"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.input)
    sim = prepare_similarity_data(pred)

    rng = np.random.default_rng(args.seed)

    error_ci = bootstrap_error_ci(
        sim,
        args.bootstrap,
        rng,
    )

    gain_ci = bootstrap_gain_ci(
        sim,
        args.bootstrap,
        rng,
    )

    paired = paired_method_difference(
        sim,
        args.bootstrap,
        rng,
    )

    group_errors = build_group_core3_error(sim)

    changes = group_window_changes(
        group_errors,
        args.change_tol,
    )

    summary = build_robustness_summary(changes)

    save_table(
        error_ci,
        table_dir / "bootstrap_error_ci.csv",
    )

    save_table(
        gain_ci,
        table_dir / "bootstrap_gain_ci.csv",
    )

    save_table(
        paired,
        table_dir / "paired_method_difference.csv",
    )

    save_table(
        changes,
        table_dir / "group_window_changes.csv",
    )

    save_table(
        summary,
        table_dir / "robustness_summary.csv",
    )

    plot_error_ci(
        error_ci,
        figure_dir,
    )

    plot_gain_ci(
        gain_ci,
        figure_dir,
    )

    plot_method_difference(
        paired,
        figure_dir,
    )

    plot_group_change_distribution(
        summary,
        figure_dir,
    )

    print_summary(
        error_ci,
        gain_ci,
        paired,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] 05 稳健性分析失败：{exc}",
            file=sys.stderr,
        )
        raise
