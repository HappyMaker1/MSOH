from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from config_v2 import (
    CLEAN_DATA_PATH,
    DEFAULT_THRESHOLDS,
    PHASE1_AUDIT_OUTPUT,
    RESULTS_V2,
    SIGNAL_REGISTRY_OUTPUT,
    SIGNAL_SPECS,
    SPARSE_SCHEDULE_OUTPUT,
    SPARSE_TARGET_OUTPUT,
    TARGET_DAYS,
    WINDOWS,
)
from src_v2.signal_registry import (
    build_registry,
    registry_table,
    validate_registry_against_data,
)
from src_v2.sparse_schedule import (
    infer_sparse_schedule,
    sparse_information_count_by_window,
)


def main():
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 V1 清洗数据：{CLEAN_DATA_PATH}\n"
            "V2 Phase 1 只读取 01_prepare_data.py 已生成的清洗结果。"
        )

    df = pd.read_csv(CLEAN_DATA_PATH)
    registry = build_registry(SIGNAL_SPECS, DEFAULT_THRESHOLDS)
    validate_registry_against_data(df, registry)

    out = RESULTS_V2 / "01_registry"
    out.mkdir(parents=True, exist_ok=True)

    reg_table = registry_table(registry)
    reg_table.to_csv(SIGNAL_REGISTRY_OUTPUT, index=False, encoding="utf-8-sig")

    def_spec = registry["Defecation"]
    schedule, day_summary, target_summary = infer_sparse_schedule(
        df=df,
        signal="Defecation",
        observed_col=def_spec.observed_col,
        candidate_windows=WINDOWS,
        target_days=TARGET_DAYS,
    )
    sparse_windows = sparse_information_count_by_window(schedule)

    schedule_table = day_summary.merge(
        sparse_windows,
        left_on="Day",
        right_on="Window",
        how="left",
        suffixes=("", "_window"),
    )
    schedule_table.to_csv(
        SPARSE_SCHEDULE_OUTPUT, index=False, encoding="utf-8-sig"
    )
    target_summary.to_csv(
        SPARSE_TARGET_OUTPUT, index=False, encoding="utf-8-sig"
    )

    audit = {
        "n_rows": int(len(df)),
        "n_groups": int(df["Group"].astype(str).nunique()),
        "days": sorted(df["Day"].dropna().astype(int).unique().tolist()),
        "candidate_windows": list(WINDOWS),
        "target_days": list(TARGET_DAYS),
        "defecation_design_observed_days": list(schedule.design_observed_days),
        "defecation_effective_windows": list(schedule.effective_windows),
        "defecation_observed_target_days": list(schedule.observed_target_days),
        "defecation_schedule_consistent_across_groups":
            schedule.schedule_consistent_across_groups,
        "gate_b_status": "PENDING_REVIEW",
        "gate_b_note": (
            "Phase 1 intentionally does not invent a minimum number of sparse "
            "windows/targets required for estimability. Gate B must approve "
            "that rule before formal Defecation MSOH decisions."
        ),
    }
    PHASE1_AUDIT_OUTPUT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 92)
    print("MSOH V2 Phase 1 完成：四信号 registry + Defecation schedule-aware 审计")
    print()
    print(reg_table[[
        "name", "role", "density", "outcome_kind",
        "error_threshold", "gain_delta", "alpha"
    ]].to_string(index=False))
    print()
    print("Defecation:")
    print(f"  真实观测设计日: {list(schedule.design_observed_days)}")
    print(f"  有效候选窗口:   {list(schedule.effective_windows)}")
    print(f"  真实目标日:     {list(schedule.observed_target_days)}")
    print(
        "  跨组日程一致:   "
        f"{schedule.schedule_consistent_across_groups}"
    )
    print()
    print("Gate B 暂不自动通过：只输出事实，不事后发明可估性阈值。")
    print(f"输出: {SIGNAL_REGISTRY_OUTPUT}")
    print(f"输出: {SPARSE_SCHEDULE_OUTPUT}")
    print(f"输出: {SPARSE_TARGET_OUTPUT}")
    print(f"输出: {PHASE1_AUDIT_OUTPUT}")
    print("=" * 92)


if __name__ == "__main__":
    main()
