from __future__ import annotations

import hashlib
import json
import runpy
import sys
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import config_v2
from src.preprocessing import (
    load_raw,
    prepare_data,
    group_day_audit,
    missingness_by_group,
    audit_summary,
)

ROOT = Path(__file__).resolve().parent
RAW_INPUT = ROOT / "data" / "animal_data.xlsx"
SIM_RESULTS = ROOT / "results_v2_simulation"
CLEAN_DIR = SIM_RESULTS / "01_output"
CLEAN_DATA = CLEAN_DIR / "animal_data_clean.csv"

PHASE_SCRIPTS = [
    "v2_01_signal_registry.py",
    "v2_03_signal_predictions.py",
    "v2_04_signal_metrics.py",
    "v2_05_uncertainty.py",
    "v2_06_signal_msoh.py",
    "v2_07_overall_msoh.py",
]

# v2_01_signal_registry.py in the current project writes to this historical
# fixed folder instead of respecting config_v2.RESULTS_V2. The simulation
# runner therefore handles Phase 1 transactionally: back up the real-data
# registry, run Phase 1, copy its outputs into results_v2_simulation, then
# restore the real-data registry exactly.
REAL_REGISTRY_DIR = ROOT / "results_v2" / "01_registry"
SIM_REGISTRY_DIR = SIM_RESULTS / "01_registry"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_raw_columns_for_audit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cleaned = [str(c).strip() for c in out.columns]

    if len(cleaned) != len(set(cleaned)):
        raise ValueError(
            "字段名 strip 后出现重复列名，请检查模拟 Excel。"
        )

    out.columns = cleaned
    return out


def raw_schedule_audit(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_raw_columns_for_audit(df)

    required = {
        "Group",
        "Day",
        "Weight",
        "Status_Score",
        "Feeding",
        "Defecation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "模拟数据缺少必要字段："
            f"{sorted(missing)}\n"
            f"当前字段：{list(df.columns)}"
        )

    rows = []

    for day, part in df.groupby("Day", sort=True):
        x = pd.to_numeric(
            part["Defecation"],
            errors="coerce",
        )

        rows.append(
            {
                "Day": int(day),
                "N_Rows": int(len(part)),
                "N_NonMissing": int(x.notna().sum()),
                "N_NonZero": int((x.fillna(0) != 0).sum()),
                "Min": float(x.min()) if x.notna().any() else None,
                "Max": float(x.max()) if x.notna().any() else None,
            }
        )

    return pd.DataFrame(rows)


def prepare_simulation_data():
    if not RAW_INPUT.exists():
        raise FileNotFoundError(
            f"找不到模拟数据：{RAW_INPUT}"
        )

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_raw(
        RAW_INPUT,
        0,
    )

    schedule = raw_schedule_audit(
        raw
    )

    clean = prepare_data(
        raw
    )

    clean.to_csv(
        CLEAN_DATA,
        index=False,
        encoding="utf-8-sig",
    )

    gd = group_day_audit(
        clean
    )

    miss = missingness_by_group(
        clean
    )

    summary = audit_summary(
        clean
    )

    gd.to_csv(
        CLEAN_DIR / "group_day_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    miss.to_csv(
        CLEAN_DIR / "missingness_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        CLEAN_DIR / "data_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    schedule.to_csv(
        CLEAN_DIR / "raw_defecation_schedule_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Explicit conversion to native Python scalar types so json.dump()
    # never sees numpy.int64 / numpy.str_ objects.
    groups = [
        str(x)
        for x in sorted(
            clean["Group"].astype(str).unique().tolist()
        )
    ]

    days = [
        int(x)
        for x in sorted(
            clean["Day"].astype(int).unique().tolist()
        )
    ]

    manifest = {
        "raw_input": str(RAW_INPUT),
        "raw_sha256": str(sha256(RAW_INPUT)),
        "clean_data": str(CLEAN_DATA),
        "clean_sha256": str(sha256(CLEAN_DATA)),
        "n_rows": int(len(clean)),
        "n_groups": int(len(groups)),
        "groups": groups,
        "days": days,
    }

    with (
        CLEAN_DIR / "simulation_input_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 108)
    print("Simulation input audit")
    print("=" * 108)
    print(f"Raw input:  {RAW_INPUT}")
    print(f"Clean data: {CLEAN_DATA}")
    print(f"Rows: {len(clean)}")
    print(f"Groups ({len(groups)}): {groups}")
    print(f"Days: {days}")
    print("\nRaw Defecation schedule audit:")
    print(schedule.to_string(index=False))
    print("=" * 108)


def patch_config_for_simulation():
    config_v2.CLEAN_DATA_PATH = CLEAN_DATA
    config_v2.RESULTS_V2 = SIM_RESULTS


def _copy_tree_contents(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        raise FileNotFoundError(
            f"找不到目录：{src_dir}"
        )

    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in src_dir.iterdir():
        target = dst_dir / item.name

        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def run_phase1_transactionally(script: Path):
    """
    Current Phase 1 has a historical fixed output directory:
        results_v2/01_registry

    To keep real-data results frozen:
    1) snapshot the existing real registry;
    2) run Phase 1 using the simulation CLEAN_DATA_PATH;
    3) copy the newly produced registry into results_v2_simulation/01_registry;
    4) restore the original real registry byte-for-byte.
    """
    with tempfile.TemporaryDirectory(
        prefix="msoh_real_registry_backup_"
    ) as td:
        backup = Path(td) / "01_registry"
        had_real_registry = REAL_REGISTRY_DIR.exists()

        if had_real_registry:
            shutil.copytree(
                REAL_REGISTRY_DIR,
                backup,
            )

        old_argv = sys.argv[:]

        try:
            sys.argv = [str(script)]

            runpy.run_path(
                str(script),
                run_name="__main__",
            )

            if not REAL_REGISTRY_DIR.exists():
                raise FileNotFoundError(
                    "Phase 1 运行后未找到预期输出目录："
                    f"{REAL_REGISTRY_DIR}"
                )

            if SIM_REGISTRY_DIR.exists():
                shutil.rmtree(
                    SIM_REGISTRY_DIR
                )

            shutil.copytree(
                REAL_REGISTRY_DIR,
                SIM_REGISTRY_DIR,
            )

        finally:
            sys.argv = old_argv

            if REAL_REGISTRY_DIR.exists():
                shutil.rmtree(
                    REAL_REGISTRY_DIR
                )

            if had_real_registry:
                shutil.copytree(
                    backup,
                    REAL_REGISTRY_DIR,
                )

    required = [
        SIM_REGISTRY_DIR / "signal_registry.csv",
        SIM_REGISTRY_DIR / "sparse_schedule.csv",
        SIM_REGISTRY_DIR / "sparse_target_availability.csv",
        SIM_REGISTRY_DIR / "phase1_audit.json",
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Simulation Phase 1 relocation failed; missing: "
            + ", ".join(missing)
        )

    print(
        "Simulation Phase 1 outputs safely relocated to:",
        SIM_REGISTRY_DIR,
    )


def run_phase(script_name: str):
    script = ROOT / script_name

    if not script.exists():
        raise FileNotFoundError(
            f"找不到 V2 phase script：{script}"
        )

    print(
        "\\n>>> SIMULATION:",
        script_name,
    )

    if script_name == "v2_01_signal_registry.py":
        run_phase1_transactionally(
            script
        )
        return

    old_argv = sys.argv[:]

    try:
        sys.argv = [str(script)]

        runpy.run_path(
            str(script),
            run_name="__main__",
        )

    finally:
        sys.argv = old_argv


def main():
    prepare_simulation_data()
    patch_config_for_simulation()

    for script in PHASE_SCRIPTS:
        run_phase(
            script
        )

    print("\n" + "=" * 108)
    print("模拟数据 V2 end-to-end pipeline 完成")
    print(f"输出目录: {SIM_RESULTS}")
    print("=" * 108)


if __name__ == "__main__":
    main()
