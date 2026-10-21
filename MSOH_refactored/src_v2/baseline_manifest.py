"""
Phase 0: fingerprint the frozen V1 code/results without modifying them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for p in paths:
        if p.is_dir():
            expanded.extend(sorted(x for x in p.rglob("*") if x.is_file()))
        else:
            expanded.append(p)
    # preserve order, remove duplicates
    seen = set()
    out = []
    for p in expanded:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def build_manifest(paths: list[Path], project_root: Path) -> pd.DataFrame:
    rows = []
    for p in expand_paths(paths):
        exists = p.exists() and p.is_file()
        try:
            rel = p.resolve().relative_to(project_root.resolve())
            shown = str(rel)
        except Exception:
            shown = str(p)

        rows.append({
            "Path": shown,
            "Exists": bool(exists),
            "Size_Bytes": int(p.stat().st_size) if exists else None,
            "Modified_NS": int(p.stat().st_mtime_ns) if exists else None,
            "SHA256": sha256_file(p) if exists else None,
        })
    return pd.DataFrame(rows)


def snapshot_known_metrics(project_root: Path) -> pd.DataFrame:
    """
    Copy selected V1 summary rows into a compact audit table if the files exist.
    This function reads only; it never edits V1 output.
    """
    candidates = [
        (
            "04_error_gain",
            project_root / "results" / "04_output" / "table" / "error_gain_core3.csv",
        ),
        (
            "06_msoh",
            project_root / "results" / "06_output" / "table" / "msoh_summary.csv",
        ),
        (
            "09_signal_msoh",
            project_root / "results" / "09_signal_specific_msoh" / "table"
            / "signal_specific_msoh_summary.csv",
        ),
    ]

    frames = []
    for label, path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df.copy()
        df.insert(0, "Snapshot_Source", label)
        df.insert(1, "Snapshot_Path", str(path.relative_to(project_root)))
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["Snapshot_Source", "Snapshot_Path", "Note"]
        ).assign(Note=["No known V1 metric table found at configured paths."])

    # CSV schemas differ, so concat union is intentional.
    return pd.concat(frames, ignore_index=True, sort=False)
