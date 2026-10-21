from __future__ import annotations
import argparse
from config_v2 import PROJECT_ROOT, RESULTS_V2, V1_BASELINE_MANIFEST, V1_KEY_METRICS_SNAPSHOT, V1_BASELINE_PATHS
from src_v2.baseline_manifest import build_manifest, snapshot_known_metrics

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    a=parse_args()
    out=RESULTS_V2/"00_baseline"; out.mkdir(parents=True,exist_ok=True)
    if V1_BASELINE_MANIFEST.exists() and V1_KEY_METRICS_SNAPSHOT.exists() and not a.force:
        print("="*88)
        print("MSOH V2 Phase 0：baseline 已存在，默认保持冻结，不覆盖。")
        print(f"Manifest: {V1_BASELINE_MANIFEST}")
        print(f"Snapshot: {V1_KEY_METRICS_SNAPSHOT}")
        print("如确需重建，请显式使用 --force。")
        print("="*88)
        return
    manifest=build_manifest(V1_BASELINE_PATHS, PROJECT_ROOT)
    metrics=snapshot_known_metrics(PROJECT_ROOT)
    manifest.to_csv(V1_BASELINE_MANIFEST,index=False,encoding="utf-8-sig")
    metrics.to_csv(V1_KEY_METRICS_SNAPSHOT,index=False,encoding="utf-8-sig")
    print("MSOH V2 Phase 0 完成：V1 baseline 已冻结")

if __name__=="__main__":
    main()
