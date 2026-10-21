"""
MSOH V2 configuration.

Phase 0-1 implementation:
- V1 remains untouched.
- V2 writes only under results_v2/.
- Four signals are registered explicitly.
- Defecation is treated as a planned sparse longitudinal signal.
- Thresholds remain the pre-specified exploratory defaults from V1, but are
  stored per-signal so they can be overridden later using independent
  biological/measurement justification.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Reuse V1 cleaned data; do not rewrite it in V2.
CLEAN_DATA_PATH = PROJECT_ROOT / "results" / "01_output" / "animal_data_clean.csv"

# V2 has an isolated output root.
RESULTS_V2 = PROJECT_ROOT / "results_v2"

V1_BASELINE_MANIFEST = RESULTS_V2 / "00_baseline" / "v1_baseline_manifest.csv"
V1_KEY_METRICS_SNAPSHOT = RESULTS_V2 / "00_baseline" / "v1_key_metrics_snapshot.csv"
SIGNAL_REGISTRY_OUTPUT = RESULTS_V2 / "01_registry" / "signal_registry.csv"
SPARSE_SCHEDULE_OUTPUT = RESULTS_V2 / "01_registry" / "sparse_schedule.csv"
SPARSE_TARGET_OUTPUT = RESULTS_V2 / "01_registry" / "sparse_target_availability.csv"
PHASE1_AUDIT_OUTPUT = RESULTS_V2 / "01_registry" / "phase1_audit.json"

WINDOWS = [2, 3, 4, 5, 6]
TARGET_DAYS = [7, 8, 9]

# Exploratory defaults carried forward without post-hoc relaxation.
DEFAULT_THRESHOLDS = {
    "error_threshold": 0.40,
    "gain_delta": 0.03,
    "alpha": 0.10,
    "stability_threshold": 0.75,
    "group_epsilon": 0.03,
}

# Candidate predictor panel from the V2 plan. Gate A must approve the exact
# inner-CV scoring/tie-break rules before Phase 2 formal predictions.
DEFAULT_PREDICTOR_PANEL = [
    "euclidean_feature",
    "dtw_core3",
    "dtw_all4",
    "ridge_direct",
    "mixed_effects",
]

# Spline df3 remains sensitivity-only and is not in the default main panel.
SENSITIVITY_ONLY_MODELS = ["status_spline_mixed_df3"]

SIGNAL_SPECS = {
    "Weight": {
        "role": "core3",
        "density": "dense",
        "outcome_kind": "continuous",
        "observed_col": "Weight_observed",
        "predictor_panel": DEFAULT_PREDICTOR_PANEL,
    },
    "Status_Score": {
        "role": "core3",
        "density": "dense",
        "outcome_kind": "continuous_group_mean",
        "observed_col": "Status_Score_observed",
        "predictor_panel": DEFAULT_PREDICTOR_PANEL,
    },
    "Feeding": {
        "role": "core3",
        "density": "dense",
        "outcome_kind": "continuous",
        "observed_col": "Feeding_observed",
        "predictor_panel": DEFAULT_PREDICTOR_PANEL,
    },
    "Defecation": {
        "role": "extended_all4",
        "density": "planned_sparse",
        "outcome_kind": "continuous_sparse",
        "observed_col": "Defecation_observed",
        # Gate A/B will decide the exact sparse-signal panel.
        "predictor_panel": ["dtw_all4", "ridge_direct", "mixed_effects"],
    },
}

# Files/directories to fingerprint for Phase 0 if they exist locally.
# Missing paths are recorded as missing; V2 never creates/changes V1 outputs.
V1_BASELINE_PATHS = [
    PROJECT_ROOT / "config.py",
    PROJECT_ROOT / "01_prepare_data.py",
    PROJECT_ROOT / "02_features.py",
    PROJECT_ROOT / "03_models_cv.py",
    PROJECT_ROOT / "03_models_cv_probes.py",
    PROJECT_ROOT / "04_analysis.py",
    PROJECT_ROOT / "05_robustness.py",
    PROJECT_ROOT / "06_msoh.py",
    PROJECT_ROOT / "07_signal_contribution.py",
    PROJECT_ROOT / "08_signal_error_decomposition.py",
    PROJECT_ROOT / "09_signal_specific_msoh.py",
    PROJECT_ROOT / "10_status_ordinal_probe.py",
    PROJECT_ROOT / "11_status_nonlinear_probe.py",
    PROJECT_ROOT / "results" / "04_output",
    PROJECT_ROOT / "results" / "06_output",
    PROJECT_ROOT / "results" / "09_signal_specific_msoh",
]
