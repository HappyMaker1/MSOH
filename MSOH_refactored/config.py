from pathlib import Path

# Dataset-1 defaults. Future datasets can override these through CLI/config.
ALL_SIGNALS = ["Weight", "Status_Score", "Feeding", "Defecation"]
CORE_SIGNALS = ["Weight", "Status_Score", "Feeding"]
WINDOWS = [2, 3, 4, 5, 6]
TARGET_DAYS = [7, 8, 9]
EXPECTED_DAYS = list(range(1, 10))

K_NEIGHBORS = 2
DTW_RADIUS = 1
DEFECATION_WEIGHT = 1.0
EPS = 1e-8

BOOTSTRAP_B = 5000
RANDOM_SEED = 20260924

# Exploratory MSOH thresholds; do not treat as locked scientific cutoffs.
ERROR_THRESHOLD = 0.40
GAIN_DELTA = 0.03
ALPHA = 0.10
STABILITY_THRESHOLD = 0.75
GROUP_EPSILON = 0.03
CONTRIBUTION_DELTA = 0.03
BASELINE_TOL = 1e-6
DEFECATION_MEASUREMENT_DAYS = [2, 4, 6, 9]
DATA_PATH = Path("data/animal_data_true.xlsx")
CLEAN_PATH = Path("results/01_output/animal_data_clean.csv")
FEATURE_PATH = Path("results/02_output/early_features.csv")
PREDICTION_DIR = Path("results/03_output")
ANALYSIS_DIR = Path("results/04_output")
ROBUSTNESS_DIR = Path("results/05_output")
MSOH_DIR = Path("results/06_output")
ATTRIBUTION_DIR = Path("results/07_output")
