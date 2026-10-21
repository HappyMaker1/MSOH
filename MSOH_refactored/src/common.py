from pathlib import Path
import pandas as pd

SIMILARITY_METHODS = ["euclidean_feature", "dtw_core3", "dtw_all4"]
PROBE_METHODS = ["ridge", "mixed_effects"]
SIMILARITY_PREFIX = "Similarity_"
PROBE_PREFIX = "Probe_"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    return pd.read_csv(path)


def model_name(method: str, k: int) -> str:
    return {
        "euclidean_feature": f"Similarity_EuclideanFeature_K{k}",
        "dtw_core3": f"Similarity_DTW_Core3_K{k}",
        "dtw_all4": f"Similarity_DTW_All4_K{k}",
    }[method]


def probe_model_name(method: str) -> str:
    return {
        "ridge": "Probe_Ridge_Direct",
        "mixed_effects": "Probe_MixedEffects_BLUP",
    }[method]


def similarity_label(model: str) -> str:
    """Backward-compatible display label for all prediction probes.

    Historical output tables use the column name ``SimilarityLabel``.  We keep
    that schema for regression compatibility, but the mapping now also covers
    non-similarity prediction probes.
    """
    return {
        "Similarity_EuclideanFeature_K2": "Euclidean Feature",
        "Similarity_DTW_Core3_K2": "DTW Core3",
        "Similarity_DTW_All4_K2": "DTW All4",
        "Probe_Ridge_Direct": "Ridge Direct",
        "Probe_MixedEffects_BLUP": "Mixed Effects",
    }.get(str(model), str(model))


def is_similarity_model(model: str) -> bool:
    return str(model).startswith(SIMILARITY_PREFIX)


def is_prediction_probe(model: str) -> bool:
    """Models allowed to define Error(k)/Gain(k)/MSOH.

    Includes the original three trajectory-similarity models plus the two
    diagnostic prediction probes introduced for model-family robustness.
    """
    x = str(model)
    return x.startswith(SIMILARITY_PREFIX) or x.startswith(PROBE_PREFIX)


def ensure_dirs(root: Path):
    table = root / "table"
    figure = root / "figure"
    table.mkdir(parents=True, exist_ok=True)
    figure.mkdir(parents=True, exist_ok=True)
    return table, figure


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
