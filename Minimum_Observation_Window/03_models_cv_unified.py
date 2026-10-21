from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_PATH = Path("results/01_output/animal_data_clean.csv")
DEFAULT_FEATURE_PATH = Path("results/02_output/early_features.csv")
DEFAULT_OUTPUT_DIR = Path("results/03_output")

WINDOWS = [2, 3, 4, 5, 6]
TARGET_DAYS = [7, 8, 9]
ALL_SIGNALS = ["Weight", "Status_Score", "Feeding", "Defecation"]
CORE3_SIGNALS = ["Weight", "Status_Score", "Feeding"]
SIMILARITY_METHODS = ["euclidean_feature", "dtw_core3", "dtw_all4"]
EPS = 1e-8

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--similarity-method", choices=SIMILARITY_METHODS, default="euclidean_feature")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--dtw-radius", type=int, default=1)
    p.add_argument("--defecation-weight", type=float, default=1.0)
    return p.parse_args()

def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def model_name(method, k):
    return {
        "euclidean_feature": f"Similarity_EuclideanFeature_K{k}",
        "dtw_core3": f"Similarity_DTW_Core3_K{k}",
        "dtw_all4": f"Similarity_DTW_All4_K{k}",
    }[method]

def get_truth(df, group):
    out = {}
    part = df[(df["Group"].astype(str) == str(group)) & df["Day"].isin(TARGET_DAYS)]
    for signal in ALL_SIGNALS:
        for day in TARGET_DAYS:
            row = part[part["Day"] == day]
            out[(signal, day)] = float(row.iloc[0][signal]) if len(row) and not pd.isna(row.iloc[0][signal]) else np.nan
    return out

def predict_locf(df, group, window):
    part = df[(df["Group"].astype(str) == str(group)) & (df["Day"] <= window)].sort_values("Day")
    out = {}
    for signal in ALL_SIGNALS:
        valid = part.dropna(subset=[signal])
        value = float(valid.iloc[-1][signal]) if len(valid) else np.nan
        for day in TARGET_DAYS:
            out[(signal, day)] = value
    return out

def predict_linear(df, group, window):
    part = df[(df["Group"].astype(str) == str(group)) & (df["Day"] <= window)].sort_values("Day")
    out = {}
    for signal in ALL_SIGNALS:
        valid = part.dropna(subset=[signal])
        if len(valid) >= 2:
            slope, intercept = np.polyfit(valid["Day"].to_numpy(float), valid[signal].to_numpy(float), 1)
            vals = {day: float(intercept + slope * day) for day in TARGET_DAYS}
        elif len(valid) == 1:
            vals = {day: float(valid.iloc[0][signal]) for day in TARGET_DAYS}
        else:
            vals = {day: np.nan for day in TARGET_DAYS}
        for day, value in vals.items():
            out[(signal, day)] = value
    return out

def predict_global_mean(df, train_groups):
    train = df[df["Group"].astype(str).isin(train_groups) & df["Day"].isin(TARGET_DAYS)]
    out = {}
    for signal in ALL_SIGNALS:
        for day in TARGET_DAYS:
            s = train.loc[train["Day"] == day, signal].dropna()
            out[(signal, day)] = float(s.mean()) if len(s) else np.nan
    return out

def euclidean_feature_neighbors(feature_df, test_group, train_groups, window):
    cols = []
    for signal in ALL_SIGNALS:
        cols += [f"{signal}_last", f"{signal}_slope"]
    part = feature_df[feature_df["Window"] == window].copy()
    part["Group"] = part["Group"].astype(str)
    train = part[part["Group"].isin(train_groups)].set_index("Group").loc[train_groups].reset_index()
    test = part[part["Group"] == test_group]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = StandardScaler()
    xtr = sc.fit_transform(imp.fit_transform(train[cols]))
    xte = sc.transform(imp.transform(test[cols]))[0]
    d = np.linalg.norm(xtr - xte, axis=1)
    return pd.DataFrame({
        "NeighborGroup": train_groups,
        "Distance": d,
        "Core3_DTW_Distance": np.nan,
        "Defecation_Distance": np.nan,
        "Defecation_Method": "feature_euclidean",
    }).sort_values("Distance").reset_index(drop=True)

def fit_fold_scaler(df, train_groups, window):
    train = df[df["Group"].astype(str).isin(train_groups) & (df["Day"] <= window)]
    stats = {}
    for signal in ALL_SIGNALS:
        obs = f"{signal}_observed"
        if obs in train.columns:
            vals = train.loc[(train[obs] == 1) & train[signal].notna(), signal].to_numpy(float)
        else:
            vals = train[signal].dropna().to_numpy(float)
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        if np.isclose(sd, 0):
            sd = 1.0
        stats[signal] = (mean, sd)
    return stats

def build_core3_sequence(df, group, window, scaler):
    part = df[(df["Group"].astype(str) == str(group)) & (df["Day"] <= window)].sort_values("Day")
    cols = []
    for signal in CORE3_SIGNALS:
        mean, sd = scaler[signal]
        cols.append((part[signal].to_numpy(float) - mean) / sd)
    return np.column_stack(cols)

def build_def_sequence(df, group, window, scaler):
    part = df[(df["Group"].astype(str) == str(group)) & (df["Day"] <= window)].copy()
    if "Defecation_observed" in part.columns:
        part = part[(part["Defecation_observed"] == 1) & part["Defecation"].notna()]
    else:
        part = part[part["Defecation"].notna()]
    part = part.sort_values("Day")
    mean, sd = scaler["Defecation"]
    return (part["Defecation"].to_numpy(float) - mean) / sd

def dtw_distance(a, b, radius=1):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    if radius is None:
        radius = max(n, m)
    radius = max(radius, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    plen = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            ai, bj = a[i - 1], b[j - 1]
            local = abs(float(ai) - float(bj)) if np.ndim(ai) == 0 else np.linalg.norm(ai - bj)
            prev = min(
                [(cost[i-1,j], plen[i-1,j]), (cost[i,j-1], plen[i,j-1]), (cost[i-1,j-1], plen[i-1,j-1])],
                key=lambda x: x[0]
            )
            if np.isinf(prev[0]):
                continue
            cost[i, j] = prev[0] + local
            plen[i, j] = prev[1] + 1
    return float(cost[n, m] / plen[n, m]) if plen[n, m] else np.nan

def dtw_neighbors(df, test_group, train_groups, window, method, radius, def_weight):
    scaler = fit_fold_scaler(df, train_groups, window)
    test_core = build_core3_sequence(df, test_group, window, scaler)
    test_def = build_def_sequence(df, test_group, window, scaler) if method == "dtw_all4" else None
    rows = []
    for g in train_groups:
        train_core = build_core3_sequence(df, g, window, scaler)
        core_d = dtw_distance(test_core, train_core, radius)
        if method == "dtw_core3":
            total_d, def_d, def_method = core_d, np.nan, "not_used"
        else:
            train_def = build_def_sequence(df, g, window, scaler)
            if len(test_def) == 1 and len(train_def) == 1:
                def_d = float(abs(test_def[0] - train_def[0]))
                def_method = "single_point"
            elif len(test_def) and len(train_def):
                def_d = dtw_distance(test_def, train_def, None)
                def_method = "dtw"
            else:
                def_d = np.nan
                def_method = "missing"
            total_d = core_d if pd.isna(def_d) else float(np.sqrt(core_d**2 + def_weight * def_d**2))
        rows.append({
            "NeighborGroup": g,
            "Distance": total_d,
            "Core3_DTW_Distance": core_d,
            "Defecation_Distance": def_d,
            "Defecation_Method": def_method,
        })
    return pd.DataFrame(rows).sort_values("Distance").reset_index(drop=True)

def compute_neighbors(method, data_df, feature_df, test_group, train_groups, window, radius, def_weight):
    if method == "euclidean_feature":
        return euclidean_feature_neighbors(feature_df, test_group, train_groups, window)
    return dtw_neighbors(data_df, test_group, train_groups, window, method, radius, def_weight)

def predict_similarity(data_df, neighbors, k):
    selected = neighbors.head(k).copy()
    selected["RawWeight"] = 1.0 / (selected["Distance"] + EPS)
    selected["Weight"] = selected["RawWeight"] / selected["RawWeight"].sum()
    out = {}
    for signal in ALL_SIGNALS:
        for day in TARGET_DAYS:
            vals, w = [], []
            for _, r in selected.iterrows():
                target = data_df[(data_df["Group"].astype(str) == str(r["NeighborGroup"])) & (data_df["Day"] == day)]
                if len(target) == 0:
                    continue
                v = target.iloc[0][signal]
                if pd.isna(v):
                    continue
                vals.append(float(v))
                w.append(float(r["Weight"]))
            if len(vals) == 0:
                out[(signal, day)] = np.nan
            else:
                w = np.asarray(w, float)
                w = w / w.sum()
                out[(signal, day)] = float(np.sum(np.asarray(vals) * w))
    return out, selected

def append_rows(rows, group, window, model, truth, pred):
    for signal in ALL_SIGNALS:
        for day in TARGET_DAYS:
            yt, yp = truth[(signal, day)], pred[(signal, day)]
            err = yp - yt if not pd.isna(yt) and not pd.isna(yp) else np.nan
            rows.append({
                "Group": group, "Window": window, "Model": model, "Signal": signal,
                "TargetDay": day, "y_true": yt, "y_pred": yp, "error": err,
                "abs_error": abs(err) if not pd.isna(err) else np.nan,
                "squared_error": err**2 if not pd.isna(err) else np.nan,
            })

def run_loog(data_df, feature_df, method, k, radius, def_weight):
    groups = sorted(data_df["Group"].astype(str).unique())
    pred_rows, neighbor_rows = [], []
    sim_name = model_name(method, k)
    for window in WINDOWS:
        for test_group in groups:
            train_groups = [g for g in groups if g != test_group]
            truth = get_truth(data_df, test_group)
            append_rows(pred_rows, test_group, window, "LOCF", truth, predict_locf(data_df, test_group, window))
            append_rows(pred_rows, test_group, window, "Linear", truth, predict_linear(data_df, test_group, window))
            append_rows(pred_rows, test_group, window, "GlobalMean", truth, predict_global_mean(data_df, train_groups))
            neigh = compute_neighbors(method, data_df, feature_df, test_group, train_groups, window, radius, def_weight)
            neigh["Rank"] = np.arange(len(neigh)) + 1
            pred, selected = predict_similarity(data_df, neigh, k)
            append_rows(pred_rows, test_group, window, sim_name, truth, pred)
            chosen = set(selected["NeighborGroup"].astype(str))
            for _, r in neigh.iterrows():
                neighbor_rows.append({
                    "SimilarityMethod": method,
                    "Model": sim_name,
                    "TestGroup": test_group,
                    "Window": window,
                    "NeighborGroup": r["NeighborGroup"],
                    "Distance": r["Distance"],
                    "Rank": int(r["Rank"]),
                    "Selected": str(r["NeighborGroup"]) in chosen,
                    "Core3_DTW_Distance": r["Core3_DTW_Distance"],
                    "Defecation_Distance": r["Defecation_Distance"],
                    "Defecation_Method": r["Defecation_Method"],
                })
    return pd.DataFrame(pred_rows), pd.DataFrame(neighbor_rows)

def basic_metrics(pred):
    rows = []
    for (window, model, signal), part in pred.groupby(["Window", "Model", "Signal"]):
        valid = part.dropna(subset=["y_true", "y_pred"])
        if len(valid):
            e = (valid["y_pred"] - valid["y_true"]).to_numpy(float)
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e**2)))
        else:
            mae = rmse = np.nan
        rows.append({"Window": window, "Model": model, "Signal": signal, "N": len(valid), "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    data_df = load_csv(args.data)
    feature_df = load_csv(args.features) if args.similarity_method == "euclidean_feature" else None
    pred, neigh = run_loog(
        data_df, feature_df, args.similarity_method, args.k,
        args.dtw_radius, args.defecation_weight
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.similarity_method
    pred_path = args.output_dir / f"predictions_loog_{suffix}.csv"
    neigh_path = args.output_dir / f"similarity_neighbors_{suffix}.csv"
    metric_path = args.output_dir / f"basic_metrics_{suffix}.csv"
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    neigh.to_csv(neigh_path, index=False, encoding="utf-8-sig")
    basic_metrics(pred).to_csv(metric_path, index=False, encoding="utf-8-sig")
    print("=" * 72)
    print("统一版 LOOG 完成")
    print("=" * 72)
    print(f"方法：{args.similarity_method}")
    print(f"模型：{model_name(args.similarity_method, args.k)}")
    print(f"输出目录：{args.output_dir}")
    print("=" * 72)
    print(pred_path)
    print(neigh_path)
    print(metric_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
