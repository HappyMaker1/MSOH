from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2_simstudy" / "raw_longitudinal_dgm"

WINDOWS = [2, 3, 4, 5, 6]
TARGET_DAYS = [7, 8, 9]
SIGNALS = ["Weight", "Status_Score", "Feeding"]

TAU_E = 0.40
DELTA_G = 0.03
ALPHA = 0.10
GROUP_EPSILON = 0.03

RULES = {"gain_only": None, "hard_stability_075": 0.75}

N_GROUPS_GRID = [12, 24, 48]
N_REPLICATES = 250
BOOTSTRAP_B = 1000
BASE_SEED = 20261002
RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class Scenario:
    name: str
    true_msoh: int | None
    obs_noise: float
    future_noise: float
    ancillary_noise: float


SCENARIOS = [
    Scenario("true_D3", 3, 0.18, 0.16, 0.30),
    Scenario("true_D4", 4, 0.18, 0.16, 0.30),
    Scenario("true_D5", 5, 0.18, 0.16, 0.30),
    Scenario("no_msoh", None, 0.18, 0.16, 0.30),
]


def reveal_fraction(day, true_msoh):
    if true_msoh is None:
        return min(0.98, 0.15 + 0.14 * day)
    if day >= true_msoh:
        return 1.0
    return max(0.15, day / true_msoh)


def simulate_dataset(rng, scenario, n_groups):
    rows = []
    z1 = rng.normal(size=n_groups)
    z2 = rng.normal(size=n_groups)
    z3 = rng.normal(size=n_groups)
    b1 = rng.normal(0, 0.20, size=n_groups)
    b2 = rng.normal(0, 0.16, size=n_groups)
    b3 = rng.normal(0, 0.16, size=n_groups)
    u1 = rng.normal(size=n_groups)
    u2 = rng.normal(size=n_groups)
    u3 = rng.normal(size=n_groups)

    for g in range(n_groups):
        lw = 1.7 * z1[g] + 0.6 * z2[g]
        ls = 1.1 * z2[g] - 0.45 * z1[g]
        lf = 0.85 * z3[g] + 0.45 * z1[g]

        for day in range(1, 10):
            if day <= 6:
                frac = reveal_fraction(day, scenario.true_msoh)

                if scenario.true_msoh is None:
                    fw, fs, ff = frac * lw, frac * ls, frac * lf
                    aw = ast = af = 0.0
                elif day <= scenario.true_msoh:
                    fw, fs, ff = frac * lw, frac * ls, frac * lf
                    aw = ast = af = 0.0
                else:
                    # After k*, no fresh future-relevant information is added.
                    # Only ancillary variation independent of future Y is added.
                    fw, fs, ff = lw, ls, lf
                    step = day - scenario.true_msoh
                    aw = scenario.ancillary_noise * (0.35 + 0.10 * step) * u1[g]
                    ast = scenario.ancillary_noise * (0.30 + 0.08 * step) * u2[g]
                    af = scenario.ancillary_noise * (0.30 + 0.08 * step) * u3[g]

                weight = 10 + fw + 0.08 * day + b1[g] + aw + rng.normal(0, scenario.obs_noise)
                status = 5 + fs + 0.025 * day + b2[g] + ast + rng.normal(0, scenario.obs_noise)
                feeding = 3 + ff + 0.015 * day + b3[g] + af + rng.normal(0, scenario.obs_noise)
            else:
                h = day - 6
                weight = 10.8 + lw + 0.10 * h + b1[g] + rng.normal(0, scenario.future_noise)
                status = 5.2 + ls + 0.04 * h + b2[g] + rng.normal(0, scenario.future_noise)
                feeding = 3.2 + lf + 0.03 * h + b3[g] + rng.normal(0, scenario.future_noise)

            rows.append({
                "Group": f"G{g+1:03d}",
                "Day": day,
                "Weight": weight,
                "Status_Score": status,
                "Feeding": feeding,
            })

    return pd.DataFrame(rows)


def feature_vector(df, group, window):
    p = df[(df["Group"] == group) & (df["Day"] <= window)].sort_values("Day")
    feats = []
    for signal in SIGNALS:
        y = p[signal].to_numpy(float)
        x = p["Day"].to_numpy(float)
        feats += [
            float(y[-1]),
            float(np.mean(y)),
            float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0,
        ]
    return np.asarray(feats, float)


def build_future_target_matrix(df, groups):
    output_index = [(s, d) for s in SIGNALS for d in TARGET_DAYS]
    Y = np.empty((len(groups), len(output_index)), float)
    for gi, g in enumerate(groups):
        gp = df[df["Group"] == g]
        for oi, (s, d) in enumerate(output_index):
            Y[gi, oi] = float(gp.loc[gp["Day"] == d, s].iloc[0])
    return Y, output_index


def outer_loog_predictions(df):
    groups = sorted(df["Group"].unique())
    Y, output_index = build_future_target_matrix(df, groups)
    rows = []

    for window in WINDOWS:
        X = np.vstack([feature_vector(df, g, window) for g in groups])

        for test_idx, test_group in enumerate(groups):
            mask = np.ones(len(groups), bool)
            mask[test_idx] = False

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[mask])
            Xte = scaler.transform(X[test_idx:test_idx+1])

            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(Xtr, Y[mask])
            pred = model.predict(Xte)[0]

            for oi, (signal, target_day) in enumerate(output_index):
                rows.append({
                    "Group": test_group,
                    "Window": window,
                    "Signal": signal,
                    "TargetDay": target_day,
                    "y_true": float(Y[test_idx, oi]),
                    "y_pred": float(pred[oi]),
                })

    return pd.DataFrame(rows)


def pooled_nrmse_from_sums(n, sy, sy2, sse):
    n = np.asarray(n, float)
    sy = np.asarray(sy, float)
    sy2 = np.asarray(sy2, float)
    sse = np.asarray(sse, float)
    out = np.full(np.broadcast(n, sy, sy2, sse).shape, np.nan)
    var = (sy2 - sy * sy / n) / (n - 1.0)
    rmse2 = sse / n
    valid = (n > 1) & (var > 0) & (rmse2 >= 0) & np.isfinite(var) & np.isfinite(rmse2)
    out[valid] = np.sqrt(rmse2[valid]) / np.sqrt(var[valid])
    return out


def sufficient_stats(pred, signal):
    p = pred[pred["Signal"] == signal]
    groups = sorted(p["Group"].unique())
    gi = {g:i for i,g in enumerate(groups)}
    wi = {w:i for i,w in enumerate(WINDOWS)}
    shape = (len(groups), len(WINDOWS))
    n = np.zeros(shape)
    sy = np.zeros(shape)
    sy2 = np.zeros(shape)
    sse = np.zeros(shape)

    for (g, w), q in p.groupby(["Group", "Window"], sort=False):
        y = q["y_true"].to_numpy(float)
        pr = q["y_pred"].to_numpy(float)
        a, b = gi[g], wi[int(w)]
        n[a,b] = len(y)
        sy[a,b] = y.sum()
        sy2[a,b] = (y*y).sum()
        sse[a,b] = ((pr-y)**2).sum()

    return n, sy, sy2, sse


def compute_diagnostics(rng, pred, signal):
    n, sy, sy2, sse = sufficient_stats(pred, signal)

    pn = n.sum(0)
    psy = sy.sum(0)
    psy2 = sy2.sum(0)
    psse = sse.sum(0)

    point_error = pooled_nrmse_from_sums(pn, psy, psy2, psse)
    adequacy = point_error <= TAU_E

    full_var = (psy2 - psy*psy/pn) / (pn - 1.0)
    full_sd = np.sqrt(full_var)
    group_error = np.sqrt(sse/n) / full_sd[None,:]

    stability = np.array([
        np.mean(np.abs(group_error[:,i] - group_error[:,i+1]) <= GROUP_EPSILON)
        for i in range(len(WINDOWS)-1)
    ], float)

    ng = n.shape[0]
    idx = rng.integers(0, ng, size=(BOOTSTRAP_B, ng))
    boot_error = np.empty((BOOTSTRAP_B, len(WINDOWS)), float)

    for j in range(len(WINDOWS)):
        bn = n[idx,j].sum(1)
        bsy = sy[idx,j].sum(1)
        bsy2 = sy2[idx,j].sum(1)
        bsse = sse[idx,j].sum(1)
        boot_error[:,j] = pooled_nrmse_from_sums(bn,bsy,bsy2,bsse)

    p_gain = np.array([
        np.mean((boot_error[:,i] - boot_error[:,i+1]) > DELTA_G)
        for i in range(len(WINDOWS)-1)
    ], float)

    return {"Point_Error":point_error, "Adequacy":adequacy, "P_Gain":p_gain, "Stability":stability}


def apply_rule(diag, tau_s):
    local = []
    for i in range(len(WINDOWS)-1):
        pg = diag["P_Gain"][i]
        stab = diag["Stability"][i]
        local.append(pg < ALPHA if tau_s is None else (pg < ALPHA and stab >= tau_s))

    persistent = {}
    for i,w in enumerate(WINDOWS[:-1]):
        persistent[w] = bool(all(local[i:]))

    for i,w in enumerate(WINDOWS[:-1]):
        if diag["Adequacy"][i] and persistent[w]:
            return int(w)
    return None


def classify(truth, estimate):
    if truth is None:
        return "correct_no_msoh" if estimate is None else "false_finite"
    if estimate is None:
        return "not_identified"
    if estimate == truth:
        return "exact"
    return "early" if estimate < truth else "late"


def run():
    rows, examples = [], []
    seeds = np.random.SeedSequence(BASE_SEED).spawn(
        len(SCENARIOS) * len(N_GROUPS_GRID) * N_REPLICATES
    )
    si = 0
    total = len(seeds)
    done = 0
    t0 = time.perf_counter()

    for scenario in SCENARIOS:
        for ng in N_GROUPS_GRID:
            print(f"[START] {scenario.name}, N={ng}")

            for rep in range(N_REPLICATES):
                rng = np.random.default_rng(seeds[si])
                si += 1

                df = simulate_dataset(rng, scenario, ng)
                pred = outer_loog_predictions(df)

                diag = {
                    signal: compute_diagnostics(rng, pred, signal)
                    for signal in SIGNALS
                }

                if rep < 3:
                    for signal in SIGNALS:
                        for j,w in enumerate(WINDOWS):
                            examples.append({
                                "Scenario": scenario.name,
                                "N_Groups": ng,
                                "Replicate": rep,
                                "Signal": signal,
                                "Window": w,
                                "Error": float(diag[signal]["Point_Error"][j]),
                                "Adequate": bool(diag[signal]["Adequacy"][j]),
                            })

                for rule_name, tau_s in RULES.items():
                    per_signal = [apply_rule(diag[s], tau_s) for s in SIGNALS]
                    overall = int(max(per_signal)) if all(x is not None for x in per_signal) else None

                    rows.append({
                        "Scenario": scenario.name,
                        "True_MSOH": scenario.true_msoh,
                        "N_Groups": ng,
                        "Replicate": rep,
                        "Rule": rule_name,
                        "Weight_MSOH": per_signal[0],
                        "Status_MSOH": per_signal[1],
                        "Feeding_MSOH": per_signal[2],
                        "Core3_MSOH": overall,
                        "Recovery_Class": classify(scenario.true_msoh, overall),
                    })

                done += 1
                if rep == 0 or (rep+1) % 25 == 0 or rep+1 == N_REPLICATES:
                    elapsed = time.perf_counter() - t0
                    rate = done/elapsed if elapsed > 0 else np.nan
                    eta = (total-done)/rate/60 if rate > 0 else np.nan
                    print(f"  rep {rep+1:>3}/{N_REPLICATES} | overall {done}/{total} | elapsed {elapsed/60:.1f} min | ETA {eta:.1f} min")

    return pd.DataFrame(rows), pd.DataFrame(examples)


def summarize(results):
    rows = []
    for keys, p in results.groupby(["Scenario","True_MSOH","N_Groups","Rule"], dropna=False, sort=True):
        scenario, truth, ng, rule = keys
        rows.append({
            "Scenario": scenario,
            "True_MSOH": truth,
            "N_Groups": int(ng),
            "Rule": rule,
            "N_Replicates": len(p),
            "Exact_Rate": float((p["Recovery_Class"]=="exact").mean()),
            "Correct_No_MSOH_Rate": float((p["Recovery_Class"]=="correct_no_msoh").mean()),
            "Not_Identified_Rate": float((p["Recovery_Class"]=="not_identified").mean()),
            "False_Finite_Rate": float((p["Recovery_Class"]=="false_finite").mean()),
            "Early_Rate": float((p["Recovery_Class"]=="early").mean()),
            "Late_Rate": float((p["Recovery_Class"]=="late").mean()),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("="*124)
    print("MSOH V2 conditional-sufficiency raw longitudinal DGM")
    print("="*124)
    print(f"replicates/cell={N_REPLICATES}; bootstrap B={BOOTSTRAP_B}; seed={BASE_SEED}")
    print(f"Predictor: Ridge Direct (alpha={RIDGE_ALPHA}) with outer LOOG")
    print(f"Frozen: tau_E={TAU_E}, delta_G={DELTA_G}, alpha={ALPHA}, group_epsilon={GROUP_EPSILON}")
    print("Finite-scenario truth: no new future-relevant latent information is added after k*.")
    print()

    t0 = time.perf_counter()
    results, examples = run()
    summary = summarize(results)
    elapsed = time.perf_counter() - t0

    results.to_csv(OUT/"replicate_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT/"raw_dgm_summary.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(OUT/"example_signal_error_curves.csv", index=False, encoding="utf-8-sig")

    print("\nOperating characteristics:")
    print(summary[
        ["Scenario","True_MSOH","N_Groups","Rule","Exact_Rate","Correct_No_MSOH_Rate",
         "Not_Identified_Rate","False_Finite_Rate","Early_Rate","Late_Rate"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nElapsed: {elapsed/60:.2f} minutes")
    print(f"Output: {OUT}")
    print("="*124)


if __name__ == "__main__":
    main()
