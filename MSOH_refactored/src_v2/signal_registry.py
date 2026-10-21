"""
Signal registry for MSOH V2.

This module makes the four-signal design explicit and auditable. It does not
select a winning predictor and does not compute MSOH.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SignalSpec:
    name: str
    role: str
    density: str
    outcome_kind: str
    observed_col: str
    predictor_panel: tuple[str, ...]
    error_threshold: float
    gain_delta: float
    alpha: float
    stability_threshold: float
    group_epsilon: float

    @property
    def is_sparse(self) -> bool:
        return self.density == "planned_sparse"


def build_registry(signal_specs: dict, default_thresholds: dict) -> dict[str, SignalSpec]:
    registry: dict[str, SignalSpec] = {}

    required_thresholds = {
        "error_threshold",
        "gain_delta",
        "alpha",
        "stability_threshold",
        "group_epsilon",
    }
    missing_thresholds = required_thresholds - set(default_thresholds)
    if missing_thresholds:
        raise ValueError(f"Default thresholds missing: {sorted(missing_thresholds)}")

    for name, spec in signal_specs.items():
        overrides = spec.get("thresholds", {})
        values = dict(default_thresholds)
        values.update(overrides)

        registry[name] = SignalSpec(
            name=name,
            role=str(spec["role"]),
            density=str(spec["density"]),
            outcome_kind=str(spec["outcome_kind"]),
            observed_col=str(spec["observed_col"]),
            predictor_panel=tuple(spec["predictor_panel"]),
            error_threshold=float(values["error_threshold"]),
            gain_delta=float(values["gain_delta"]),
            alpha=float(values["alpha"]),
            stability_threshold=float(values["stability_threshold"]),
            group_epsilon=float(values["group_epsilon"]),
        )

    return registry


def validate_registry_against_data(
    df: pd.DataFrame,
    registry: dict[str, SignalSpec],
) -> None:
    required_base = {"Group", "Day"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Clean data missing required columns: {sorted(missing)}")

    for signal, spec in registry.items():
        if signal not in df.columns:
            raise ValueError(f"Clean data missing signal column: {signal}")
        if spec.observed_col not in df.columns:
            raise ValueError(
                f"Clean data missing observed flag for {signal}: {spec.observed_col}"
            )

        obs = pd.to_numeric(df[spec.observed_col], errors="coerce")
        bad = obs.dropna()[~obs.dropna().isin([0, 1])]
        if len(bad):
            raise ValueError(
                f"{spec.observed_col} must contain only 0/1/NaN; "
                f"found examples {bad.head().tolist()}"
            )

        # Strong audit rule: observed=1 requires a real numeric value.
        signal_values = pd.to_numeric(df[signal], errors="coerce")
        inconsistent = (obs == 1) & signal_values.isna()
        if inconsistent.any():
            n = int(inconsistent.sum())
            raise ValueError(
                f"{signal}: {n} rows have observed=1 but missing signal value."
            )

    if df.duplicated(["Group", "Day"]).any():
        raise ValueError("Clean data contains duplicated Group-Day rows.")


def registry_table(registry: dict[str, SignalSpec]) -> pd.DataFrame:
    rows = []
    for spec in registry.values():
        d = asdict(spec)
        d["predictor_panel"] = ";".join(spec.predictor_panel)
        d["is_sparse"] = spec.is_sparse
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["role", "name"]).reset_index(drop=True)
