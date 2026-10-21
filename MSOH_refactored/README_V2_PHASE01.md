# MSOH V2 — Phase 0/1 implementation

This package is the first implementation slice of the approved V2 plan.

## Implemented

### Phase 0 — Freeze / regression baseline
`v2_00_freeze_baseline.py`

- fingerprints configured V1 code/results with SHA-256;
- copies selected V1 summary tables into a read-only-style snapshot under
  `results_v2/00_baseline/`;
- never modifies `results/`.

### Phase 1 — Four-signal registry + sparse schedule
`v2_01_signal_registry.py`

- registers Weight, Status_Score, Feeding, Defecation;
- carries forward the fixed exploratory thresholds:
  `tau_E=0.40, delta_G=0.03, alpha=0.10, tau_S=0.75,
  group_epsilon=0.03`;
- detects Defecation observation days from `Defecation_observed`;
- computes effective sparse observation windows without interpolation;
- detects which Day7–9 targets are actually observed;
- audits whether the planned sparse schedule is consistent across groups.

## Deliberately not implemented yet

Formal Phase 2 nested predictions are blocked until:

- **Gate A** approves the exact inner group-CV scoring and tie-break rules;
- **Gate B** approves the Defecation estimability rule.

This is intentional and prevents changing rules after seeing V2 results.

## Installation into the existing project

Copy/merge these files into `E:\MSOH_refactored\`.
Do not delete or overwrite V1 scripts/results.

Expected clean data:
`results/01_output/animal_data_clean.csv`

## Run

```powershell
python .\run_msoh_v2.py --through-phase 1
```

or separately:

```powershell
python .\v2_00_freeze_baseline.py
python .\v2_01_signal_registry.py
```

## Tests

```powershell
python -m pytest .\tests_v2 -q
```

## Expected new directories

```text
results_v2/
  00_baseline/
  01_registry/
```
