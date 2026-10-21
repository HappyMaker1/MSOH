# MSOH Refactored

本目录是对原 01–07 脚本的结构化重构。核心目标是 **Single Source of Truth**：DTW、LOOG、Core3 NRMSE、Group bootstrap、MSOH 判据和信号贡献各自只实现一次，避免 03/07 再次出现算法漂移。

## 目录

```text
MSOH_refactored/
├─ config.py
├─ run_pipeline.py
├─ requirements.txt
├─ 01_prepare_data.py
├─ 02_features.py
├─ 03_models_cv.py
├─ 04_analysis.py
├─ 05_robustness.py
├─ 06_msoh.py
├─ 07_signal_contribution.py
├─ src/
│  ├─ __init__.py
│  ├─ common.py
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ distance.py
│  ├─ prediction.py
│  ├─ metrics.py
│  ├─ bootstrap.py
│  ├─ msoh.py
│  ├─ attribution.py
│  ├─ analysis.py
│  └─ plots.py
└─ tests/
   └─ test_core_consistency.py
```

## 模块职责

- `config.py`：所有共享参数、路径、窗口、阈值。
- `src/distance.py`：唯一 DTW / Euclidean 实现。
- `src/prediction.py`：唯一 LOOG、KNN 权重预测、基线预测实现。
- `src/metrics.py`：唯一 Error(k)、NRMSE、Gain、Trend 指标实现。
- `src/bootstrap.py`：唯一 Group-level bootstrap 实现；支持 paired resampling。
- `src/msoh.py`：唯一 MSOH 判定和阈值敏感性实现。
- `src/attribution.py`：LOSO 信号贡献、baseline validation、ΔMSOH。
- `01–07`：仅保留参数解析、调用共享模块、保存输出。

## 运行

依旧可以逐步运行：

```powershell
python .\01_prepare_data.py
python .\02_features.py
python .\03_models_cv.py --similarity-method euclidean_feature
python .\03_models_cv.py --similarity-method dtw_core3
python .\03_models_cv.py --similarity-method dtw_all4
python .\04_analysis.py
python .\05_robustness.py
python .\06_msoh.py
python .\07_signal_contribution.py
```

或一次运行完整流程：

```powershell
python .\run_pipeline.py
```

只运行某一步：

```powershell
python .\run_pipeline.py --stage 7
```

## 与旧版的关键对应

- 原 03 的 DTW 定义被固定到 `src/distance.py::dtw_distance`。
- 07 不再拥有自己的 DTW 实现，而是直接调用同一个距离函数。
- 05/06/07 的 bootstrap 均使用 `src/bootstrap.py`。
- 06/07 的 MSOH 判据均使用 `src/msoh.py::decide_msoh`。
- 07 仍保留 All4 对 03/04 基准预测的强制一致性校验。

## 多数据集扩展

下一阶段建议把 `config.py` 再升级为 `config/dataset_01.yaml`、`dataset_02.yaml`、`dataset_03.yaml`。目前先保持 Dataset 1 行为稳定，避免重构和多数据集改造同时发生。
