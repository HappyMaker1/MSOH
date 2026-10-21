# MSOH: Minimum Sufficient Observation Horizon

动物实验早期纵向预测与最小观察窗口研究项目。

## 研究问题

动物实验成本高、周期长、持续监测困难，并非所有预设观察时点都能提供等量的新信息。本项目关注：

> 动物实验究竟需要观察多久，才能获得足够的信息去预测后续实验轨迹？

即：**继续增加观察时间，从什么时候开始已经不能带来实质性的预测性能提升。**

## 核心方法：MSOH

**MSOH（Minimum Sufficient Observation Horizon，最小充分观察窗口）**：从实验开始逐步延长早期观察窗口（如 Day1–2 → Day1–6），利用当前窗口内的纵向轨迹统一预测后续轨迹（如 Day7–Day9）；当继续增加观察时间不再带来实质性预测提升时，最早达到该状态的观察窗口即为候选 MSOH。

## 目录结构

```text
├─ Minimum_Observation_Window/   # 原始分析管线（01–07 脚本）
│  ├─ 01_prepare_data.py         # 数据准备
│  ├─ 02_features.py             # 特征构建
│  ├─ 03_models_cv_unified.py    # 交叉验证模型
│  ├─ 04_analysis_plot_unified.py# 分析与绘图
│  ├─ 05_robustness_corrected.py # 稳健性检验
│  ├─ 06_msoh_decision.py        # MSOH 判定
│  ├─ 07_signal_contribution_bootstrap.py # 信号贡献（bootstrap）
│  ├─ data/                      # 示例数据
│  ├─ doc/                       # 技术路线文档
│  └─ results/                   # 分析输出
│
└─ MSOH_refactored/              # 结构化重构版（Single Source of Truth）
   ├─ config.py                  # 共享参数、路径、窗口、阈值
   ├─ src/                       # 共享模块（DTW/LOOG/NRMSE/bootstrap/MSOH 各实现一次）
   ├─ src_v2/                    # v2 信号注册与判定框架
   ├─ 01–07 + run_pipeline.py    # 管线入口
   ├─ v2_00–v2_14                # v2 信号度量/模拟/消融实验
   ├─ tests/, tests_v2/          # 一致性测试
   └─ doc/                       # 文档与排查记录
```

## 运行

```powershell
pip install -r MSOH_refactored/requirements.txt

cd MSOH_refactored
python .\run_pipeline.py
# 或逐步运行
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

依赖：numpy、pandas、matplotlib、scikit-learn、openpyxl
