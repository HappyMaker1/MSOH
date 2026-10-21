---
title: "MSOH 四项创新：可执行研究方案"
aliases:
  - "MSOH Innovation Roadmap"
  - "Minimum Sufficient Observation Horizon Framework"
tags:
  - MSOH
  - animal-experiment
  - longitudinal-data
  - trajectory-prediction
  - uncertainty
  - signal-attribution
  - external-validation
status: active
project: Minimum_Observation_Window
version: 1.0
created: 2026-09-24
---

# MSOH 四项创新：可执行研究方案

> [!abstract]
> 本文件用于统一项目后续的方法学开发路线。核心目标不是“寻找一个最好的预测模型”，而是建立一套可解释、可量化、可验证的 **Minimum Sufficient Observation Horizon（MSOH，最小充分观察窗口）** 方法框架，用于回答：
>
> **动物实验最少需要观察到什么时候，才已经获得足够的未来轨迹预测信息，而继续增加观察时间的边际信息价值已经很低？**

---

# 0. 项目总目标

动物实验通常存在以下现实约束：

- 成本高；
- 周期长；
- 高频、连续监测困难；
- 部分指标需要人工操作或额外实验资源；
- 重复测量可能增加动物和实验人员负担；
- 预设的所有研究性观察时点未必都提供等量的新信息。

因此，本项目不把“未来预测”本身作为终点，而是把预测模型作为 **信息探针（information probe）**，用来衡量：

\[
\text{观察到 Day }k
\]

时，现有纵向轨迹对未来结局还剩多少不确定性。

项目核心逻辑为：

\[
\text{早期纵向轨迹}
\rightarrow
\text{未来轨迹预测}
\rightarrow
Error(k)
\rightarrow
Gain(k)
\rightarrow
\text{Information Saturation}
\rightarrow
MSOH
\]

最终希望得到：

\[
\boxed{
\text{在保证预测信息充分的前提下，识别最早可停止增加研究性观测的时间窗口}
}
\]

> [!warning]
> MSOH 不等于“达到该窗口后停止动物福利或安全监测”。
>
> 本项目针对的是 **研究性重复测量的最小充分观察窗口**。必要的动物福利、安全、伦理和临床状态监测应独立保留。

---

# 1. 四项创新的整体结构

四项创新不是互相独立，而是递进关系：

```mermaid
flowchart LR
    A[纵向动物实验数据] --> B[创新1：定义 MSOH]
    B --> C[创新2：加入不确定性判定]
    C --> D[创新3：分解各信号对 MSOH 的贡献]
    D --> E[创新4：跨数据集验证]
```

可以概括成：

\[
\boxed{
\text{定义}
\rightarrow
\text{可靠判定}
\rightarrow
\text{机制解释}
\rightarrow
\text{泛化验证}
}
\]

项目最终可形成一套：

\[
\boxed{
\text{Uncertainty-aware MSOH Framework}
}
\]

---

# 2. 创新一：MSOH 定义与判定框架

## 2.1 要解决的问题

传统预测问题通常问：

> 给定早期数据，未来预测得准不准？

本项目问的是：

> **为了获得足够的未来预测信息，至少需要观察到什么时候？**

因此我们的核心研究对象不是单个模型，而是：

\[
\boxed{
\text{观察窗口长度}
\rightarrow
\text{未来轨迹预测信息}
}
\]

## 2.2 基本定义

设纵向观察时间为：

\[
t=1,\ldots,T
\]

定义早期观察窗口：

\[
W_k=[1,k]
\]

例如当前 Dataset 1：

- \(W_2=\) Day1–2
- \(W_3=\) Day1–3
- \(W_4=\) Day1–4
- \(W_5=\) Day1–5
- \(W_6=\) Day1–6

统一预测固定未来区间：

\[
Y_{future}=Day7\text{–}Day9
\]

对于每一个窗口 \(k\)，定义预测误差：

\[
E(k)
\]

当前项目中主误差定义为 Core3 NRMSE：

\[
E(k)
=
\frac{1}{3}
\sum_{s \in
\{Weight,Status,Feeding\}}
\frac{RMSE_s(k)}{SD(y_s)}
\]

其中：

\[
RMSE_s(k)
=
\sqrt{
\frac{1}{N}
\sum_i
\left(
y_{i,s}
-
\hat y_{i,s}^{(k)}
\right)^2
}
\]

## 2.3 边际信息增益

定义：

\[
G(k)
=
E(k-1)-E(k)
\]

解释：

- \(G(k)>0\)：新增观察时间提高了未来预测能力；
- \(G(k)\approx0\)：新增观察时间几乎没有提供新的预测信息；
- \(G(k)<0\)：新增窗口没有改善，甚至可能增加噪声。

MSOH 的核心不是：

\[
\arg\min_k E(k)
\]

而是：

\[
\boxed{
\text{最早进入“预测已经充分 + 继续观察收益已经很低”的时间点}
}
\]

## 2.4 第一版 MSOH 判据

可先定义一个基础版本：

\[
MSOH
=
\min
\left\{
k:
E(k)\le \tau_E
\quad\text{且}\quad
G(k+1)\le \tau_G
\right\}
\]

其中：

- \(\tau_E\)：可接受误差阈值；
- \(\tau_G\)：最小有意义的边际改善阈值。

> [!important]
> \(\tau_E\) 与 \(\tau_G\) 不应该在看到结果后为了得到某个窗口而后验调整。
>
> 后续应通过预设原则、多数据集经验、领域可接受误差和灵敏度分析来确定其合理范围。

## 2.5 当前 Dataset 1 的已有证据

### Euclidean Feature

\[
0.5435
\rightarrow
0.5548
\rightarrow
0.4335
\rightarrow
0.4312
\rightarrow
0.4629
\]

### DTW All4

\[
0.5540
\rightarrow
0.5538
\rightarrow
0.3512
\rightarrow
0.3484
\rightarrow
0.3763
\]

### DTW Core3

\[
0.6368
\rightarrow
0.4747
\rightarrow
0.4688
\rightarrow
0.3374
\rightarrow
0.3443
\]

初步现象：

- Euclidean 与 DTW All4 都在 Day3→Day4 出现明显改善；
- DTW All4 在 Day4→Day5 后几乎进入平台；
- DTW Core3 在 Day4→Day5 仍存在较明显改善；
- 因此当前候选窗口主要集中在 Day4–Day5。

## 2.6 可执行任务

### 已完成

- [x] 定义固定未来预测目标 Day7–Day9
- [x] 构建窗口 \(k=2,\ldots,6\)
- [x] 建立 LOOG 预测
- [x] 建立 Error(k)
- [x] 建立 Gain(k)
- [x] 比较 Euclidean / DTW Core3 / DTW All4

### 待完成

- [ ] 正式定义 MSOH 判据
- [ ] 明确 \(\tau_E\)
- [ ] 明确最小有意义 Gain 阈值 \(\delta_G\)
- [ ] 增加连续平台判定规则
- [ ] 在多数据集中验证判据是否稳定

## 2.7 建议新增输出

```text
results/<dataset>/06_msoh/
├─ table/
│  ├─ msoh_candidate_table.csv
│  ├─ msoh_decision_trace.csv
│  └─ msoh_threshold_sensitivity.csv
└─ figure/
   ├─ error_gain_joint_curve.png
   ├─ msoh_decision_map.png
   └─ threshold_sensitivity.png
```

---

# 3. 创新二：不确定性感知的停止 / 饱和规则

## 3.1 为什么需要这一层

仅仅观察：

\[
G(k)=0.003
\]

不能说明继续观察真的没有价值。

当前 Dataset 1 的独立统计单位只有 12 个 Group，因此必须回答：

\[
\boxed{
\text{继续观察还能获得“有实际意义改善”的概率是多少？}
}
\]

## 3.2 从点估计升级到分布

通过 Group-level bootstrap：

\[
b=1,\ldots,B
\]

每一次都重新得到：

\[
E_b(k)
\]

以及：

\[
G_b(k)=E_b(k-1)-E_b(k)
\]

从而形成 Gain 的经验分布。

## 3.3 不仅判断 \(G>0\)

仅计算：

\[
P(G(k)>0)
\]

还不够，因为极小的正改善可能没有实际意义。

更重要的是：

\[
\boxed{
P(G(k)>\delta_G)
}
\]

其中 \(\delta_G\) 表示“最小有实际意义的预测改善”。

例如：

\[
\delta_G=0.03
\]

则：

\[
P(G(k)>0.03)
\]

回答的是：

> 多观察一个窗口后，预测误差至少改善 0.03 的概率有多大？

## 3.4 不确定性感知的 MSOH

建议升级为：

\[
MSOH
=
\min k
\]

满足：

\[
E(k)\le\tau_E
\]

以及：

\[
P(G(k+1)>\delta_G)<\alpha
\]

其中：

- \(\tau_E\)：足够低的当前预测误差；
- \(\delta_G\)：最小实际意义改善；
- \(\alpha\)：仍然可能获得明显改善的最大允许概率。

## 3.5 加入连续平台条件

为避免偶然出现一个低 Gain 窗口，可要求：

\[
P(G(k+1)>\delta_G)<\alpha
\]

并且：

\[
P(G(k+2)>\delta_G)<\alpha
\]

即连续后续窗口均缺乏实质性新增信息。

## 3.6 加入组级稳定性

平均 Gain 接近 0 还可能隐藏组间异质性。

定义：

\[
S(k)
=
P_{group}
\left(
|\Delta E_g(k)|<\epsilon
\right)
\]

最终可形成：

\[
\boxed{
MSOH
=
\min k
}
\]

满足：

\[
E(k)\le\tau_E
\]

\[
P(G(k+1)>\delta_G)<\alpha
\]

\[
S(k+1)\ge\tau_S
\]

即：

> **误差已经足够低 + 平均新增信息不足 + 大多数组也已经稳定。**

## 3.7 为什么不以 p<0.05 为核心

传统显著性检验问：

\[
H_0:G=0
\]

但我们的实际问题是：

> 改善是否已经小到不值得继续增加观察？

因此本项目更接近：

- 等效性思想；
- 非劣性思想；
- 实际意义阈值；
- 序贯停止思想。

> [!important]
> “没有显著改善”不等于“已经证明改善足够小”。
>
> 因此优先报告：Gain 点估计、95% bootstrap CI、\(P(G>0)\)、\(P(G>\delta_G)\)、组级稳定比例。

## 3.8 当前项目对应实现

当前 `05_robustness.py` 已经开始承担这一层任务。

下一版建议正式新增：

```text
P_Gain_GT_delta
P_Gain_LT_delta
Stable_Group_Proportion
Saturation_Flag
Candidate_MSOH
```

## 3.9 建议新增脚本

```text
06_msoh_decision.py
```

职责：

1. 读取 `05_robustness` 结果；
2. 输入 \(\tau_E,\delta_G,\alpha,\tau_S\)；
3. 自动判定每个方法的候选 MSOH；
4. 输出判定轨迹；
5. 做阈值敏感性分析。

---

# 4. 创新三：多信号对 MSOH 的贡献分解

## 4.1 要解决的问题

仅得到：

\[
MSOH=Day4
\]

还不能解释：

> 为什么 Day4 就够了？

进一步的问题是：

\[
\boxed{
\text{究竟哪些纵向信号推动了信息在 Day4 前后变得充分？}
}
\]

当前数据包括：

- Weight
- Status_Score
- Feeding
- Defecation

而且当前结果显示 DTW Core3 与 DTW All4 的候选平台不同，这提示 Defecation 可能改变 MSOH。

## 4.2 Leave-One-Signal-Out 贡献

定义全信号模型：

\[
E_{all}(k)
\]

去掉某个信号 \(s\)：

\[
E_{-s}(k)
\]

定义信号贡献：

\[
\boxed{
C_s(k)
=
E_{-s}(k)
-
E_{all}(k)
}
\]

解释：

- \(C_s(k)>0\)：去掉信号后误差增加，该信号有正贡献；
- \(C_s(k)\approx0\)：该信号当前贡献有限；
- \(C_s(k)<0\)：加入该信号可能引入噪声或不稳定。

## 4.3 对 MSOH 的贡献

比较：

\[
MSOH_{all}
\]

和：

\[
MSOH_{-s}
\]

定义：

\[
\boxed{
\Delta MSOH_s
=
MSOH_{-s}
-
MSOH_{all}
}
\]

例如：

\[
MSOH_{all}=4
\]

而：

\[
MSOH_{-Defecation}=5
\]

则：

\[
\Delta MSOH_{Defecation}=1
\]

解释：

> Defecation 的引入，使达到充分预测信息所需的观察窗口提前约 1 天。

## 4.4 信号信息获得曲线

对于每一个信号，绘制：

\[
k\rightarrow C_s(k)
\]

形成：

\[
\boxed{
\text{Signal Information Acquisition Profile}
}
\]

它回答：

> 某个信号在什么时候开始真正提供新的预测价值？

## 4.5 从 MSOH 扩展到测量调度优化

未来可进一步从：

\[
\text{Minimum Sufficient Observation Horizon}
\]

发展到：

\[
\boxed{
\text{Minimum Sufficient Monitoring Schedule}
}
\]

即不仅问：

> 最少观察几天？

还问：

> 每一天哪些指标值得测？

## 4.6 当前优先方案

当前样本量较小，优先采用：

\[
\boxed{
\text{Leave-One-Signal-Out}
}
\]

暂不把复杂 feature importance 或 SHAP 作为主方案。

## 4.7 可执行任务

建议新增：

```text
07_signal_contribution.py
```

执行：

1. 全信号分析；
2. 去掉 Weight；
3. 去掉 Status_Score；
4. 去掉 Feeding；
5. 去掉 Defecation；
6. 比较 Error(k)；
7. 比较 Gain(k)；
8. 比较 MSOH；
9. bootstrap 信号贡献。

输出：

```text
results/<dataset>/07_signal_contribution/
├─ table/
│  ├─ leave_one_signal_out_error.csv
│  ├─ signal_contribution_by_window.csv
│  ├─ signal_msoh_shift.csv
│  └─ signal_contribution_bootstrap.csv
└─ figure/
   ├─ signal_contribution_curve.png
   ├─ signal_msoh_shift.png
   └─ signal_information_profile.png
```

---

# 5. 创新四：跨动物实验的 MSOH 泛化验证

## 5.1 为什么这一层非常重要

当前 Dataset 1 即使得到非常漂亮的结果，也只能说明：

> 在这一份动物实验数据中存在可识别的信息饱和窗口。

加入额外 1–2 个纵向动物实验数据集以后，可以检验：

\[
\boxed{
\text{MSOH 框架是否具有跨实验可迁移性？}
}
\]

## 5.2 不要求不同实验得到同一个 Day

例如：

\[
Dataset_A: MSOH=Day4
\]

\[
Dataset_B: MSOH=Day6
\]

\[
Dataset_C: MSOH=Day3
\]

这不代表方法失败。

真正需要验证的是：

\[
\boxed{
\text{同一套 MSOH 判定逻辑能否在不同实验中识别各自的信息饱和点}
}
\]

## 5.3 每个数据集独立生成

对 Dataset \(d\)：

\[
E_d(k)
\]

\[
G_d(k)
\]

\[
MSOH_d
\]

最终形成：

| Dataset | Candidate MSOH | Main Information Jump | Plateau |
|---|---:|---|---|
| Dataset 1 | Day4/5 | Day3→4 | Day4–5 |
| Dataset 2 | 待定 | 待定 | 待定 |
| Dataset 3 | 待定 | 待定 | 待定 |

## 5.4 相对时间标准化

如果不同实验总时长不同：

\[
T_1\neq T_2\neq T_3
\]

则不应只比较绝对 Day。

可定义：

\[
R_d
=
\frac{
t_{MSOH,d}
}{
T_d
}
\]

用于比较信息饱和发生在完整实验周期的相对位置。

## 5.5 固定规则，而不是每个数据集重新调规则

开发阶段确定并锁定：

- \(\tau_E\)
- \(\delta_G\)
- \(\alpha\)
- \(\tau_S\)

然后原样应用到 Dataset 2、Dataset 3。

这样才能回答：

\[
\boxed{
\text{MSOH 判定规则是否真正具有跨实验适用性}
}
\]

## 5.6 Internal-External Validation

当有 3 个数据集后，可进一步考虑：

\[
A+B\rightarrow C
\]

\[
A+C\rightarrow B
\]

\[
B+C\rightarrow A
\]

重点不是简单训练大模型，而是研究：

- MSOH 阈值能否迁移；
- Error–Gain 关系是否稳定；
- 信号贡献是否具有跨实验一致性；
- 哪些实验特征导致 MSOH 提前或延后。

## 5.7 跨数据集的最终研究问题

可进一步研究：

\[
MSOH_d
=
f(
\text{噪声},
\text{轨迹复杂度},
\text{变化速度},
\text{信号数量},
\text{采样频率}
)
\]

从而回答：

> 哪些类型的动物实验更早达到信息饱和？

## 5.8 建议项目结构升级

```text
Minimum_Observation_Window/
├─ 00_project_mainline.md
├─ 00_msoh_innovation_roadmap.md
├─ config/
│  ├─ dataset_01.yaml
│  ├─ dataset_02.yaml
│  └─ dataset_03.yaml
├─ data/
│  ├─ dataset_01/
│  ├─ dataset_02/
│  └─ dataset_03/
├─ results/
│  ├─ dataset_01/
│  ├─ dataset_02/
│  └─ dataset_03/
├─ 01_prepare_data.py
├─ 02_features.py
├─ 03_models_cv.py
├─ 04_analysis_plot.py
├─ 05_robustness.py
├─ 06_msoh_decision.py
├─ 07_signal_contribution.py
└─ 08_cross_dataset_validation.py
```

---

# 6. 四项创新最终如何组成一个完整框架

最终方法链：

\[
X_{1:k}
\]

↓

\[
\text{Future Trajectory Prediction}
\]

↓

\[
E(k)
\]

↓

\[
G(k)=E(k-1)-E(k)
\]

↓

\[
P(G(k)>\delta_G)
\]

↓

\[
\text{Information Saturation}
\]

↓

\[
MSOH
\]

↓

\[
C_s(k),\ \Delta MSOH_s
\]

↓

\[
\text{Cross-dataset Validation}
\]

可以概括成：

> **Trajectory → Prediction → Information Gain → Uncertainty → MSOH → Signal Attribution → External Validation**

---

# 7. 建议的方法命名

## Uncertainty-aware MSOH Framework

中文：

> **不确定性感知的最小充分观察窗口识别框架**

也可使用：

## Predictive Information Saturation–based MSOH

中文：

> **基于预测信息饱和的最小充分观察窗口方法**

当前更推荐前者作为总框架名。

---

# 8. 开发优先级

## P0：当前必须完成

- [x] Dataset 1 数据清洗
- [x] Feature-based Euclidean
- [x] DTW Core3
- [x] DTW All4
- [x] LOOG
- [x] Error(k)
- [x] Gain(k)
- [ ] 修正版 Group bootstrap
- [ ] 确认 05 与 04 Error 完全一致

## P1：核心方法成型

- [ ] 定义 \(\tau_E\)
- [ ] 定义 \(\delta_G\)
- [ ] 定义 \(\alpha\)
- [ ] 定义 \(\tau_S\)
- [ ] 新建 `06_msoh_decision.py`
- [ ] 输出自动 MSOH 判定
- [ ] 做阈值敏感性分析

## P2：信号贡献解释

- [ ] 新建 `07_signal_contribution.py`
- [ ] 全信号 vs leave-one-signal-out
- [ ] 计算 \(C_s(k)\)
- [ ] 计算 \(\Delta MSOH_s\)
- [ ] 对信号贡献做 bootstrap
- [ ] 特别检查 Defecation 对 Day4/Day5 判断的影响

## P3：跨数据集泛化

- [ ] 接入 Dataset 2
- [ ] 接入 Dataset 3
- [ ] 每个数据集统一运行 01–07
- [ ] 固定 MSOH 判据，不后验调参
- [ ] 新建 `08_cross_dataset_validation.py`
- [ ] 比较绝对 MSOH
- [ ] 比较相对 MSOH
- [ ] 比较 Error–Gain 结构
- [ ] 比较信号贡献结构

---

# 9. 推荐实施顺序

> [!todo]
> **现在不建议继续添加大量预测模型。**

```text
Step 1
修正并完成 05 Group-level bootstrap

↓

Step 2
正式定义 uncertainty-aware MSOH 判据

↓

Step 3
建立 06_msoh_decision.py

↓

Step 4
做阈值敏感性分析

↓

Step 5
做 Defecation / 多信号贡献分解

↓

Step 6
接入 Dataset 2、Dataset 3

↓

Step 7
冻结规则并做跨数据集验证

↓

Step 8
形成统一 MSOH Framework
```

---

# 10. 每一项创新最终回答什么问题

| 创新 | 核心问题 | 最终输出 |
|---|---|---|
| MSOH 定义 | 最少需要观察多久？ | \(MSOH\) |
| 不确定性感知 | 我们对“已经够了”有多大把握？ | CI、\(P(G>\delta)\)、稳定性 |
| 多信号贡献 | 为什么这个窗口已经够了？ | \(C_s(k)\)、\(\Delta MSOH_s\) |
| 跨实验验证 | 这个框架能不能迁移到其他实验？ | \(MSOH_d\)、跨数据集稳定性 |

---

# 11. 项目最终可能形成的论文主线

> 动物实验纵向监测通常遵循预设的固定观察周期，但不同时间点所提供的新增信息价值并不相同。本研究提出 Minimum Sufficient Observation Horizon（MSOH）框架，将早期纵向轨迹对固定未来结局的预测性能作为观察信息量的代理，通过预测误差曲线、边际信息增益及其不确定性分布，识别最早达到信息充分且后续边际收益有限的观察窗口。进一步通过多信号贡献分解解释不同生理指标对 MSOH 的影响，并在多个独立动物实验数据集中评估该判定框架的可迁移性。

---

# 12. 项目边界

## 本项目要解决

- 最少观察多久才足够；
- 新增观察时间还提供多少预测信息；
- 什么时候进入信息饱和；
- 哪些信号帮助更早达到信息充分；
- MSOH 是否能跨实验识别。

## 本项目不直接声称

- MSOH 之后可以停止所有动物监测；
- 单个数据集得到的 Day4/Day5 可以推广到所有动物实验；
- 预测误差最低窗口就是 MSOH；
- p>0.05 就代表没有信息；
- 某个预测算法本身是项目的核心创新。

---

# 13. 最终目标

本项目最终希望从：

> “这个实验预测模型的 RMSE 是多少？”

升级为：

\[
\boxed{
\text{这项动物实验需要观察多久，才能获得足够的未来信息？}
}
\]

再进一步升级为：

\[
\boxed{
\text{哪些信号、哪些时间点是真正必要的？}
}
\]

最终形成：

\[
\boxed{
\text{一个可量化、可解释、带不确定性、可跨实验验证的 MSOH 方法框架}
}
\]

---

# 14. 与现有项目文件的关系

- [[00_project_mainline]]
- [[01_prepare_data]]
- [[02_features]]
- [[03_models_cv]]
- [[04_analysis_plot]]
- [[05_robustness]]
- [[06_msoh_decision]]
- [[07_signal_contribution]]
- [[08_cross_dataset_validation]]

> [!note]
> `06–08` 为规划中的下一阶段模块，尚未全部实现。

---

# 15. 当前下一步

- [ ] 完成 `05_robustness_corrected.py`
- [ ] 验证 `Error_Point` 与 04 的 `Core3_NRMSE_Mean` 一致
- [ ] 检查 Day3→4、Day4→5、Day5→6 的 Gain bootstrap 分布
- [ ] 在此基础上正式设计 `06_msoh_decision.py`
- [ ] 暂不新增更多预测算法
