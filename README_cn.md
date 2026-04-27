# Transformer 滤波器解释 — 论文修订项目

IEEE IVMSP 2026 论文修订项目。

[English README](README.md)

## 项目概述

本项目从**状态空间视角**出发，对 Transformer 的注意力机制进行**最优滤波理论**解释，并提供实验验证。

> 论文标题：*"A Filtering-Theoretic Interpretation of Transformers: A State-Space Perspective"*

## 论文核心原理

本文提出将 Transformer 自注意力机制重新解释为**自适应非参数估计器**，其数学形式与**维纳滤波器（Wiener Filter）**完全对应。

**核心公式：** 在每个时间步 `t`，注意力加权预测可以写成：

```
θ_t = (X^T W_t X + λI)^{-1} X^T W_t y
```

这正是**带岭正则化的加权最小二乘（WLS）**，即离散时间维纳滤波器，其中：

| 符号 | Transformer 含义 | 滤波器含义 |
|------|-----------------|-----------|
| `X` | Key/Value 矩阵（潜在状态） | 观测矩阵 |
| `W_t = diag(softmax(q_t K^T / √d))` | 注意力权重 | 自适应核权重 |
| `λI` | 岭正则化项 | 噪声-信号比 |
| `q_t` | Query 向量 | 期望信号 |
| `θ_t` | 输出预测 | 最小方差线性估计 |

**解释链：**
1. 编码器将原始观测映射到潜在状态空间（声学指纹 → 64 维向量）
2. Query `q_t` 代表时刻 `t` 的"期望信号"
3. 注意力分数 `W_t` 作为**自适应核权重**，聚焦于最相关的历史状态
4. WLS 解 `θ_t` 是最小方差线性估计量，即维纳滤波器

这将注意力机制从"软查找"重新定义为**数据驱动的谱估计器**，为 Transformer 在序列预测任务中的有效性提供了原理性的状态空间解释。

## 目录结构

```
transformer_filter_revision/
├── code/
│   ├── sim_main.py                # 主仿真与训练脚本
│   └── analysis/
│       ├── attention_analysis.py  # 注意力 vs 时空距离分析
│       ├── kernel_compare.py      # 点积核 vs 均匀核对比
│       ├── knn_metrics.py         # 潜在空间 KNN 一致性指标
│       └── run_all_analysis.py    # 运行所有分析
├── docs/
│   ├── revision_plan.md           # 修订计划与优先级
│   └── revier_comments.md         # IEEE 审稿人意见（3位）
├── outputs/
│   ├── figs/                      # 生成的图表（PNG）
│   └── logs/                      # 指标数据（JSON）
├── paper/
│   ├── original.tex               # 原始投稿版本
│   ├── revised.tex                # 修订版本（进行中）
│   ├── revised.pdf                # 编译后的 PDF
│   └── refs.bib                   # 参考文献
└── template/
    └── 2026IVMSP_paper.tex        # IVMSP 2026 官方模板
```

## 修订目标

基于审稿人反馈（见 [docs/revier_comments.md](docs/revier_comments.md)）：

1. 软化理论表述（"prove" → "interpret"，"equivalent" → "analogous to"）
2. 用定量验证指标替代定性实验
3. 重构实验结构：预测 → 设置 → 结果
4. 修复 IEEE 格式合规问题

## 环境依赖

```bash
pip install torch pyroomacoustics numpy matplotlib scikit-learn
```

## 使用方法

```bash
# 运行主仿真与训练
python code/sim_main.py

# 运行所有分析
python code/analysis/run_all_analysis.py
```

## 输出

- 图表保存至 `outputs/figs/`
- 指标数据保存至 `outputs/logs/`
