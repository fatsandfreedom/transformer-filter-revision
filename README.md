# Transformer Filter Revision

A research paper revision project for IEEE IVMSP 2026.

[中文版 README](README_cn.md)

## Overview

This project develops and validates a **filtering-theoretic interpretation of Transformers** from a state-space perspective. The central thesis interprets the Transformer attention mechanism as an **adaptive non-parametric estimator**, analogous to the **Wiener filter**, operating in a latent state space.

> Paper title: *"A Filtering-Theoretic Interpretation of Transformers: A State-Space Perspective"*

## Repository Structure

```
transformer_filter_revision/
├── code/
│   ├── sim_main.py                # Main simulation and training script
│   └── analysis/
│       ├── attention_analysis.py  # Attention vs spatial/temporal distance
│       ├── kernel_compare.py      # Dot-product vs uniform kernel comparison
│       ├── knn_metrics.py         # Latent space KNN consistency metric
│       └── run_all_analysis.py    # Run all analyses
├── docs/
│   ├── revision_plan.md           # Revision priorities and plan
│   └── revier_comments.md         # IEEE reviewer comments (3 reviewers)
├── outputs/
│   ├── figs/                      # Generated figures (PNG)
│   └── logs/                      # Metrics (JSON) and cached simulation data
├── paper/
│   ├── original.tex               # Original submitted version
│   ├── revised.tex                # Revised version (in progress)
│   ├── revised.pdf                # Compiled PDF
│   ├── refs.bib                   # Bibliography
│   └── figures/                   # Paper figures
└── template/
    └── 2026IVMSP_paper.tex        # Official IVMSP 2026 template
```

## Paper Principle

The paper proposes interpreting the Transformer's self-attention mechanism through the lens of **optimal filtering theory**.

**Key insight:** At each timestep `t`, the attention-weighted prediction can be written as:

```
θ_t = (X^T W_t X + λI)^{-1} X^T W_t y
```

This is exactly the **Wiener filter** (regularized WLS), where:
- `X` = latent state matrix (keys/values from encoder)
- `W_t = diag(softmax(q_t K^T / √d))` = attention weights as a diagonal weighting matrix
- `λI` = ridge regularization (analogous to noise-to-signal ratio in Wiener filter)
- `q_t` = query vector at time `t`

**Interpretation chain:**
1. The encoder maps raw observations into a latent state space (acoustic fingerprints → 64D vectors)
2. The query `q_t` represents the "desired signal" at time `t`
3. Attention scores `W_t` act as **adaptive kernel weights**, concentrating on the most relevant past states
4. The WLS solution `θ_t` is the minimum-variance linear estimator — the discrete-time Wiener filter

This reframes attention not as a "soft lookup" but as a **data-driven spectral estimator** that adapts its frequency response to the local signal statistics, providing a principled state-space interpretation of why Transformers work for sequential prediction tasks.

## Core Model: ISFOTransformer

The `ISFOTransformer` implements:
- Acoustic fingerprint encoder → 64-dimensional latent space
- Causal attention (upper triangular mask)
- At each timestep `t`, solves a **weighted least squares with ridge regularization**:

  `θ_t = (X^T W_t X + λI)^{-1} X^T W_t y`

  where `W_t` are the attention scores — the precise analogy to the Wiener filter.

## Revision Goals

Based on reviewer feedback (see [docs/revier_comments.md](docs/revier_comments.md)):

1. Soften theoretical language ("prove" → "interpret", "equivalent" → "analogous to")
2. Replace qualitative experiments with quantitative validation metrics
3. Restructure experiments: Prediction → Setup → Results
4. Fix IEEE formatting compliance

## Requirements

```bash
pip install torch pyroomacoustics numpy matplotlib scikit-learn
```

## Usage

```bash
# Run main simulation and training
python code/sim_main.py

# Run all analyses
python code/analysis/run_all_analysis.py
```

## Output

- Figures saved to `outputs/figs/`
- Metrics saved to `outputs/logs/`
