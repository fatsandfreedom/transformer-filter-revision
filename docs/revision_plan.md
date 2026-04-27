# Revision Plan for Transformer Filtering-Theoretic Paper

## 🎯 Overall Objective

Revise the paper to address reviewer concerns while preserving the core idea:

> Transformer as a filtering-theoretic interpretation (NOT strict equivalence)

The revision must:

- Reduce over-strong theoretical claims
- Align derivations with assumptions
- Convert existing experiments into theory-driven validation
- Add minimal quantitative evidence using existing simulation data

---

# 🟥 Priority 1: Theoretical Alignment (MANDATORY)

## Problem

Reviewers指出：

- claims stronger than derivations
- approximation not justified
- misuse of "prove / equivalent"

---

## Required Actions

### 1.1 Language Correction

Replace globally:

- "prove" → "interpret / characterize"
- "equivalent" → "analogous to / can be viewed as"
- "recover" → "relates to / approximates"

---

### 1.2 Add Assumptions Section

Create new subsection:

"Assumptions Underlying the Interpretation"

Include:

- latent space smoothness
- local linearity
- norm stabilization
- conditional zero-mean noise

---

### 1.3 Fix Theoretical Consistency

Ensure:

- Eq.(9) → MAP / regularized WLS (NOT pure MLE)
- Softmax → derived from MaxEnt (NOT MLE)

---

### 1.4 Theorem Adjustment

- downgrade to "interpretation theorem"
- remove strict equivalence claims
- emphasize "induces kernel weights"

---

### 1.5 Multi-head Clarification

Add short explanation:

multi-head = mixture of local estimators / subspace filters

---

# 🟧 Priority 2: Experiment → Theory Validation (CORE)

## Problem

Current experiments are:

❌ qualitative visualization  
✅ but not tied to theory  

---

## Required Transformation

ALL experiments must follow:

1. Theoretical Prediction  
2. Experimental Setup  
3. Quantitative Evidence  

---

# 🟨 Task 2.1: Attention Behavior Validation

## Theoretical Prediction

If attention is state-driven:

- weights correlate with spatial similarity
- NOT with temporal distance

---

## Implementation

Add code to compute:

- top-k attention indices
- spatial distance (z-space)
- temporal distance

---

## Output

- numerical averages
- scatter plot

---

## Expected Result

spatial distance < temporal distance (statistically)

---

# 🟩 Task 2.2: Latent Space Geometry Validation

## Theoretical Prediction

Latent space preserves local structure

---

## Implementation

Add:

- KNN consistency metric
- compare raw fingerprint PCA vs embedding PCA

---

## Output

- KNN consistency score

---

# 🟦 Task 2.3: Kernel Interpretation (Optional, lightweight)

## Theoretical Prediction

attention ≈ kernel weighting

---

## Implementation (minimal)

Compare:

- dot-product attention
- uniform weights

---

## Output

- training loss comparison (reuse existing loop)

---

# 🟪 Priority 3: Rewrite Experimental Section

## Problem

Current writing:

- descriptive
- not theory-driven

---

## Required Structure

Each subsection must include:

### 1. Theoretical Prediction

Based on proposed framework...

### 2. Experimental Setup

We construct...

### 3. Quantitative Results

Results show...

---

# 🟫 Priority 4: Formatting and Cleanup

- remove duplicated paragraphs in Introduction
- align with IVMSP template
- ensure figure captions include interpretation context

---

# 🧩 Constraints

- NO new datasets
- NO new model architectures
- NO large-scale experiments
- ONLY extend existing simulation

---

# ✅ Success Criteria

The revision is successful if:

- theoretical claims are no longer overstated
- experiments directly support theory
- at least 2 quantitative metrics are added
- reviewers' concerns are explicitly addressed

---

# 📌 Execution Strategy for Codex

When modifying:

- prefer incremental edits over full rewrites
- always preserve original intent
- only change necessary parts
- output LaTeX-ready text