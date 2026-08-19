# PROVA-AD

**A Provenance-Gated Dual-Stream Network with a Differentiable Cost-Calibrated
Decision Head for Leakage-Audited Alzheimer's Disease Classification**

MD. BAKIKIBILLAH · NAIMA NAJAM NEJUM
Department of Computer Science, American International University-Bangladesh

**`PROVA-AD.ipynb` is the entire project**: one self-contained notebook that
defines the architecture from scratch, runs every experiment, and produces
every table and figure below, with all outputs already executed and stored in
`results/` and `figures/`. It has no project imports — only NumPy, pandas,
scikit-learn, SciPy and Matplotlib — so it runs unchanged on Kaggle, Colab, or
a local kernel.

---

## What the model is

Tabular AD benchmarks report 95 %+ accuracy that comes almost entirely from ten
clinician-administered assessment items — instruments produced *by* the
diagnostic work-up the model is supposed to predict. No conventional
architecture can report how much of a given prediction rested on those items,
because a flat feature vector carries no record of where each feature came from.

PROVA-AD makes provenance a first-class architectural object, in three stages:

**Stage 1 — Provenance-Partitioned Stream Encoder.** Three parallel banks of
K = 4 heterogeneous learners (boosted trees, extra trees, RBF-SVM, logistic) are
fitted on the screening stream `x_s ∈ ℝ²²`, the clinical stream `x_c ∈ ℝ¹⁰`, and
their union. The meta-features handed downstream are *provenance-indexed*.

**Stage 2 — Provenance Gate and Cross-Stream Fusion.** A small gate that sees
screening features only emits a per-patient reliance coefficient
`α(x) ∈ (0,1)` and fuses the two stream-evidence logits:

```
ℓ = α·e_c + (1−α)·e_s + vᵀz̃ + b₀
```

`α` is directly readable as *the share of this patient's decision evidence that
came from clinician-administered items* — a per-prediction leakage audit,
available at inference time without labels. A gate-entropy regulariser keeps
routing crisp (measured mean entropy 0.066 nats out of a maximum 0.693).

**Stage 3 — Differentiable Cost-Calibrated Decision Head.** A monotone
piecewise-linear spline `Γ_θ` (non-decreasing by construction via softplus
increments) is trained *jointly* with a proper scoring rule, a soft-binned
calibration penalty, and a smoothed clinical decision cost — replacing post-hoc
isotonic regression. Because the output probabilities are calibrated, the
cost-optimal operating point follows in closed form:

```
τ* = c_FP / (c_FP + c_FN)        (= 1/6 for the 5:1 clinical cost used here)
```

so no labelled threshold-tuning set is needed. All gradients are hand-derived
and verified against central finite differences to `1.6e-10`.

---

## Headline results

| | PROVA-AD | next-best on that metric | strongest classifier (Hist. GB) |
|---|---|---|---|
| Accuracy (Full track, τ=0.5) | 0.950 | — | **0.951** |
| ROC-AUC | 0.951 | — | **0.952** |
| **Expected calibration error** | **0.015** | 0.028 (Logistic Reg.) | 0.034 |
| Clinical cost `5·FN + 1·FP` | **339** | 358 (Gradient Boosting) | 365 |
| Screening track ROC-AUC | 0.511 | 0.530 (Logistic Reg.) | 0.494 |

- Accuracy is a **tie** with the boosting baselines (McNemar p = 0.83 and 0.74);
  significantly better than Random Forest (p = 0.026). We do not claim an
  accuracy improvement.
- Calibration error is the lowest of any model tested: **1.9× below the
  next-best** (logistic regression — a weak classifier, so this is the least
  flattering comparison available), **2.3× below the strongest classifier**, and
  10.8× below Random Forest. This is where the architecture pays off.
- Withholding the ten assessment features collapses **every** model, including
  ours, to chance — the central negative result of the paper.
- Under distribution shift the closed-form τ* lands within **2.6 %** of the
  unavailable target oracle cost, while a threshold grid-searched on the source
  stratum exceeds it by **365 %**.

All eleven baselines, PROVA-AD, and every derived statistic above are recomputed
fresh each time the notebook runs — nothing in this table is hand-entered.

---

## Layout

```
PROVA-AD.ipynb   <- the whole study: architecture, experiments, tables, figures
data/
  data_alzheimers.csv   the 2,149-record cohort
figures/            every figure the notebook produces, vector PDF + PNG
results/            every exported table and JSON summary, as produced by the run
LICENSE
README.md
```

## Notebook map

| § | Produces |
|---|---|
| 0 | seed, versions, global configuration |
| 1 | cohort loading, provenance partition, target correlations |
| 2 | Stage 1 — the three heterogeneous learner banks |
| 3 | Stages 2–3 — the novel head, in NumPy with analytic gradients (finite-difference check) |
| 4 | `ProvaAD` estimator + nested 5×5-fold out-of-fold evaluation |
| 5 | metrics, ECE, bootstrap CIs, exact McNemar, cost curve |
| 6 | eleven reference baseline classifiers under the identical protocol |
| 7–8 | full-track and screening-track result tables, leakage audit |
| 9–10 | PROVA-AD main result + screening-only collapse |
| 11–12 | confidence intervals, McNemar tests, closed-form threshold check |
| 13 | provenance-gate reliance analysis (per-patient α) |
| 14 | ablation: cumulative build-up + leave-one-component-out |
| 15–17 | learning curve, domain shift, permutation attribution |
| 18–19 | model complexity, optional external-cohort validation hook |
| 20 | every figure saved to `figures/` |

---

## Reproducing

Open `PROVA-AD.ipynb` and run all cells — roughly 40 minutes, CPU only. On
Kaggle, attach the dataset `rabieelkharoua/alzheimers-disease-dataset` and the
path is resolved automatically; locally, keep `data_alzheimers.csv` under
`data/` next to the notebook. Every table in `results/` and every figure in
`figures/` is regenerated in place.

### Requirements

Python ≥ 3.10, NumPy, pandas, scikit-learn, SciPy, Matplotlib. No GPU, no
deep-learning framework: the decision head is implemented directly in NumPy
with hand-derived gradients.

---

## Running an external cohort

The notebook's Section IX names the absence of an independently collected
external cohort as its main limitation. A hook is wired up in the
"External cohort validation" cell near the end of the notebook: set
`EXTERNAL_CSV` to the path of a second CSV before running that cell.

```python
EXTERNAL_CSV = "data_external.csv"   # e.g. an OASIS export
```

The external CSV needs a binary `Diagnosis` column and any overlapping subset of
the primary cohort's column names; columns present in only one cohort are
dropped from both so the feature spaces align exactly. Features named in the
notebook's clinical-feature list are routed to the clinical stream, everything
else to the screening stream.

---

## Data

Alzheimer's Disease dataset by R. El Kharoua, hosted on Kaggle
(<https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset>),
2,149 records, CC-BY 4.0. The copy in this repository is the same file used for
every reported result.

**The dataset is synthetic.** The paper's claims are methodological and
comparative; nothing here is a clinical tool or a diagnostic claim.

---

## Licence

Code released under the MIT licence. Dataset under CC-BY 4.0.
