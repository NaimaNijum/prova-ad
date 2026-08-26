# PROVA-AD

**Provenance-gated dual-stream classification with a differentiable,
cost-calibrated decision head for Alzheimer's disease benchmarks.**

MD. Bakibillah Rahat · Naima Najam Nejum
Department of Computer Science, American International University-Bangladesh

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

This repository is organized around one executable artifact, [`prova-ad.ipynb`](prova-ad.ipynb). The notebook contains the model implementation, evaluation protocol, baselines, statistical analysis, ablations, robustness checks, and figure generation. The `results/` and `figures/` directories contain exported outputs from the reported run.

## Study at a glance

The dataset contains 2,149 records and 32 predictors. Ten clinical assessment variables are separated from 22 pre-assessment screening variables so that the model can measure how much its predictions depend on information produced by the diagnostic work-up itself.

The notebook evaluates two tracks:

- **Full track:** all 32 predictors are available.
- **Screening track:** only the 22 screening predictors are available.

The dataset is synthetic. Results are methodological and comparative; this project is not a clinical diagnostic tool.

## Architecture

### 1. Provenance-partitioned encoder

Three banks of four heterogeneous scikit-learn learners are trained on:

1. the screening stream,
2. the clinical stream, and
3. the union of both streams.

Their out-of-fold predictions become provenance-indexed meta-features for the custom head.

### 2. Provenance gate and fusion

A 16-unit NumPy gate receives screening features and stream-level evidence summaries. It outputs a patient-specific reliance value `alpha` and fuses clinical and screening evidence:

```text
ell = alpha * e_c + (1 - alpha) * e_s + v.T @ z_tilde + b_0
```

The gate does not receive raw clinical features, preserving the interpretation of `alpha` as a provenance-reliance signal. A gate-entropy penalty encourages decisive routing.

### 3. Differentiable decision head

A 16-knot monotone spline calibrates the fused logit. It is trained in NumPy with a proper scoring term, soft calibration penalty, smoothed decision-cost term, entropy regularization, and L2 regularization. The analytic gradients are checked against central finite differences.

For false-negative cost `c_FN = 5` and false-positive cost `c_FP = 1`, the calibrated operating point is:

```text
tau* = c_FP / (c_FP + c_FN) = 1/6
```

## Current results

All values below are from the committed CSV/JSON exports in `results/`.

| Metric | PROVA-AD | Best or reference result |
|---|---:|---:|
| Full-track accuracy at `tau = 0.5` | 0.9507 | 0.9511, Hist. Gradient Boosting |
| Full-track ROC-AUC | 0.9507 | 0.9525, Hist. Gradient Boosting |
| Full-track ECE | **0.0149** | 0.0343, Hist. Gradient Boosting |
| Full-track cost, `5*FN + FP` | **338** | 358, Gradient Boosting |
| Screening-track ROC-AUC | 0.5107 | 0.5300, Logistic Regression |

The screening track is near chance for every model, showing that the high full-track performance is driven by the ten clinical assessment variables. PROVA-AD's full-track accuracy is statistically comparable to Hist. Gradient Boosting (McNemar `p = 1.0`); the project does not claim an accuracy improvement.

Additional measured facts:

- 95% bootstrap CI for PROVA-AD accuracy: `0.9409` to `0.9595`.
- 95% bootstrap CI for ROC-AUC: `0.9382` to `0.9626`.
- Mean gate output: `alpha = 0.6828`; mean gate entropy: `0.0652` nats.
- 462 trainable head parameters; 0.422 ms per record during the measured CPU inference run.
- The empirical source-cohort cost minimum is 338 at threshold 0.385. The closed-form `tau*` cost is 359, 6.2% higher on that cohort.

## Repository layout

```text
prova-ad/
├── prova-ad.ipynb                 # complete implementation and experiments
├── data/
│   └── data_alzheimers.csv        # 2,149-record input cohort
├── figures/                       # 14 figures, each as PDF and PNG
├── results/                       # exported tables and JSON summaries
├── LICENSE                        # repository code licence
└── README.md
```

### Important result files

| File | Contents |
|---|---|
| [`table_full_track.csv`](results/table_full_track.csv) | Full-track metrics for PROVA-AD and 11 baselines |
| [`table_screening_track.csv`](results/table_screening_track.csv) | Screening-only metrics |
| [`prova_operating_points.csv`](results/prova_operating_points.csv) | PROVA-AD metrics at `0.5` and `tau*` |
| [`leakage_audit.csv`](results/leakage_audit.csv) | Full-versus-screening AUC decrement for every model |
| [`gate_analysis.json`](results/gate_analysis.json) | Per-patient gate reliance summary |
| [`threshold_analysis.json`](results/threshold_analysis.json) | Empirical and closed-form threshold comparison |
| [`domain_shift.csv`](results/domain_shift.csv) | Age, education, and BMI source-to-target shifts |
| [`complexity.json`](results/complexity.json) | Parameter count and timing measurements |

Figures include architecture schematics, class distribution, feature correlations, leakage, calibration, cost curves, ROC/PR curves, confusion matrix, gate reliance, permutation importance, learning curves, ablations, and domain-shift analysis.

## Reproduce the results

Requirements: Python 3.10 or newer, Jupyter, NumPy, pandas, SciPy, scikit-learn, and Matplotlib. No GPU or deep-learning framework is required.

1. Open [`prova-ad.ipynb`](prova-ad.ipynb) in Jupyter or VS Code.
2. Run the cells from top to bottom. The configured seed is `42`; the evaluation uses nested 5-fold cross-validation and 2,000 bootstrap resamples.
3. Confirm that `data/data_alzheimers.csv` is available beside the notebook.

The notebook resolves the project root automatically, creates `results/` and `figures/` when needed, and overwrites exported artifacts during a fresh run. The full run is CPU-only and takes approximately 40 minutes on the reference setup.

## Feature partition

The clinical stream is:

```text
MMSE, FunctionalAssessment, ADL, MemoryComplaints,
BehavioralProblems, Confusion, Disorientation, PersonalityChanges,
DifficultyCompletingTasks, Forgetfulness
```

The screening stream contains the remaining 22 predictors, including age, demographics, lifestyle, family history, comorbidities, blood pressure, and cholesterol measurements. `PatientID` and `DoctorInCharge` are excluded, and `Diagnosis` is the binary target.

## External cohort hook

The notebook includes an optional external-validation cell. Set `EXTERNAL_CSV` to a second CSV with a binary `Diagnosis` column before running that cell. Shared columns are aligned automatically; clinical-feature names use the same partition above, and non-overlapping columns are dropped.

Independent external validation is not included in the current results and remains a primary limitation.

## Data and licence

The cohort is the Alzheimer's Disease dataset by R. El Kharoua, hosted on [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset), licensed under CC-BY 4.0. Repository code is released under the MIT licence; see [`LICENSE`](LICENSE).
