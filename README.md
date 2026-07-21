# CALiBRE-AD: Leakage-Audited Alzheimer Classification

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-02569B)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC4E20)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7B61FF)](https://shap.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Stat%20Plots-4C72B0)](https://seaborn.pydata.org/)

A Kaggle-ready notebook project for Alzheimer disease classification using a leakage-audited, calibrated, and cost-sensitive ensemble pipeline.

## Project Overview

This project benchmarks classical and boosting-based classifiers, then compares them to a stacked calibrated model named CALiBRE-AD.

Key workflow:

- Leakage audit with Full vs Screening-only feature sets
- Stratified cross-validation with out-of-fold predictions
- Cost-sensitive threshold tuning
- Calibration, ROC, PR, confusion matrix, and learning-curve analysis
- SHAP feature attribution
- Automatic export of charts and CSV outputs

## Repository Files

- [calibre-ad-alzheimers-analysis.ipynb](calibre-ad-alzheimers-analysis.ipynb): End-to-end notebook pipeline
- [LICENSE](LICENSE): MIT license file

## Runtime Target

- Kaggle Notebooks (Python 3.12)

The notebook auto-installs missing packages when required:

- xgboost
- lightgbm
- shap

## Dataset Loading Logic

The loader tries the dataset in this order:

1. DATASET_PATH environment variable
2. Known Kaggle input path candidates
3. Any CSV in /kaggle/input with alzheimers in the name
4. A single detected CSV under /kaggle/input or current directory

If multiple CSV files exist and no clear match is found, set DATASET_PATH manually.

## How To Run

1. Open [calibre-ad-alzheimers-analysis.ipynb](calibre-ad-alzheimers-analysis.ipynb) in Kaggle.
2. Attach the Alzheimer dataset from the Add Data panel.
3. Run all cells from top to bottom.
4. Collect outputs from /kaggle/working/calibre_exports.

## Exported Artifacts

### Figure Exports (PNG and PDF)

- class_distribution
- correlation_heatmap
- leakage_audit
- reliability_diagram
- roc_pr_curves
- cost_sensitive_threshold
- confusion_matrix
- learning_curve
- shap_summary
- shap_feature_importance

### CSV Exports

- alzheimers_disease_data_loaded.csv
- diagnosis_class_balance.csv
- correlation_matrix.csv
- target_correlations.csv
- results_full.csv
- results_screening.csv
- leakage_audit_data.csv
- cost_threshold_curve.csv
- learning_curve_data.csv
- confusion_matrix.csv
- oof_probabilities.csv
- shap_importance.csv

## Notes

- The notebook keeps DataFrame feature names in LightGBM evaluation paths to avoid feature-name mismatch warnings.
- All generated files are written to /kaggle/working/calibre_exports.

## License

Released under the MIT License. See [LICENSE](LICENSE).
