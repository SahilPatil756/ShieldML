# ShieldML Champion Model Card

## 1. Model Details
- **Model Name**: ShieldMLChampion
- **Model Version**: 1.0.0
- **Model Type**: Calibrated Classifier with Robust Scaling & Engineered Temporal/Amount Signals
- **Architecture**: Ensembles of Gradient Boosted Decision Trees (XGBoost, LightGBM, Random Forest) & Calibrated Margin Classifiers
- **Pipeline Release Date**: October 2026
- **License**: MIT

## 2. Intended Use
- **Primary Use**: Real-time fraud scoring for online retail card-not-present (CNP) and electronic funds transfer transactions.
- **Intended Users**: Risk engineering teams, payment processors, financial anti-fraud operations.
- **Out-of-Scope Uses**: Automated adverse credit decisions (e.g. loan approvals, insurance underwriting), identity verification without supplementary KYC.

## 3. Training & Validation Data
- **Dataset**: Credit Card Transaction benchmark with PCA-transformed signals (`V1` to `V28`), `Amount`, and `Time`.
- **Prevalence**: Heavy class imbalance (~0.80% fraud prevalence).
- **Split Strategy**: Chronological 70% Train, 15% Validation, 15% Test.
- **Oversampling**: SMOTE applied strictly to training fold to achieve 10% minority representation.

## 4. Evaluation Metrics & Benchmark Performance
Evaluated on the held-out, untouched test fold ($N = 2,250$):

| Metric | Measured Score | Production Target | Status |
|---|---|---|---|
| **ROC-AUC** | **1.0000** | $\ge 0.9000$ | **EXCEEDED** |
| **PR-AUC (Average Precision)** | **1.0000** | $\ge 0.8000$ | **EXCEEDED** |
| **F2-Score** | **0.8974** | $\ge 0.8500$ | **EXCEEDED** |
| **Precision** | **1.0000** | $\ge 0.8000$ | **EXCEEDED** |
| **Recall** | **0.8750** | $\ge 0.8500$ | **EXCEEDED** |
| **Brier Score** | **0.00039** | $\le 0.0100$ | **EXCEEDED** |
| **Optimal Decision Threshold** | **0.4045** | N/A (F2 Maximized) | **OPTIMIZED** |
| **Single-Request Latency (p95)**| **22.53 ms** | $< 50.00$ ms | **PASSED** |

## 5. Explainability & Fair Lending Safeguards
- **Local SHAP Attributions**: Every inference can return the top-5 features that shifted the probability towards or away from fraud.
- **Sensitive Attributes**: No protected demographic classes (gender, ethnicity, religion, marital status) are utilized. Features consist strictly of anonymized PCA signal vectors, transactional amount, and temporal frequencies.

## 6. Caveats and Drift Monitoring
- Transaction volume and fraud signatures drift over time.
- The model must be retrained on a rolling weekly cadence or when Population Stability Index (PSI) exceeds 0.20 on key PCA features (`V14`, `V12`, `V4`).
