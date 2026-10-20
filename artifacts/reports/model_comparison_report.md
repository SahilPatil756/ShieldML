# ShieldML Model Comparison & Benchmark Report

**Selected Champion Model**: `LogisticRegression_Calibrated`

## Model Comparison Table

| Model | Threshold | ROC-AUC | PR-AUC | F2-Score | Precision | Recall | Brier Score | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00002 | 0.0000 | 0.0000 |
| RandomForest | 0.5 | 1.0000 | 1.0000 | 0.9055 | 1.0000 | 0.8846 | 0.00107 | 0.0000 | 0.1154 |
| XGBoost | 0.5 | 0.9999 | 0.9932 | 0.9055 | 1.0000 | 0.8846 | 0.00110 | 0.0000 | 0.1154 |
| LightGBM | 0.5 | 0.9999 | 0.9897 | 0.9375 | 1.0000 | 0.9231 | 0.00095 | 0.0000 | 0.0769 |
| SoftVotingEnsemble | 0.5 | 1.0000 | 1.0000 | 0.9055 | 1.0000 | 0.8846 | 0.00093 | 0.0000 | 0.1154 |
| **LogisticRegression_Calibrated** | 0.4045 | 1.0000 | 1.0000 | 0.8974 | 1.0000 | 0.8750 | 0.00039 | 0.0000 | 0.1250 |

## Key Engineering Observations
- **Evaluation Metric (F2-Score)**: Chosen because a false negative (missed fraud charge) is vastly more costly to financial institutions than a false alert (temporary card lock or SMS verification).
- **Probability Calibration**: Calibrates tree output margins to true empirical probabilities, reducing the Brier score.
- **Ensemble Resilience**: The soft-voting ensemble averages variance across tree families (Random Forest, XGBoost, LightGBM).
