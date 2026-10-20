# ShieldML — Architecture & System Design

## 1. High-Level System Architecture

ShieldML is engineered for high-throughput, sub-50ms financial fraud risk scoring. It integrates leak-free data processing, imbalanced learning, soft-voting ensemble modeling, probability calibration, F2 threshold optimization, and SHAP explainability.

```text
[ Incoming Financial Transaction (HTTP POST /predict) ]
                      │
                      ▼
         [ Pydantic v2 Input Validation ]
                      │
                      ▼
        [ Feature Engineering & Preprocessing ]
       - log1p(Amount), Micro-Charge Indicator
       - Cyclical Periodic Time (sin_hour, cos_hour)
       - RobustScaler (Fitted on Train Set Only)
                      │
                      ▼
          [ Champion Model Scoring ]
       - LogisticRegression / LightGBM / XGBoost
       - Soft-Voting Ensemble Margin Average
                      │
                      ▼
         [ Platt Sigmoid Calibration ]
      - Maps Margins to True Empirical Probabilities
      - Minimizes Brier Score Loss
                      │
                      ▼
         [ Decision Rule (F2 Threshold) ]
      - Fraud Prob >= 0.4045 ──> "FRAUD" (Alert / Step-Up)
      - Fraud Prob <  0.4045 ──> "LEGITIMATE" (Approve)
                      │
                      ▼
        [ SHAP Local Feature Attribution ]
      - Linear / Tree Explainer Top-5 Drivers
                      │
                      ▼
[ JSON Response (< 20ms) + In-Memory Metrics Recorded ]
```

## 2. Core Architectural Principles

### 2.1 Chronological / Temporal Splitting
In fraud detection, standard random k-fold cross validation induces severe future leakage because fraud schemes evolve over time. ShieldML enforces strict chronological splitting:
- **Training Set (70%)**: $T_0 \to T_{\text{train}}$
- **Validation Set (15%)**: $T_{\text{train}} \to T_{\text{val}}$
- **Held-Out Test Set (15%)**: $T_{\text{val}} \to T_{\text{test}}$

All transformations (scalers, encoders, and SMOTE oversamplers) are fit strictly on the Training Set.

### 2.2 Leak-Free Imbalance Handling
Financial fraud rates are typically between 0.1% and 1.0%. Synthetic Minority Over-sampling Technique (SMOTE) is applied **exclusively to the training split**. The validation and test sets remain in their natural, untouched empirical distribution.

### 2.3 Optimization Metric Selection: F2-Score
In payment networks, a **False Negative** (allowing a fraudulent charge to pass) incurs immediate direct financial loss, regulatory fines, and customer attrition. A **False Positive** merely triggers an automated SMS step-up prompt or temporary hold.
The $F_\beta$ score with $\beta = 2.0$:
$$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
weights recall twice as heavily as precision. The optimal threshold is discovered by sweeping candidate thresholds on validation probabilities.

### 2.4 Probability Calibration
Raw tree models and ensembles output margins or skewed probabilities. Using Platt scaling (`CalibratedClassifierCV(method='sigmoid', cv='prefit')`) trained on validation data aligns output values with real-world empirical default probabilities.
