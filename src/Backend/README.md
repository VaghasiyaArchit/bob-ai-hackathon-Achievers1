# Power Grid AI: Intelligent Predictive Maintenance & Risk Engine

**Power Grid AI** is a production-grade machine learning system engineered for high-voltage power transformer health monitoring, fault diagnosis, remaining useful life estimation, and operational risk triage.

The system is built and evaluated strictly on the **real Kaggle transformer Dissolved Gas Analysis (DGA) dataset** (3,000 transformers $\times$ 420 time steps = 1,260,000 sensor observations).

---

## 📌 Architecture & Subsystems

```
       +-------------------------------------------------------------+
       |         Real Substation DGA Telemetry (420 Time Steps)      |
       |             [H2, CO, C2H4, C2H2] in Dielectric Oil          |
       +------------------------------+------------------------------+
                                      |
                                      v
                       +------------------------------+
                       |    data_preprocessing.py     |
                       | Extract 82 Domain Features   |
                       +--------------+---------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
      +---------------------------+       +---------------------------+
      |        train_fdd.py       |       |        train_rul.py       |
      |   HistGradientBoosting    |       |   HistGradientBoosting    |
      | 4-Class Fault Classifier  |       | Continuous RUL Regressor  |
      | (Acc: 96.8%, F1-M: 0.923) |       | (MAE: 74.6, R²: 0.809)    |
      +-------------+-------------+       +-------------+-------------+
                    |                                   |
                    +-----------------+-----------------+
                                      |
                                      v
                       +------------------------------+
                       |         inference.py         |
                       |    TransformerRiskEngine     |
                       | 40% FDD + 35% RUL + 25% Anom |
                       |  Top 3 Risk Factors + XAI    |
                       +--------------+---------------+
                                      |
                                      v
                       +------------------------------+
                       |      run_inference.py        |
                       | Batch Inference (900 Units)  |
                       | LOW / MEDIUM / HIGH / CRIT   |
                       | Actionable Dispatch & Alert  |
                       +------------------------------+
```

---

## 📂 Repository Structure

```
power-grid-ai/
│
├── ml_model/
│   │
│   ├── data/
│   │   ├── raw/
│   │   │   ├── data_train/                    # 2,100 transformer DGA CSV files (420 rows each)
│   │   │   ├── data_test/                     # 900 transformer DGA CSV files (420 rows each)
│   │   │   ├── labels_fdd_train.csv           # Ground truth FDD labels (Train)
│   │   │   ├── labels_fdd_test.csv            # Ground truth FDD labels (Test)
│   │   │   ├── labels_rul_train.csv           # Ground truth RUL labels (Train)
│   │   │   └── labels_rul_test.csv            # Ground truth RUL labels (Test)
│   │   │
│   │   └── processed/
│   │       ├── train_features.csv             # Master train features (2,100 x 85)
│   │       ├── test_features.csv              # Master test features (900 x 85)
│   │       ├── train_fdd.csv                  # FDD-segregated train set (no RUL leakage)
│   │       ├── test_fdd.csv                   # FDD-segregated test set
│   │       ├── train_rul.csv                  # RUL-segregated train set (no FDD leakage)
│   │       └── test_rul.csv                   # RUL-segregated test set
│   │
│   ├── models/
│   │   ├── scaler.pkl                         # Fitted training scaler bundle
│   │   ├── fdd_model.pkl                      # Trained FDD classifier bundle
│   │   └── rul_model.pkl                      # Trained RUL regressor bundle
│   │
│   ├── api/
│   │   ├── main.py                            # Production FastAPI microservice
│   │   ├── test_api.py                        # Automated API QA test suite (10 tests)
│   │   ├── example_client.py                  # Integration client demo
│   │   └── API_README.md                      # Developer integration & Swagger guide
│   │
│   ├── src/
│   │   ├── data_preprocessing.py              # Real-data feature extraction (82 features)
│   │   ├── model_utils.py                     # Data verification, leakage checks & metrics
│   │   ├── train_fdd.py                       # FDD cross-validation & model training
│   │   ├── train_rul.py                       # RUL cross-validation & model training
│   │   ├── evaluate_models.py                 # Unified model testing & verification
│   │   ├── inference.py                       # TransformerRiskEngine & XAI generator
│   │   └── run_inference.py                   # 900-sample test batch runner & validation
│   │
│   ├── outputs/
│   │   ├── dataset_audit.md                   # 20-point raw data audit report
│   │   ├── dataset_schema.json                # Machine-readable schema catalog
│   │   ├── preprocessing_report.md            # Feature engineering documentation
│   │   ├── fdd_evaluation.md                  # FDD classification metrics & confusion matrix
│   │   ├── rul_evaluation.md                  # RUL regression metrics & tolerance bands
│   │   ├── model_comparison.md                # Unified cross-model comparison report
│   │   ├── fdd_feature_importance.csv         # Top FDD features ranked by permutation importance
│   │   ├── rul_feature_importance.csv         # Top RUL features ranked by permutation importance
│   │   ├── fdd_feature_importance.png         # FDD feature importance bar chart
│   │   ├── rul_feature_importance.png         # RUL feature importance bar chart
│   │   ├── xai_report.md                      # Comprehensive Explainable AI report
│   │   └── test_risk_predictions.csv          # Final test risk predictions with top 3 factors & XAI
│   │
│   ├── requirements.txt                       # Project dependencies
│   └── README.md                              # Subsystem documentation
│
└── README.md                                  # Main system documentation
```

---

## ⚡ Execution Commands & Workflows

### 1. Preprocess the Real Dataset (82 Domain Features)
```bash
python ml_model/src/data_preprocessing.py
```
*Validates 100% ID matching, extracts 82 DGA trajectory features, and creates target-segregated datasets in `data/processed/`.*

### 2. Train the Machine Learning Models
```bash
# Train FDD Fault Classification Model (HistGradientBoosting with class weighting)
python ml_model/src/train_fdd.py

# Train RUL Lifetime Regression Model (HistGradientBoosting)
python ml_model/src/train_rul.py

# Run Unified Evaluation & Verification
python ml_model/src/evaluate_models.py
```

### 3. Generate Feature Importance & XAI Visualizations
```bash
python -c "
import os, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

fdd_b = joblib.load('ml_model/models/fdd_model.pkl')
rul_b = joblib.load('ml_model/models/rul_model.pkl')
test_df = pd.read_csv('ml_model/data/processed/test_features.csv')
features = fdd_b['feature_names']
X_test = test_df[features]

fdd_perm = permutation_importance(fdd_b['model'], X_test, test_df['target_fdd'], n_repeats=10, random_state=42, n_jobs=-1)
rul_perm = permutation_importance(rul_b['model'], X_test, test_df['target_rul'], n_repeats=10, random_state=42, n_jobs=-1)

df_fdd = pd.DataFrame({'feature': features, 'importance': fdd_perm.importances_mean}).sort_values(by='importance', ascending=False).reset_index(drop=True)
df_fdd['rank'] = range(1, len(df_fdd) + 1)
df_fdd.to_csv('ml_model/outputs/fdd_feature_importance.csv', index=False)

df_rul = pd.DataFrame({'feature': features, 'importance': rul_perm.importances_mean}).sort_values(by='importance', ascending=False).reset_index(drop=True)
df_rul['rank'] = range(1, len(df_rul) + 1)
df_rul.to_csv('ml_model/outputs/rul_feature_importance.csv', index=False)
"
```
*Outputs: `fdd_feature_importance.csv`, `rul_feature_importance.csv`, `fdd_feature_importance.png`, and `rul_feature_importance.png`.*

### 4. Run Batch Inference & Generate XAI Predictions
```bash
python ml_model/src/run_inference.py
```
*Executes the `TransformerRiskEngine` across all 900 test transformers, validates 0 NaNs/Infs, computes `top_3_risk_factors` and localized explanations, and saves [test_risk_predictions.csv](file:///c:/power-grid-ai/ml_model/outputs/test_risk_predictions.csv).*

### 5. Inspect the Comprehensive XAI Report
View [xai_report.md](file:///c:/power-grid-ai/ml_model/outputs/xai_report.md) for full case studies of CRITICAL, HIGH, MEDIUM, and LOW risk transformers, global importance rankings, and operational decision bounds.

### 6. Launch the Real-Time Prediction API Microservice
```bash
# Start FastAPI Server (Swagger available at http://127.0.0.1:8000/docs)
python -m uvicorn ml_model.api.main:app --host 127.0.0.1 --port 8000 --reload

# Run Automated 10-Point API Test Suite
python ml_model/api/test_api.py

# Run Example Backend Integration Client
python ml_model/api/example_client.py
```
*API documentation and schemas are detailed in [API_README.md](file:///c:/power-grid-ai/ml_model/api/API_README.md).*

---

## ⚠️ Important Boundary & Environmental Note
The model operates **exclusively on internal Dissolved Gas Analysis (DGA) sensor measurements** (`H2`, `CO`, `C2H4`, `C2H2`). Weather variables (ambient temperature, humidity, wind, lightning strikes) are **NOT currently included** because they are absent from the Kaggle dataset. Zero synthetic data or fake weather variables were fabricated.
