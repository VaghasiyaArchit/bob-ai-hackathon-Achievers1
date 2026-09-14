# Machine Learning Subsystem: Transformer Predictive Maintenance & Explainable AI (XAI)

This subsystem contains the data preprocessing routines, trained models, batch inference pipelines, and Explainable AI (XAI) components for electrical transformer health monitoring.

---

## 🔬 Subsystem Components

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Ingests raw time-series CSVs from `data/raw/data_train/` (2,100 files) and `data/raw/data_test/` (900 files).
- Extracts 82 domain-engineered features per transformer (per-gas trajectory statistics, slopes, TDCG accumulation, and IEEE/IEC Duval diagnostic ratios).
- Output files in `data/processed/`: `train_features.csv`, `test_features.csv`, `train_fdd.csv`, `test_fdd.csv`, `train_rul.csv`, `test_rul.csv`.

### 2. Fault Detection & Diagnosis Model (`src/train_fdd.py` -> `models/fdd_model.pkl`)
- **Algorithm**: `HistGradientBoostingClassifier(class_weight='balanced')`
- **Classes**: 
  - `1`: Normal Baseline Degradation / Cellulose Aging (81.2%)
  - `2`: Low-Energy Partial Discharge / Corona (4.2%)
  - `3`: High-Temperature Thermal Oil Breakdown $>700^\circ\text{C}$ (5.4%)
  - `4`: High-Energy Electrical Arcing / Flashover (9.2%)
- **Test Performance**: **96.78% Accuracy**, **94.04% Balanced Accuracy**, **0.9227 Macro F1**.

### 3. Remaining Useful Life Regressor (`src/train_rul.py` -> `models/rul_model.pkl`)
- **Algorithm**: `HistGradientBoostingRegressor`
- **Target Range**: 362 to 1,093 cycles/steps.
- **Test Performance**: **MAE = 74.57 cycles**, **$R^2$ = 0.8093**, **85.89% within $\pm 150$ cycles**.

### 4. Unified Transformer Risk Engine with XAI (`src/inference.py`, `src/run_inference.py`)
- Combines:
  - FDD Fault Risk ($40\%$)
  - RUL Degradation Risk ($35\%$)
  - Statistical Anomaly Score ($25\%$)
- Maps final score to risk categories (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- Extracts **`top_3_risk_factors`** and generates localized deterministic explanations using physical DGA evidence.
- Batch output: `outputs/test_risk_predictions.csv`.

---

## 📊 Key Execution Commands

```bash
# 1. Run Feature Extraction on Real Data
python src/data_preprocessing.py

# 2. Train and Evaluate Models
python src/train_fdd.py
python src/train_rul.py
python src/evaluate_models.py

# 3. Execute Batch Inference & Generate XAI Predictions
python src/run_inference.py
```
