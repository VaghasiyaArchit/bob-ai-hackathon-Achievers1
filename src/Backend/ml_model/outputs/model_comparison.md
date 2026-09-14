# Machine Learning Models Unified Comparison & Evaluation

**Project**: IBM Hackathon - Power Grid AI  
**Pipeline**: Real Data Machine Learning Pipeline (TASK 3)  
**Dataset**: Real Kaggle Transformer DGA Telemetry (2,100 Train / 900 Test)  

---

## 1. System Architecture & Model Selection

Two distinct supervised learning engines were engineered, validated via 5-fold cross-validation, and evaluated on the independent 900-sample test set.

| Model Subsystem | Problem Formulation | Selected Algorithm | Cross-Validation Score (Train) | Test Score (Test Set) |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: Fault Detection & Diagnosis (FDD)** | 4-Class Classification | `HistGradientBoostingClassifier` (balanced weights) | **Macro F1 = 0.9272** | **Macro F1 = 0.9227** (Acc: 96.78%) |
| **Model 2: Remaining Useful Life (RUL)** | Continuous Regression | `HistGradientBoostingRegressor` | **MAE = 74.76 cycles** | **MAE = 74.57 cycles** ($R^2$: 0.8093) |

---

## 2. Model 1: Fault Detection & Diagnosis (FDD) Summary

- **Class Balance Treatment**: Balanced sample weighting applied to counteract severe class imbalance (81.2% normal vs 4.2% partial discharge).
- **Test Accuracy**: **96.78%**
- **Balanced Accuracy**: **94.04%**
- **Macro F1**: **0.9227**
- **Weighted F1**: **0.9680**

### Confusion Matrix on Test Set (900 Transformers):
```
                Pred Cat 1   Pred Cat 2   Pred Cat 3   Pred Cat 4
Actual Cat 1      717          0            5            9     
Actual Cat 2      0            36           0            2     
Actual Cat 3      1            0            48           0     
Actual Cat 4      6            4            2            70    
```

---

## 3. Model 2: Remaining Useful Life (RUL) Summary

- **Target Range**: 362 to 1,093 cycles/steps.
- **Mean Absolute Error (MAE)**: **74.57 steps**
- **Root Mean Squared Error (RMSE)**: **104.13**
- **Coefficient of Determination ($R^2$)**: **0.8093**
- **Tolerance Window Accuracy**:
  - Within $\pm 50$ steps: **50.78%**
  - Within $\pm 100$ steps: **72.78%**
  - Within $\pm 150$ steps: **85.89%**

---

## 4. Top Shared Predictive Features Across Models

1. **`C2H2_final` / `C2H2_recent_10pct_mean`**: Acetylene concentration dominates both acute arcing fault diagnosis and lifetime horizon degradation.
2. **`CO_recent_10pct_mean` / `CO_final`**: Carbon Monoxide accumulation measures cumulative solid insulation decomposition.
3. **`ratio_C2H2_C2H4` / `C2H4_hydrocarbon_ratio`**: Key Duval diagnostic ratios differentiating thermal hot-spots from electrical flashover.
4. **`TDCG_slope` / `TDCG_final`**: Total combustible gas generation rate indicates asset degradation velocity.

---

## 5. Artifact Verification & Persistence

- `ml_model/models/fdd_model.pkl` (Serialized bundle: model, feature names, metrics, hyperparameters).
- `ml_model/models/rul_model.pkl` (Serialized bundle: model, feature names, metrics, hyperparameters).
- Both models successfully loaded and validated on independent 900-sample test matrices with zero leakage.
