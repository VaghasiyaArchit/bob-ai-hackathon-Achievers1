# Fault Detection & Diagnosis (FDD) - Model Evaluation Report

**Project**: IBM Hackathon - Power Grid AI  
**Model Task**: Multi-Class Fault Classification (`target_fdd` $\in \{1, 2, 3, 4\}$)  
**Selected Algorithm**: `HistGradientBoosting` (with balanced class weighting)  
**Trained Artifact**: `ml_model/models/fdd_model.pkl`  

---

## 1. Cross-Validation Model Comparison (5-Fold Stratified CV)

Cross-validation was performed exclusively on the 2,100 training transformers to select the optimal architecture:

| Candidate Algorithm | Macro F1 (Mean ± Std) | Balanced Accuracy | Overall Accuracy | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| **`HistGradientBoosting`** | **0.9272 ± 0.0149** | **0.9276** | **0.9695** | **0.9695** |
| `RandomForest` | 0.8943 ± 0.0117 | 0.9193 | 0.9571 | 0.9583 |
| `ExtraTrees` | 0.8850 ± 0.0199 | 0.8636 | 0.9557 | 0.9538 |

---

## 2. Final Test Performance (900 Untouched Transformers)

Evaluated on the independent test split:

- **Accuracy**: **96.78%**
- **Balanced Accuracy**: **94.04%**
- **Macro Precision**: **0.9068**
- **Macro Recall**: **0.9404**
- **Macro F1-Score**: **0.9227**
- **Weighted F1-Score**: **0.9680**

### Per-Class Diagnostic Performance:
| Category | Physical Diagnosis | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Normal / Cellulose Aging | 0.9903 | 0.9808 | 0.9856 | 731.0 |
| **2** | Low-Energy Partial Discharge (PD) | 0.9000 | 0.9474 | 0.9231 | 38.0 |
| **3** | High-Temp Thermal Oil Fault | 0.8727 | 0.9796 | 0.9231 | 49.0 |
| **4** | High-Energy Arcing / Mixed | 0.8642 | 0.8537 | 0.8589 | 82.0 |

---

## 3. Confusion Matrix (Test Set)

```
                Predicted Cat 1   Predicted Cat 2   Predicted Cat 3   Predicted Cat 4
Actual Cat 1        717               0                 5                 9     
Actual Cat 2        0                 36                0                 2     
Actual Cat 3        1                 0                 48                0     
Actual Cat 4        6                 4                 2                 70    
```

---

## 4. Top 15 Most Predictive Features

| Rank | Feature Name | Importance Metric |
| :--- | :--- | :--- |
| 1 | `C2H4_TDCG_ratio` | 0.037778 |
| 2 | `H2_TDCG_ratio` | 0.035556 |
| 3 | `CO_TDCG_ratio` | 0.020889 |
| 4 | `ratio_H2_C2H4` | 0.018000 |
| 5 | `CO_recent_50pct_mean` | 0.005111 |
| 6 | `CO_cv` | 0.002889 |
| 7 | `CO_recent_20pct_mean` | 0.001778 |
| 8 | `H2_std` | 0.000889 |
| 9 | `ratio_CO_H2` | 0.000667 |
| 10 | `H2_min` | 0.000667 |
| 11 | `H2_recent_20pct_slope` | 0.000444 |
| 12 | `H2_final` | 0.000444 |
| 13 | `H2_cv` | 0.000444 |
| 14 | `C2H2_min` | 0.000444 |
| 15 | `C2H2_range` | 0.000444 |

---

## 5. Model Hyperparameters

```python
{'categorical_features': 'from_dtype', 'class_weight': 'balanced', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 0.0, 'learning_rate': 0.08, 'loss': 'log_loss', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 200, 'max_leaf_nodes': 31, 'min_samples_leaf': 15, 'monotonic_cst': None, 'n_iter_no_change': 10, 'random_state': 42, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
```
