# Remaining Useful Life (RUL) - Model Evaluation Report

**Project**: IBM Hackathon - Power Grid AI  
**Model Task**: Continuous RUL Estimation (`target_rul` $\in [362, 1093]$ cycles/steps)  
**Selected Algorithm**: `HistGradientBoosting`  
**Trained Artifact**: `ml_model/models/rul_model.pkl`  

---

## 1. Cross-Validation Model Comparison (5-Fold CV on Train Set)

Three ensemble algorithms were evaluated across 5 folds on the 2,100 training transformers:

| Candidate Algorithm | MAE (Mean ± Std) | RMSE (Mean ± Std) | R² Score | Median Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
| **`HistGradientBoosting`** | **74.76 ± 3.92** | **105.95 ± 6.82** | **0.8090** | **48.58** |
| `ExtraTrees` | 79.70 ± 4.14 | 109.37 ± 5.66 | 0.7966 | 54.88 |
| `RandomForest` | 82.30 ± 4.38 | 112.56 ± 6.45 | 0.7846 | 58.03 |

---

## 2. Final Test Performance (900 Untouched Transformers)

Evaluated on the independent test split:

- **Mean Absolute Error (MAE)**: **74.57 cycles/steps**
- **Root Mean Squared Error (RMSE)**: **104.13**
- **Coefficient of Determination ($R^2$)**: **0.8093**
- **Median Absolute Error**: **49.34 cycles/steps**

### Tolerance Band Accuracy:
| Tolerance Window | Operational Interpretation | Test Accuracy (%) |
| :--- | :--- | :--- |
| **Within ±50 RUL** | High precision predictive dispatch | **50.78%** |
| **Within ±100 RUL** | Standard maintenance planning window | **72.78%** |
| **Within ±150 RUL** | Strategic asset replacement horizon | **85.89%** |

---

## 3. Top 15 Most Predictive Features

| Rank | Feature Name | Importance Metric |
| :--- | :--- | :--- |
| 1 | `H2_final` | 0.239678 |
| 2 | `C2H4_final` | 0.184925 |
| 3 | `TDCG_final` | 0.172962 |
| 4 | `C2H2_final` | 0.158439 |
| 5 | `H2_recent_10pct_slope` | 0.077530 |
| 6 | `C2H4_recent_10pct_slope` | 0.052428 |
| 7 | `C2H2_recent_10pct_slope` | 0.044967 |
| 8 | `CO_recent_10pct_slope` | 0.036093 |
| 9 | `CO_final` | 0.022764 |
| 10 | `C2H4_recent_20pct_slope` | 0.015918 |
| 11 | `H2_median` | 0.010233 |
| 12 | `C2H4_std` | 0.008914 |
| 13 | `CO_TDCG_ratio` | 0.007288 |
| 14 | `H2_recent_20pct_slope` | 0.007152 |
| 15 | `C2H4_abs_change` | 0.005471 |

---

## 4. Model Hyperparameters

```python
{'categorical_features': 'from_dtype', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 0.0, 'learning_rate': 0.08, 'loss': 'squared_error', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 200, 'max_leaf_nodes': 31, 'min_samples_leaf': 15, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None, 'random_state': 42, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
```
