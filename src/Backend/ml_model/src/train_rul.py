"""
Remaining Useful Life (RUL) Training Pipeline
Power Grid AI - Real Data Machine Learning Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

from model_utils import (
    load_and_verify_dataset, compute_regression_metrics,
    compute_feature_importances
)


def train_rul_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load and strictly verify dataset
    X_train, y_train, X_test, y_test, feature_names, train_ids, test_ids = load_and_verify_dataset(processed_dir, 'rul')
    
    print("\n[RUL TRAIN] Training RUL Distribution Summary:")
    print(y_train.describe().to_dict())
    
    # 2. Cross-Validation Model Comparison on Training Set
    print("\n[RUL CV] Running 5-Fold Cross-Validation on Training Set...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    candidate_models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(
            random_state=42, max_iter=200, min_samples_leaf=15, learning_rate=0.08
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=15
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=15
        )
    }
    
    cv_results = {}
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'neg_median_absolute_error']
    
    for name, reg in candidate_models.items():
        print(f"  Evaluating {name}...")
        scores = cross_validate(reg, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mae_scores = -scores['test_neg_mean_absolute_error']
        rmse_scores = np.sqrt(-scores['test_neg_mean_squared_error'])
        r2_scores = scores['test_r2']
        med_scores = -scores['test_neg_median_absolute_error']
        
        cv_results[name] = {
            'mae_mean': float(mae_scores.mean()),
            'mae_std': float(mae_scores.std()),
            'rmse_mean': float(rmse_scores.mean()),
            'rmse_std': float(rmse_scores.std()),
            'r2_mean': float(r2_scores.mean()),
            'r2_std': float(r2_scores.std()),
            'med_ae_mean': float(med_scores.mean())
        }
        print(f"    {name} -> MAE: {cv_results[name]['mae_mean']:.2f} | RMSE: {cv_results[name]['rmse_mean']:.2f} | R2: {cv_results[name]['r2_mean']:.4f}")
        
    # Select best model based on lowest MAE and RMSE
    best_model_name = min(cv_results, key=lambda k: cv_results[k]['mae_mean'])
    print(f"\n[RUL SELECTION] Selected Best Model: {best_model_name} (MAE = {cv_results[best_model_name]['mae_mean']:.2f}, RMSE = {cv_results[best_model_name]['rmse_mean']:.2f})")
    
    # 3. Retrain Best Model on Complete Training Set (2,100 samples)
    print(f"\n[RUL FIT] Retraining {best_model_name} on all 2,100 training samples...")
    final_model = candidate_models[best_model_name]
    final_model.fit(X_train, y_train)
    
    # 4. Final Evaluation on Untouched Test Set (900 samples)
    print("[RUL EVAL] Evaluating on untouched test set (900 transformers)...")
    y_test_pred = final_model.predict(X_test)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("RUL FINAL TEST SET PERFORMANCE")
    print("="*50)
    print(f"Mean Absolute Error (MAE)   : {test_metrics['mae']:.2f} cycles/steps")
    print(f"Root Mean Squared Error     : {test_metrics['rmse']:.2f}")
    print(f"R² Score                    : {test_metrics['r2']:.4f}")
    print(f"Median Absolute Error       : {test_metrics['median_absolute_error']:.2f}")
    print(f"Predictions within ±50 RUL  : {test_metrics['pct_within_50']:.2f}%")
    print(f"Predictions within ±100 RUL : {test_metrics['pct_within_100']:.2f}%")
    print(f"Predictions within ±150 RUL : {test_metrics['pct_within_150']:.2f}%")
    print("="*50)
    
    # 5. Extract Feature Importances
    print("\n[RUL IMPORTANCE] Computing feature importances...")
    df_imp = compute_feature_importances(final_model, X_test, y_test, feature_names)
    print("\nTop 10 Most Predictive RUL Features:")
    for idx, row in df_imp.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']:30s}: {row['importance']:.4f}")
        
    # 6. Save Model Bundle
    model_save_path = os.path.join(models_dir, 'rul_model.pkl')
    model_bundle = {
        'model_name': best_model_name,
        'model': final_model,
        'feature_names': feature_names,
        'target_name': 'target_rul',
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'hyperparameters': final_model.get_params(),
        'top_features': df_imp.head(20).to_dict(orient='records')
    }
    joblib.dump(model_bundle, model_save_path)
    print(f"\n[SAVE] Model bundle saved to {model_save_path}")
    
    # 7. Generate rul_evaluation.md Report
    report_path = os.path.join(outputs_dir, 'rul_evaluation.md')
    write_rul_report(report_path, best_model_name, cv_results, test_metrics, df_imp, final_model.get_params())
    print(f"[REPORT] RUL evaluation report generated at {report_path}")
    
    return model_bundle


def write_rul_report(filepath, best_model_name, cv_results, test_metrics, df_imp, params):
    md_content = f"""# Remaining Useful Life (RUL) - Model Evaluation Report

**Project**: IBM Hackathon - Power Grid AI  
**Model Task**: Continuous RUL Estimation (`target_rul` $\in [362, 1093]$ cycles/steps)  
**Selected Algorithm**: `{best_model_name}`  
**Trained Artifact**: `ml_model/models/rul_model.pkl`  

---

## 1. Cross-Validation Model Comparison (5-Fold CV on Train Set)

Three ensemble algorithms were evaluated across 5 folds on the 2,100 training transformers:

| Candidate Algorithm | MAE (Mean ± Std) | RMSE (Mean ± Std) | R² Score | Median Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
| **`HistGradientBoosting`** | **{cv_results['HistGradientBoosting']['mae_mean']:.2f} ± {cv_results['HistGradientBoosting']['mae_std']:.2f}** | **{cv_results['HistGradientBoosting']['rmse_mean']:.2f} ± {cv_results['HistGradientBoosting']['rmse_std']:.2f}** | **{cv_results['HistGradientBoosting']['r2_mean']:.4f}** | **{cv_results['HistGradientBoosting']['med_ae_mean']:.2f}** |
| `ExtraTrees` | {cv_results['ExtraTrees']['mae_mean']:.2f} ± {cv_results['ExtraTrees']['mae_std']:.2f} | {cv_results['ExtraTrees']['rmse_mean']:.2f} ± {cv_results['ExtraTrees']['rmse_std']:.2f} | {cv_results['ExtraTrees']['r2_mean']:.4f} | {cv_results['ExtraTrees']['med_ae_mean']:.2f} |
| `RandomForest` | {cv_results['RandomForest']['mae_mean']:.2f} ± {cv_results['RandomForest']['mae_std']:.2f} | {cv_results['RandomForest']['rmse_mean']:.2f} ± {cv_results['RandomForest']['rmse_std']:.2f} | {cv_results['RandomForest']['r2_mean']:.4f} | {cv_results['RandomForest']['med_ae_mean']:.2f} |

---

## 2. Final Test Performance (900 Untouched Transformers)

Evaluated on the independent test split:

- **Mean Absolute Error (MAE)**: **{test_metrics['mae']:.2f} cycles/steps**
- **Root Mean Squared Error (RMSE)**: **{test_metrics['rmse']:.2f}**
- **Coefficient of Determination ($R^2$)**: **{test_metrics['r2']:.4f}**
- **Median Absolute Error**: **{test_metrics['median_absolute_error']:.2f} cycles/steps**

### Tolerance Band Accuracy:
| Tolerance Window | Operational Interpretation | Test Accuracy (%) |
| :--- | :--- | :--- |
| **Within ±50 RUL** | High precision predictive dispatch | **{test_metrics['pct_within_50']:.2f}%** |
| **Within ±100 RUL** | Standard maintenance planning window | **{test_metrics['pct_within_100']:.2f}%** |
| **Within ±150 RUL** | Strategic asset replacement horizon | **{test_metrics['pct_within_150']:.2f}%** |

---

## 3. Top 15 Most Predictive Features

| Rank | Feature Name | Importance Metric |
| :--- | :--- | :--- |
"""
    for idx, row in df_imp.head(15).iterrows():
        md_content += f"| {idx+1} | `{row['feature']}` | {row['importance']:.6f} |\n"
        
    md_content += f"""
---

## 4. Model Hyperparameters

```python
{params}
```
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)


if __name__ == '__main__':
    train_rul_pipeline()
