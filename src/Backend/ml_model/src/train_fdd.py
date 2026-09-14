"""
Fault Detection & Diagnosis (FDD) Training Pipeline
Power Grid AI - Real Data Machine Learning Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

from model_utils import (
    load_and_verify_dataset, compute_classification_metrics,
    compute_feature_importances
)


def train_fdd_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load and strictly verify dataset
    X_train, y_train, X_test, y_test, feature_names, train_ids, test_ids = load_and_verify_dataset(processed_dir, 'fdd')
    
    print("\n[FDD TRAIN] Training Class Distribution:")
    print(y_train.value_counts().sort_index().to_dict())
    
    # 2. Cross-Validation Model Comparison on Training Set
    print("\n[FDD CV] Running 5-Fold Stratified Cross-Validation on Training Set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidate_models = {
        'HistGradientBoosting': HistGradientBoostingClassifier(
            class_weight='balanced', random_state=42, max_iter=200, min_samples_leaf=15, learning_rate=0.08
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1, max_depth=15
        ),
        'ExtraTrees': ExtraTreesClassifier(
            class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1, max_depth=15
        )
    }
    
    cv_results = {}
    scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    
    for name, clf in candidate_models.items():
        print(f"  Evaluating {name}...")
        scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            'macro_f1_mean': float(scores['test_f1_macro'].mean()),
            'macro_f1_std': float(scores['test_f1_macro'].std()),
            'balanced_acc_mean': float(scores['test_balanced_accuracy'].mean()),
            'balanced_acc_std': float(scores['test_balanced_accuracy'].std()),
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'weighted_f1_mean': float(scores['test_f1_weighted'].mean())
        }
        print(f"    {name} -> Macro F1: {cv_results[name]['macro_f1_mean']:.4f} (+/- {cv_results[name]['macro_f1_std']:.4f}) | Balanced Acc: {cv_results[name]['balanced_acc_mean']:.4f}")
        
    # Select best model based on Macro F1
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['macro_f1_mean'])
    print(f"\n[FDD SELECTION] Selected Best Model: {best_model_name} (Macro F1 = {cv_results[best_model_name]['macro_f1_mean']:.4f})")
    
    # 3. Retrain Best Model on Complete Training Set (2,100 samples)
    print(f"\n[FDD FIT] Retraining {best_model_name} on all 2,100 training samples...")
    final_model = candidate_models[best_model_name]
    final_model.fit(X_train, y_train)
    
    # 4. Final Evaluation on Untouched Test Set (900 samples)
    print("[FDD EVAL] Evaluating on untouched test set (900 transformers)...")
    y_test_pred = final_model.predict(X_test)
    test_metrics = compute_classification_metrics(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("FDD FINAL TEST SET PERFORMANCE")
    print("="*50)
    print(f"Accuracy         : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro Precision  : {test_metrics['precision_macro']:.4f}")
    print(f"Macro Recall     : {test_metrics['recall_macro']:.4f}")
    print(f"Macro F1         : {test_metrics['f1_macro']:.4f}")
    print(f"Weighted F1      : {test_metrics['f1_weighted']:.4f}")
    print("Confusion Matrix :")
    print(np.array(test_metrics['confusion_matrix']))
    print("="*50)
    
    # 5. Extract Feature Importances
    print("\n[FDD IMPORTANCE] Computing feature importances...")
    df_imp = compute_feature_importances(final_model, X_test, y_test, feature_names)
    print("\nTop 10 Most Predictive FDD Features:")
    for idx, row in df_imp.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']:30s}: {row['importance']:.4f}")
        
    # 6. Save Model Bundle
    model_save_path = os.path.join(models_dir, 'fdd_model.pkl')
    model_bundle = {
        'model_name': best_model_name,
        'model': final_model,
        'feature_names': feature_names,
        'target_name': 'target_fdd',
        'classes': [1, 2, 3, 4],
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'hyperparameters': final_model.get_params(),
        'top_features': df_imp.head(20).to_dict(orient='records')
    }
    joblib.dump(model_bundle, model_save_path)
    print(f"\n[SAVE] Model bundle saved to {model_save_path}")
    
    # 7. Generate fdd_evaluation.md Report
    report_path = os.path.join(outputs_dir, 'fdd_evaluation.md')
    write_fdd_report(report_path, best_model_name, cv_results, test_metrics, df_imp, final_model.get_params())
    print(f"[REPORT] FDD evaluation report generated at {report_path}")
    
    return model_bundle


def write_fdd_report(filepath, best_model_name, cv_results, test_metrics, df_imp, params):
    cm = test_metrics['confusion_matrix']
    cr = test_metrics['classification_report']
    
    md_content = f"""# Fault Detection & Diagnosis (FDD) - Model Evaluation Report

**Project**: IBM Hackathon - Power Grid AI  
**Model Task**: Multi-Class Fault Classification (`target_fdd` $\in \\{{1, 2, 3, 4\\}}$)  
**Selected Algorithm**: `{best_model_name}` (with balanced class weighting)  
**Trained Artifact**: `ml_model/models/fdd_model.pkl`  

---

## 1. Cross-Validation Model Comparison (5-Fold Stratified CV)

Cross-validation was performed exclusively on the 2,100 training transformers to select the optimal architecture:

| Candidate Algorithm | Macro F1 (Mean ± Std) | Balanced Accuracy | Overall Accuracy | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| **`HistGradientBoosting`** | **{cv_results['HistGradientBoosting']['macro_f1_mean']:.4f} ± {cv_results['HistGradientBoosting']['macro_f1_std']:.4f}** | **{cv_results['HistGradientBoosting']['balanced_acc_mean']:.4f}** | **{cv_results['HistGradientBoosting']['accuracy_mean']:.4f}** | **{cv_results['HistGradientBoosting']['weighted_f1_mean']:.4f}** |
| `RandomForest` | {cv_results['RandomForest']['macro_f1_mean']:.4f} ± {cv_results['RandomForest']['macro_f1_std']:.4f} | {cv_results['RandomForest']['balanced_acc_mean']:.4f} | {cv_results['RandomForest']['accuracy_mean']:.4f} | {cv_results['RandomForest']['weighted_f1_mean']:.4f} |
| `ExtraTrees` | {cv_results['ExtraTrees']['macro_f1_mean']:.4f} ± {cv_results['ExtraTrees']['macro_f1_std']:.4f} | {cv_results['ExtraTrees']['balanced_acc_mean']:.4f} | {cv_results['ExtraTrees']['accuracy_mean']:.4f} | {cv_results['ExtraTrees']['weighted_f1_mean']:.4f} |

---

## 2. Final Test Performance (900 Untouched Transformers)

Evaluated on the independent test split:

- **Accuracy**: **{test_metrics['accuracy'] * 100:.2f}%**
- **Balanced Accuracy**: **{test_metrics['balanced_accuracy'] * 100:.2f}%**
- **Macro Precision**: **{test_metrics['precision_macro']:.4f}**
- **Macro Recall**: **{test_metrics['recall_macro']:.4f}**
- **Macro F1-Score**: **{test_metrics['f1_macro']:.4f}**
- **Weighted F1-Score**: **{test_metrics['f1_weighted']:.4f}**

### Per-Class Diagnostic Performance:
| Category | Physical Diagnosis | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Normal / Cellulose Aging | {cr['1']['precision']:.4f} | {cr['1']['recall']:.4f} | {cr['1']['f1-score']:.4f} | {cr['1']['support']} |
| **2** | Low-Energy Partial Discharge (PD) | {cr['2']['precision']:.4f} | {cr['2']['recall']:.4f} | {cr['2']['f1-score']:.4f} | {cr['2']['support']} |
| **3** | High-Temp Thermal Oil Fault | {cr['3']['precision']:.4f} | {cr['3']['recall']:.4f} | {cr['3']['f1-score']:.4f} | {cr['3']['support']} |
| **4** | High-Energy Arcing / Mixed | {cr['4']['precision']:.4f} | {cr['4']['recall']:.4f} | {cr['4']['f1-score']:.4f} | {cr['4']['support']} |

---

## 3. Confusion Matrix (Test Set)

```
                Predicted Cat 1   Predicted Cat 2   Predicted Cat 3   Predicted Cat 4
Actual Cat 1        {cm[0][0]:<6}            {cm[0][1]:<6}            {cm[0][2]:<6}            {cm[0][3]:<6}
Actual Cat 2        {cm[1][0]:<6}            {cm[1][1]:<6}            {cm[1][2]:<6}            {cm[1][3]:<6}
Actual Cat 3        {cm[2][0]:<6}            {cm[2][1]:<6}            {cm[2][2]:<6}            {cm[2][3]:<6}
Actual Cat 4        {cm[3][0]:<6}            {cm[3][1]:<6}            {cm[3][2]:<6}            {cm[3][3]:<6}
```

---

## 4. Top 15 Most Predictive Features

| Rank | Feature Name | Importance Metric |
| :--- | :--- | :--- |
"""
    for idx, row in df_imp.head(15).iterrows():
        md_content += f"| {idx+1} | `{row['feature']}` | {row['importance']:.6f} |\n"
        
    md_content += f"""
---

## 5. Model Hyperparameters

```python
{params}
```
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)


if __name__ == '__main__':
    train_fdd_pipeline()
