"""
Model Utilities & Evaluation Helpers
Power Grid AI - Real Data Machine Learning Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
)
from sklearn.inspection import permutation_importance

LEAKAGE_TARGETS = ['target_fdd', 'target_rul', 'category', 'predicted', 'id']
METADATA_COLS = ['asset_id']


def load_and_verify_dataset(processed_dir: str, task: str):
    """
    Loads train and test datasets for the specified task ('fdd' or 'rul'),
    and runs strict automated verification against data leakage and corruption.
    """
    if task not in ['fdd', 'rul']:
        raise ValueError("Task must be either 'fdd' or 'rul'")
        
    train_file = os.path.join(processed_dir, f'train_{task}.csv')
    test_file = os.path.join(processed_dir, f'test_{task}.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Processed dataset missing: {train_file} or {test_file}")
        
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    target_col = f'target_{task}'
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {task} dataset!")
        
    # Feature columns exclusion
    feature_cols = [c for c in train_df.columns if c not in [target_col] + METADATA_COLS]
    test_feature_cols = [c for c in test_df.columns if c not in [target_col] + METADATA_COLS]
    
    # 1. Feature columns identical
    if feature_cols != test_feature_cols:
        raise ValueError(f"Feature column mismatch between train and test for task {task}!")
        
    # 2. Exactly 82 features
    if len(feature_cols) != 82:
        raise ValueError(f"Expected 82 feature columns, found {len(feature_cols)}!")
        
    # 3. No leakage
    for forbidden in LEAKAGE_TARGETS:
        if forbidden in feature_cols:
            raise ValueError(f"[LEAKAGE DETECTED] Forbidden column '{forbidden}' found in feature set!")
    if 'asset_id' in feature_cols:
        raise ValueError("[LEAKAGE DETECTED] 'asset_id' must not be in model features!")
        
    # 4. Check shapes
    if len(train_df) != 2100 or len(test_df) != 900:
        raise ValueError(f"Unexpected row count: train={len(train_df)}, test={len(test_df)}")
        
    # 5. Check no NaN or Infinite values
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        raise ValueError("NaN values detected in feature matrix!")
    if np.isinf(X_train.values).sum() > 0 or np.isinf(X_test.values).sum() > 0:
        raise ValueError("Infinite values detected in feature matrix!")
        
    print(f"[VERIFY] {task.upper()} dataset verified: {len(X_train)} Train, {len(X_test)} Test, {len(feature_cols)} Features. Zero leakage.")
    return X_train, y_train, X_test, y_test, feature_cols, train_df['asset_id'], test_df['asset_id']


def compute_classification_metrics(y_true, y_pred):
    """
    Computes comprehensive multi-class classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4], output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(b_acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def compute_regression_metrics(y_true, y_pred):
    """
    Computes comprehensive continuous regression metrics including tolerance accuracy.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    pct_within_50 = float(np.mean(errors <= 50.0) * 100.0)
    pct_within_100 = float(np.mean(errors <= 100.0) * 100.0)
    pct_within_150 = float(np.mean(errors <= 150.0) * 100.0)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'median_absolute_error': float(med_ae),
        'pct_within_50': pct_within_50,
        'pct_within_100': pct_within_100,
        'pct_within_150': pct_within_150
    }


def compute_feature_importances(model, X_val, y_val, feature_names, n_repeats=5, random_state=42):
    """
    Extracts tree feature importances or permutation importances if native is absent.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Use permutation importance for HistGradientBoosting
        result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        importances = result.importances_mean
        
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    return df_imp
