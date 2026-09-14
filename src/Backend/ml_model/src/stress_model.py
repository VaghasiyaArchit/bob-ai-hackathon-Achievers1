"""
XGBoost & Gradient Boosting Transformer Stress Modeling Module
Power Grid AI - Transformer Health & Risk Engine
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing XGBoost; if unavailable, fallback to Scikit-Learn GradientBoostingRegressor
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor

STRESS_FEATURES = [
    'voltage_kV',
    'current_A',
    'load_percentage',
    'oil_temperature_c',
    'winding_temperature_c',
    'ambient_temperature_c',
    'vibration_mm_s',
    'cooling_status',
    'power_factor',
    'temp_diff_winding_oil',
    'temp_diff_oil_ambient',
    'apparent_power_mva',
    'active_power_mw',
    'is_peak_hours'
]

TARGET_COL = 'stress_index'

def train_stress_model(data_path: str, model_save_path: str):
    """
    Trains a gradient boosted decision tree regressor to predict transformer operational stress.
    Evaluates regression metrics (RMSE, MAE, R^2) and exports the serialized model.
    """
    print(f"[STRESS MODEL] Loading processed telemetry from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[STRESS_FEATURES].copy()
    y = df[TARGET_COL].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if XGB_AVAILABLE:
        print("[STRESS MODEL] Training XGBoost Regressor (n_estimators=200, max_depth=5, lr=0.05)...")
        regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("[STRESS MODEL] XGBoost not found in current environment. Using high-performance GradientBoostingRegressor...")
        regressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42
        )
        
    regressor.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n[STRESS MODEL EVALUATION METRICS]:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")
    print(f"  Test MAE  : {test_mae:.4f}")
    print(f"  Test R^2  : {test_r2:.4f}")
    
    # Feature Importances
    importances = regressor.feature_importances_
    feat_imp = sorted(zip(STRESS_FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\n[TOP FEATURE IMPORTANCES]:")
    for feat, imp in feat_imp[:6]:
        print(f"  - {feat:25s}: {imp * 100:.2f}%")
    print()
    
    # Serialize model bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_bundle = {
        'model': regressor,
        'features': STRESS_FEATURES,
        'target': TARGET_COL,
        'is_xgboost': XGB_AVAILABLE,
        'metrics': {
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2)
        }
    }
    joblib.dump(model_bundle, model_save_path)
    print(f"[STRESS MODEL] Model successfully saved to {model_save_path}")
    return regressor


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data = os.path.join(base_dir, 'data', 'processed', 'cleaned_transformer_data.csv')
    model_path = os.path.join(base_dir, 'models', 'xgboost_stress_model.pkl')
    
    train_stress_model(processed_data, model_path)
