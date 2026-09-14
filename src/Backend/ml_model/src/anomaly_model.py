"""
Isolation Forest Anomaly Detection Module
Power Grid AI - Transformer Health & Risk Engine
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Canonical telemetry feature set for anomaly isolation
FEATURE_COLS = [
    'voltage_kV',
    'current_A',
    'load_percentage',
    'oil_temperature_c',
    'winding_temperature_c',
    'ambient_temperature_c',
    'vibration_mm_s',
    'dissolved_hydrogen_ppm',
    'dissolved_methane_ppm',
    'dissolved_acetylene_ppm',
    'temp_diff_winding_oil',
    'temp_diff_oil_ambient',
    'apparent_power_mva'
]

def train_anomaly_model(data_path: str, model_save_path: str, output_csv_path: str, contamination: float = 0.06):
    """
    Trains an Isolation Forest model to flag multidimensional operational anomalies,
    serializes the trained model artifact, and outputs prediction metrics.
    """
    print(f"[ANOMALY] Loading processed telemetry from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_COLS].copy()
    
    # Standardize features for stable tree partitioning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[ANOMALY] Training Isolation Forest (n_estimators=150, contamination={contamination})...")
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    
    # In scikit-learn, decision_function returns negative values for anomalies and positive for normal
    # We normalize this into an anomaly_score [0, 1] where higher = more anomalous
    raw_scores = iso_forest.decision_function(X_scaled)
    # Min-max inversion so lower decision function -> higher anomaly score
    min_s, max_s = raw_scores.min(), raw_scores.max()
    anomaly_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s + 1e-8))
    
    # Predicted labels: -1 = outlier, 1 = inlier. Map to 1 = anomaly, 0 = normal
    raw_preds = iso_forest.predict(X_scaled)
    anomaly_preds = np.where(raw_preds == -1, 1, 0)
    
    # Save model and scaler bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_bundle = {
        'model': iso_forest,
        'scaler': scaler,
        'features': FEATURE_COLS,
        'contamination': contamination
    }
    joblib.dump(model_bundle, model_save_path)
    print(f"[ANOMALY] Model saved successfully to {model_save_path}")
    
    # Append predictions to dataframe
    results_df = df.copy()
    results_df['iso_forest_pred'] = anomaly_preds
    results_df['anomaly_score'] = anomaly_scores.round(4)
    
    # Evaluate if ground truth anomaly_flag is present
    if 'anomaly_flag' in results_df.columns:
        print("\n[ANOMALY EVALUATION against injected anomaly_flag]:")
        y_true = results_df['anomaly_flag']
        print(classification_report(y_true, anomaly_preds, target_names=['Normal', 'Anomaly']))
        roc_score = roc_auc_score(y_true, anomaly_scores)
        print(f"ROC AUC Score: {roc_score:.4f}\n")
        
    # Export anomaly results
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_cols = [
        'timestamp', 'transformer_id', 'load_percentage', 'oil_temperature_c',
        'winding_temperature_c', 'vibration_mm_s', 'dissolved_hydrogen_ppm',
        'dissolved_acetylene_ppm', 'iso_forest_pred', 'anomaly_score'
    ]
    if 'anomaly_flag' in results_df.columns:
        output_cols.append('anomaly_flag')
        
    results_df[output_cols].to_csv(output_csv_path, index=False)
    print(f"[ANOMALY] Anomaly results saved to {output_csv_path}")
    return results_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data = os.path.join(base_dir, 'data', 'processed', 'cleaned_transformer_data.csv')
    model_path = os.path.join(base_dir, 'models', 'isolation_forest.pkl')
    output_path = os.path.join(base_dir, 'outputs', 'anomaly_results.csv')
    
    train_anomaly_model(processed_data, model_path, output_path)
