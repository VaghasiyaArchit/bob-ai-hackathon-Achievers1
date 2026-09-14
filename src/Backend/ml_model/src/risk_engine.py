"""
Composite Grid Risk Engine Module
Power Grid AI - Transformer Health & Risk Engine
"""

import os
import joblib
import numpy as np
import pandas as pd

def calculate_dga_severity(df: pd.DataFrame) -> pd.Series:
    """
    Computes a Dissolved Gas Analysis (DGA) severity score (0 to 100)
    based on IEEE C57.104 gas concentration guidelines.
    """
    h2 = df['dissolved_hydrogen_ppm']
    ch4 = df['dissolved_methane_ppm']
    c2h2 = df['dissolved_acetylene_ppm']
    
    # Gas risk tiers:
    # Acetylene is most severe (indicates active high-energy arcing)
    c2h2_score = np.clip(c2h2 / 15.0 * 50.0, 0, 50)
    h2_score = np.clip(h2 / 200.0 * 25.0, 0, 25)
    ch4_score = np.clip(ch4 / 150.0 * 25.0, 0, 25)
    
    dga_severity = (c2h2_score + h2_score + ch4_score).round(2)
    return dga_severity

def assign_risk_tier(score: float) -> str:
    if score >= 80.0:
        return 'CRITICAL'
    elif score >= 60.0:
        return 'HIGH'
    elif score >= 35.0:
        return 'MEDIUM'
    else:
        return 'LOW'

def assign_mitigation_action(tier: str, dga_sev: float, stress: float) -> str:
    if tier == 'CRITICAL':
        if dga_sev > 40.0:
            return "URGENT DISPATCH: High-energy arcing detected. Trip breaker & schedule immediate physical oil de-gassing."
        else:
            return "CRITICAL OVERLOAD: Initiate emergency 25% load shedding and engage auxiliary radiator banks."
    elif tier == 'HIGH':
        if stress > 65.0:
            return "HIGH THERMAL STRESS: Force secondary stage cooling fans and reroute 15% feeder load to adjacent sub."
        else:
            return "ELEVATED DEGRADATION: Schedule field acoustic vibration test and oil sampling within 48 hours."
    elif tier == 'MEDIUM':
        return "ADVISORY: Increase SCADA polling frequency to 5 minutes; monitor hot-spot temperature delta."
    else:
        return "NOMINAL: Operating within normal thermal & electrical limits. Routine maintenance cycle."

def run_risk_engine(
    processed_data_path: str,
    iso_model_path: str,
    stress_model_path: str,
    output_risk_path: str
) -> pd.DataFrame:
    """
    Runs inference across both ML models and computes composite grid asset risk.
    """
    print(f"[RISK ENGINE] Loading telemetry from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)
    
    # 1. Load trained models
    print(f"[RISK ENGINE] Loading Isolation Forest from {iso_model_path}...")
    iso_bundle = joblib.load(iso_model_path)
    iso_model = iso_bundle['model']
    iso_scaler = iso_bundle['scaler']
    iso_features = iso_bundle['features']
    
    print(f"[RISK ENGINE] Loading Stress Model from {stress_model_path}...")
    stress_bundle = joblib.load(stress_model_path)
    stress_model = stress_bundle['model']
    stress_features = stress_bundle['features']
    
    # 2. Run Anomaly Model inference
    X_iso = iso_scaler.transform(df[iso_features])
    raw_scores = iso_model.decision_function(X_iso)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    # Normalize to 0-100 anomaly intensity
    anomaly_intensity = (1.0 - ((raw_scores - min_s) / (max_s - min_s + 1e-8))) * 100.0
    
    # 3. Run Stress Model inference
    X_stress = df[stress_features]
    predicted_stress = stress_model.predict(X_stress)
    predicted_stress = np.clip(predicted_stress, 0.0, 100.0)
    
    # 4. Compute DGA Severity
    dga_severity = calculate_dga_severity(df)
    
    # 5. Composite Risk Index Formulation:
    # 40% Operational Stress + 35% Anomaly Intensity + 25% DGA Chemical Severity
    composite_risk = (
        0.40 * predicted_stress +
        0.35 * anomaly_intensity +
        0.25 * dga_severity
    ).round(2)
    composite_risk = np.clip(composite_risk, 0.0, 100.0)
    
    # Build results dataframe
    results = df[['timestamp', 'transformer_id', 'load_percentage', 'winding_temperature_c', 'oil_temperature_c']].copy()
    results['predicted_stress_index'] = predicted_stress.round(2)
    results['anomaly_intensity'] = anomaly_intensity.round(2)
    results['dga_severity_index'] = dga_severity
    results['composite_risk_score'] = composite_risk
    results['risk_tier'] = results['composite_risk_score'].apply(assign_risk_tier)
    
    results['recommended_action'] = [
        assign_mitigation_action(t, d, s)
        for t, d, s in zip(results['risk_tier'], results['dga_severity_index'], results['predicted_stress_index'])
    ]
    
    # Save output
    os.makedirs(os.path.dirname(output_risk_path), exist_ok=True)
    results.to_csv(output_risk_path, index=False)
    print(f"[RISK ENGINE] Risk results written to {output_risk_path}")
    
    print("\n[GRID FLEET RISK SUMMARY]:")
    print(results['risk_tier'].value_counts())
    print("\n[TOP 5 HIGHEST RISK TRANSFORMER INCIDENTS]:")
    top_risks = results.sort_values(by='composite_risk_score', ascending=False).head(5)
    for _, row in top_risks.iterrows():
        print(f"  Asset: {row['transformer_id']:12s} | Risk: {row['composite_risk_score']:5.1f} [{row['risk_tier']:8s}] | Action: {row['recommended_action']}")
    print()
    return results


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data = os.path.join(base_dir, 'data', 'processed', 'cleaned_transformer_data.csv')
    iso_model = os.path.join(base_dir, 'models', 'isolation_forest.pkl')
    stress_model = os.path.join(base_dir, 'models', 'xgboost_stress_model.pkl')
    output_risk = os.path.join(base_dir, 'outputs', 'risk_results.csv')
    
    run_risk_engine(processed_data, iso_model, stress_model, output_risk)
