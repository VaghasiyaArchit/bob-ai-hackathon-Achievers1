"""
Batch Inference & Test Set Validation Runner with Explainable AI (XAI)
Power Grid AI - Real Data Machine Learning Pipeline
"""

import os
import glob
import pandas as pd
import numpy as np
from inference import TransformerRiskEngine


def run_batch_inference():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_test_dir = os.path.join(base_dir, 'data', 'raw', 'data_test')
    processed_test_file = os.path.join(base_dir, 'data', 'processed', 'test_features.csv')
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("[BATCH INFERENCE] Initializing Transformer Risk Engine with XAI...")
    engine = TransformerRiskEngine()
    
    # Load pre-extracted features
    if os.path.exists(processed_test_file):
        print(f"[BATCH INFERENCE] Loading pre-extracted features from {processed_test_file}...")
        df_test_feats = pd.read_csv(processed_test_file)
        drop_cols = [c for c in ['target_fdd', 'target_rul', 'category', 'predicted'] if c in df_test_feats.columns]
        df_features = df_test_feats.drop(columns=drop_cols)
        results_df = engine.predict_from_features(df_features)
    else:
        print(f"[BATCH INFERENCE] Reading raw CSVs from {raw_test_dir}...")
        test_files = sorted(glob.glob(os.path.join(raw_test_dir, '*.csv')))
        results_df = engine.predict_files(test_files)
        
    # Sort by risk_score descending
    results_df = results_df.sort_values(by='risk_score', ascending=False).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # VALIDATION SUITE (9-POINT AUDIT)
    # ---------------------------------------------------------
    print("\n[VALIDATION] Running automated validation checks...")
    
    # 1. Exactly 900 predictions
    if len(results_df) != 900:
        raise ValueError(f"Expected 900 predictions, but found {len(results_df)}!")
        
    # 2. No missing asset IDs
    if results_df['asset_id'].isnull().any() or (results_df['asset_id'] == '').any():
        raise ValueError("Missing or empty asset_id found in predictions!")
    if results_df['asset_id'].nunique() != 900:
        raise ValueError("Duplicate asset_id detected in prediction outputs!")
        
    # 3. No NaN predictions across any column
    nan_counts = results_df.isnull().sum().to_dict()
    if any(v > 0 for v in nan_counts.values()):
        raise ValueError(f"NaN predictions detected: {nan_counts}")
        
    # 4. Risk score between 0 and 100
    min_risk, max_risk = results_df['risk_score'].min(), results_df['risk_score'].max()
    if min_risk < 0.0 or max_risk > 100.0:
        raise ValueError(f"risk_score out of bounds [0, 100]: min={min_risk}, max={max_risk}")
        
    # 5. Anomaly score between 0 and 100
    min_anom, max_anom = results_df['anomaly_score'].min(), results_df['anomaly_score'].max()
    if min_anom < 0.0 or max_anom > 100.0:
        raise ValueError(f"anomaly_score out of bounds [0, 100]: min={min_anom}, max={max_anom}")
        
    # 6. Predicted RUL within sensible bounds (362 to 1093)
    min_rul, max_rul = results_df['predicted_rul'].min(), results_df['predicted_rul'].max()
    if min_rul < 300.0 or max_rul > 1150.0:
        raise ValueError(f"predicted_rul out of bounds: min={min_rul}, max={max_rul}")
        
    # 7. Valid risk categories
    valid_categories = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}
    unique_cats = set(results_df['risk_category'].unique())
    if not unique_cats.issubset(valid_categories):
        raise ValueError(f"Invalid risk categories: {unique_cats - valid_categories}")
        
    # 8. Valid fault classes
    valid_faults = {1, 2, 3, 4}
    unique_faults = set(results_df['predicted_fault'].unique())
    if not unique_faults.issubset(valid_faults):
        raise ValueError(f"Invalid predicted fault classes: {unique_faults - valid_faults}")
        
    # 9. Explanations and top 3 risk factors populated for all 900 rows
    if (results_df['explanation'] == '').any() or results_df['explanation'].isnull().any():
        raise ValueError("Missing explanation strings in prediction outputs!")
    if (results_df['top_3_risk_factors'] == '').any() or results_df['top_3_risk_factors'].isnull().any():
        raise ValueError("Missing top_3_risk_factors strings in prediction outputs!")
        
    # 10. No target leakage columns
    for forbidden in ['target_fdd', 'target_rul', 'category', 'predicted']:
        if forbidden in results_df.columns:
            raise ValueError(f"Leakage column '{forbidden}' found in final inference table!")
            
    print("[VALIDATION] All automated validation assertions PASSED successfully!")
    
    # Save output
    output_path = os.path.join(outputs_dir, 'test_risk_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n[SAVE] Test risk predictions saved successfully to {output_path} (900 rows, {results_df.shape[1]} columns)")
    
    # Print Fleet Risk Summary
    print("\n[FLEET RISK CATEGORY DISTRIBUTION]:")
    print(results_df['risk_category'].value_counts())
    
    # Print Top 5 Highest Risk Transformers with XAI details
    print("\n" + "="*80)
    print("TOP 5 HIGHEST-RISK TRANSFORMERS IN TEST FLEET (WITH XAI EXPLANATIONS)")
    print("="*80)
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(f"Rank {idx+1:2d} | Asset: {row['asset_id']:18s} | Risk Score: {row['risk_score']:5.1f} [{row['risk_category']:8s}]")
        print(f"        Fault: Class {row['predicted_fault']} (Prob: {row['fault_probability']*100:.1f}%) | RUL: {row['predicted_rul']:6.1f} cycles")
        print(f"        Top 3 Risk Factors: {row['top_3_risk_factors']}")
        print(f"        Action: {row['recommended_action']}")
        print(f"        Explanation: {row['explanation']}")
        print("-" * 80)
        
    return results_df, output_path


if __name__ == '__main__':
    run_batch_inference()
