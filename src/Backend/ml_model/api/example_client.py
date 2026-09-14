"""
Example Client Integration for Power Grid AI Microservice
Demonstrates how the backend / dashboard application calls the prediction API.
"""

import os
import json
import pandas as pd
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def check_health():
    """Calls the health endpoint to confirm model readiness."""
    url = f"{API_BASE_URL}/health"
    print(f"--> Checking API health: GET {url}")
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        print(f"<-- Health Status: {resp.json()}\n")
        return True
    except Exception as e:
        print(f"[ERROR] API is not accessible at {url}: {e}\n")
        return False


def predict_single_transformer(csv_filepath: str, asset_id: str):
    """
    Reads a 420-step transformer DGA CSV file and sends it to POST /predict.
    """
    url = f"{API_BASE_URL}/predict"
    print(f"--> Sending telemetry for {asset_id} to POST {url}")
    
    # 1. Load real 420-step telemetry sequence
    df = pd.read_csv(csv_filepath)
    payload = {
        "asset_id": asset_id,
        "data": df[['H2', 'CO', 'C2H4', 'C2H2']].to_dict(orient='records')
    }
    
    # 2. Call API
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f"[ERROR {resp.status_code}] {resp.text}")
        return None
        
    result = resp.json()
    
    # 3. Print formatted diagnostic summary
    print("\n" + "=" * 65)
    print(f"DIAGNOSTIC REPORT FOR: {result['asset_id']}")
    print("=" * 65)
    print(f"• Equipment Risk Score : {result['risk_score']:.1f} / 100 [{result['risk_category']}]")
    print(f"• Predicted Fault Mode : {result['predicted_fault']['label']} (Confidence: {result['predicted_fault']['confidence']*100:.1f}%)")
    print(f"• Estimated RUL        : {result['predicted_rul']:.1f} cycles/steps")
    print(f"• Anomaly Score        : {result['anomaly_score']:.1f} / 100")
    print(f"• Top Risk Factors     :")
    for idx, factor in enumerate(result['top_risk_factors'], 1):
        print(f"   {idx}. {factor}")
    print(f"• Recommended Action   : {result['recommended_action']}")
    print(f"• Local XAI Explanation: {result['explanation']}")
    print(f"• Data Scope           : DGA={result['data_scope']['dga_used']}, Weather={result['data_scope']['weather_used']}")
    print("=" * 65 + "\n")
    return result


if __name__ == '__main__':
    # Locate a sample real test file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_sample = os.path.join(base_dir, 'data', 'raw', 'data_test', '2_trans_939.csv')
    
    print("Power Grid AI - Client Integration Demonstration\n")
    if check_health():
        if os.path.exists(test_sample):
            predict_single_transformer(test_sample, "2_trans_939.csv")
        else:
            print(f"Sample file not found at {test_sample}")
    else:
        print("Note: Start the FastAPI server first using:")
        print("  uvicorn ml_model.api.main:app --reload --port 8000")
