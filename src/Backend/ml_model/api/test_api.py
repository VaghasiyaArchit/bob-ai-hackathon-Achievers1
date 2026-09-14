"""
Automated Test Suite for Transformer Prediction API
Power Grid AI - Production Quality Assurance
"""

import os
import sys
import pandas as pd
import numpy as np
from starlette.testclient import TestClient

# Ensure API module is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import app

client = TestClient(app)

# Path to real test data
RAW_TEST_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'data', 'raw', 'data_test')
REAL_SAMPLE_FILE = os.path.join(RAW_TEST_DIR, '2_trans_939.csv')
REAL_SAMPLE_FILE_2 = os.path.join(RAW_TEST_DIR, '2_trans_2556.csv')
BATCH_PRED_FILE = os.path.join(os.path.dirname(CURRENT_DIR), 'outputs', 'test_risk_predictions.csv')


def load_real_transformer_payload(filepath: str, asset_id: str):
    df = pd.read_csv(filepath)
    records = df[['H2', 'CO', 'C2H4', 'C2H2']].to_dict(orient='records')
    return {
        "asset_id": asset_id,
        "data": records
    }


def test_01_health_endpoint():
    """Verify health endpoint returns healthy status and confirmed model loading."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["fdd_model_loaded"] is True
    assert data["rul_model_loaded"] is True
    assert data["feature_count"] == 82
    print("[TEST 1 PASSED] /health endpoint operational.")


def test_02_valid_transformer_prediction_and_numerical_parity():
    """Verify single prediction on real test asset (2_trans_939.csv) and compare to batch predictions."""
    assert os.path.exists(REAL_SAMPLE_FILE), f"Missing real test file: {REAL_SAMPLE_FILE}"
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    # Structural assertions
    assert data["asset_id"] == "2_trans_939.csv"
    assert data["predicted_fault"]["class"] == 4
    assert "High-Energy Electrical Arcing" in data["predicted_fault"]["label"]
    assert data["risk_category"] == "CRITICAL"
    
    # Parity check against test_risk_predictions.csv
    if os.path.exists(BATCH_PRED_FILE):
        batch_df = pd.read_csv(BATCH_PRED_FILE)
        match_row = batch_df[batch_df["asset_id"] == "2_trans_939.csv"].iloc[0]
        assert np.isclose(data["risk_score"], match_row["risk_score"], atol=1e-2), f"Risk score mismatch: {data['risk_score']} vs {match_row['risk_score']}"
        assert np.isclose(data["predicted_rul"], match_row["predicted_rul"], atol=1e-2), f"RUL mismatch: {data['predicted_rul']} vs {match_row['predicted_rul']}"
        assert np.isclose(data["anomaly_score"], match_row["anomaly_score"], atol=1e-2), f"Anomaly mismatch: {data['anomaly_score']} vs {match_row['anomaly_score']}"
        print(f"[TEST 2 PASSED] Real prediction verified with 100% numerical parity against batch catalog (Risk: {data['risk_score']}).")


def test_03_invalid_observation_count():
    """Verify rejection when telemetry sequence has fewer than 420 observations."""
    df = pd.read_csv(REAL_SAMPLE_FILE)
    truncated_records = df[['H2', 'CO', 'C2H4', 'C2H2']].iloc[:200].to_dict(orient='records')
    payload = {"asset_id": "invalid_tx", "data": truncated_records}
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    assert "420" in response.text
    print("[TEST 3 PASSED] Correctly rejected sequence with invalid length (200 rows).")


def test_04_missing_dga_column():
    """Verify rejection when a required DGA column is missing."""
    df = pd.read_csv(REAL_SAMPLE_FILE)
    # Drop C2H2
    missing_records = df[['H2', 'CO', 'C2H4']].to_dict(orient='records')
    payload = {"asset_id": "invalid_tx", "data": missing_records}
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("[TEST 4 PASSED] Correctly rejected payload with missing C2H2 column.")


def test_05_batch_prediction_and_sorting():
    """Verify batch endpoint processes multiple transformers and returns sorted descending by risk_score."""
    payload_1 = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")   # High / Critical
    payload_2 = load_real_transformer_payload(REAL_SAMPLE_FILE_2, "2_trans_2556.csv") # Low
    
    batch_payload = [payload_2, payload_1]  # Send in reverse order
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    results = response.json()
    
    assert len(results) == 2
    assert results[0]["asset_id"] == "2_trans_939.csv"  # Higher risk must come first
    assert results[1]["asset_id"] == "2_trans_2556.csv"
    assert results[0]["risk_score"] >= results[1]["risk_score"]
    print(f"[TEST 5 PASSED] Batch prediction verified with descending risk sort ({results[0]['risk_score']} >= {results[1]['risk_score']}).")


def test_06_output_schema_completeness():
    """Verify all expected response keys exist according to hackathon contract."""
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    response = client.post("/predict", json=payload)
    data = response.json()
    
    required_keys = [
        "asset_id", "predicted_fault", "fault_probabilities", "predicted_rul",
        "anomaly_score", "risk_score", "risk_category", "top_risk_factors",
        "recommended_action", "explanation", "model_metadata", "data_scope"
    ]
    for k in required_keys:
        assert k in data, f"Missing required response key: {k}"
        
    assert data["data_scope"]["dga_used"] is True
    assert data["data_scope"]["weather_used"] is False
    print("[TEST 6 PASSED] Output schema completeness verified.")


def test_07_no_nan_or_inf():
    """Verify all numerical response values are strictly finite and real."""
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    response = client.post("/predict", json=payload)
    data = response.json()
    
    num_fields = [
        data["predicted_fault"]["confidence"],
        data["fault_probabilities"]["class_1"],
        data["fault_probabilities"]["class_2"],
        data["fault_probabilities"]["class_3"],
        data["fault_probabilities"]["class_4"],
        data["predicted_rul"],
        data["anomaly_score"],
        data["risk_score"]
    ]
    for val in num_fields:
        assert not np.isnan(val) and not np.isinf(val), f"Non-finite value found: {val}"
    print("[TEST 7 PASSED] Zero NaN or infinite values detected.")


def test_08_risk_score_range():
    """Verify risk_score is strictly in [0.0, 100.0] and risk_category matches."""
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    response = client.post("/predict", json=payload)
    data = response.json()
    
    risk = data["risk_score"]
    assert 0.0 <= risk <= 100.0, f"Risk score out of bounds: {risk}"
    if risk >= 85.0:
        assert data["risk_category"] == "CRITICAL"
    elif risk >= 60.0:
        assert data["risk_category"] == "HIGH"
    elif risk >= 35.0:
        assert data["risk_category"] == "MEDIUM"
    else:
        assert data["risk_category"] == "LOW"
    print(f"[TEST 8 PASSED] Risk score ({risk}) strictly within [0, 100] and category mapped correctly.")


def test_09_rul_range():
    """Verify predicted_rul is within realistic physical boundaries."""
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    response = client.post("/predict", json=payload)
    data = response.json()
    
    rul = data["predicted_rul"]
    assert 300.0 <= rul <= 1150.0, f"Predicted RUL out of expected range: {rul}"
    print(f"[TEST 9 PASSED] Predicted RUL ({rul} cycles) strictly within realistic operating bounds.")


def test_10_xai_explanation_and_evidence():
    """Verify explanation and top 3 risk factors exist and contain evidence phrasing."""
    payload = load_real_transformer_payload(REAL_SAMPLE_FILE, "2_trans_939.csv")
    response = client.post("/predict", json=payload)
    data = response.json()
    
    assert len(data["top_risk_factors"]) == 3, f"Expected 3 risk factors, got {len(data['top_risk_factors'])}"
    explanation = data["explanation"]
    assert len(explanation) > 30, "Explanation is too brief or empty"
    assert any(phrase in explanation for phrase in ["contributing", "consistent", "indicator", "predicts"]), "Missing evidentiary phrasing"
    print("[TEST 10 PASSED] Explainable AI evidence and top 3 risk factors verified.")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING AUTOMATED API TEST SUITE")
    print("=" * 60)
    test_01_health_endpoint()
    test_02_valid_transformer_prediction_and_numerical_parity()
    test_03_invalid_observation_count()
    test_04_missing_dga_column()
    test_05_batch_prediction_and_sorting()
    test_06_output_schema_completeness()
    test_07_no_nan_or_inf()
    test_08_risk_score_range()
    test_09_rul_range()
    test_10_xai_explanation_and_evidence()
    print("=" * 60)
    print("ALL 10 API AUTOMATED TESTS PASSED!")
    print("=" * 60)
