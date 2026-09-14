"""
FastAPI Production Microservice for Transformer Predictive Maintenance
Power Grid AI - Real Data Machine Learning API
"""

import os
import sys
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Ensure ml_model/src is on python path to import TransformerRiskEngine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import TransformerRiskEngine, extract_transformer_features, FAULT_DESCRIPTIONS

# Initialize FastAPI application
app = FastAPI(
    title="Power Grid AI — Transformer Diagnostic & Risk Engine API",
    description="""
Production-grade RESTful API for electrical substation transformer condition monitoring and predictive maintenance.

### Core Capabilities:
* **Fault Detection & Diagnosis (FDD)**: 4-class multi-gas DGA classification (Normal, Partial Discharge, Thermal Oil Breakdown, Electrical Arcing).
* **Remaining Useful Life (RUL)**: Continuous operational cycle forecasting.
* **Statistical Anomaly Detection**: Robust z-score baseline deviation scoring (0-100).
* **Composite Equipment Risk Scoring**: Operational prioritization (0-100) and actionable dispatch recommendations.
* **Explainable AI (XAI)**: Top 3 physical risk factors and deterministic evidentiary explanations for utility engineers.

### Important Data Scope:
* **Sensor Scope**: Operates strictly on Dissolved Gas Analysis (DGA) chromatography (`H2`, `CO`, `C2H4`, `C2H2`).
* **Environmental/Weather Data**: **NOT INCLUDED** (designated as `MISSING` in raw benchmark datasets).
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
try:
    risk_engine = TransformerRiskEngine()
except Exception as e:
    risk_engine = None
    print(f"[API ERROR] Failed to load Risk Engine at startup: {e}")


# -------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -------------------------------------------------------------------------

class DGAObservation(BaseModel):
    H2: float = Field(..., description="Dissolved Hydrogen concentration (mol fraction)", ge=0.0)
    CO: float = Field(..., description="Dissolved Carbon Monoxide concentration (mol fraction)", ge=0.0)
    C2H4: float = Field(..., description="Dissolved Ethylene concentration (mol fraction)", ge=0.0)
    C2H2: float = Field(..., description="Dissolved Acetylene concentration (mol fraction)", ge=0.0)


class TransformerPredictRequest(BaseModel):
    asset_id: str = Field(..., description="Transformer asset identifier or filename", min_length=1)
    data: List[DGAObservation] = Field(..., description="Time-series telemetry sequence (exactly 420 observations)")

    @field_validator('data')
    @classmethod
    def validate_time_series_length(cls, v):
        if len(v) != 420:
            raise ValueError(f"Telemetry sequence must contain exactly 420 observations (received {len(v)}).")
        return v


class PredictedFault(BaseModel):
    class_: int = Field(..., alias="class", description="Predicted fault category ID (1, 2, 3, or 4)")
    label: str = Field(..., description="Physical diagnostic name of the fault mode")
    confidence: float = Field(..., description="Classification probability of the predicted fault (0.0 to 1.0)")


class FaultProbabilities(BaseModel):
    class_1: float = Field(..., description="Probability of Class 1: Normal Cellulose Degradation")
    class_2: float = Field(..., description="Probability of Class 2: Low-Energy Partial Discharge")
    class_3: float = Field(..., description="Probability of Class 3: High-Temperature Thermal Oil Breakdown")
    class_4: float = Field(..., description="Probability of Class 4: High-Energy Electrical Arcing")


class ModelMetadata(BaseModel):
    fdd_model: str = "HistGradientBoostingClassifier"
    rul_model: str = "HistGradientBoostingRegressor"
    feature_count: int = 82


class DataScope(BaseModel):
    dga_used: bool = True
    weather_used: bool = False


class TransformerRiskResponse(BaseModel):
    asset_id: str
    predicted_fault: PredictedFault
    fault_probabilities: FaultProbabilities
    predicted_rul: float
    anomaly_score: float
    risk_score: float
    risk_category: str
    top_risk_factors: List[str]
    recommended_action: str
    explanation: str
    model_metadata: ModelMetadata
    data_scope: DataScope


class HealthResponse(BaseModel):
    status: str
    fdd_model_loaded: bool
    rul_model_loaded: bool
    feature_count: int


# -------------------------------------------------------------------------
# Custom Exception Handlers (Prevent Internal Path & Traceback Leakage)
# -------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(l) for l in err.get("loc", []))
        errors.append(f"{field_loc}: {err.get('msg', 'Invalid value')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Schema Validation Error", "details": errors}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log securely without exposing local system filesystem paths
    error_msg = str(exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred processing transformer telemetry."}
    )


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def process_single_transformer_payload(payload: TransformerPredictRequest) -> Dict[str, Any]:
    """
    Transforms 420 observations into 82 domain features and executes Risk Engine inference.
    """
    if risk_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning risk engine is not initialized or model artifacts failed to load."
        )

    # Convert observation list to DataFrame
    records = [obs.model_dump() for obs in payload.data]
    df_series = pd.DataFrame(records)

    # Validate non-emptiness & types
    if df_series.isnull().values.any():
        raise HTTPException(status_code=400, detail="Missing or NaN sensor values found in telemetry sequence.")
    if np.isinf(df_series.values).any():
        raise HTTPException(status_code=400, detail="Infinite values found in telemetry sequence.")

    # Feature extraction (identical 82 features)
    try:
        feats_dict = extract_transformer_features(df_series, payload.asset_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")

    df_feats = pd.DataFrame([feats_dict])
    pred_df = risk_engine.predict_from_features(df_feats)
    res = pred_df.iloc[0].to_dict()

    # Split top_3_risk_factors from formatted string into list
    raw_factors = res.get('top_3_risk_factors', '')
    if isinstance(raw_factors, str):
        # Format is "1. ...; 2. ...; 3. ..."
        factors_list = [f.strip()[3:] if f.strip()[:3] in ['1. ', '2. ', '3. '] else f.strip() for f in raw_factors.split(';') if f.strip()]
    else:
        factors_list = list(raw_factors)

    fault_id = int(res['predicted_fault'])
    fault_label = FAULT_DESCRIPTIONS.get(fault_id, f"Fault Class {fault_id}")

    formatted_response = {
        "asset_id": payload.asset_id,
        "predicted_fault": {
            "class": fault_id,
            "label": fault_label,
            "confidence": float(res['fault_probability'])
        },
        "fault_probabilities": {
            "class_1": float(res['prob_fault_1']),
            "class_2": float(res['prob_fault_2']),
            "class_3": float(res['prob_fault_3']),
            "class_4": float(res['prob_fault_4'])
        },
        "predicted_rul": float(res['predicted_rul']),
        "anomaly_score": float(res['anomaly_score']),
        "risk_score": float(res['risk_score']),
        "risk_category": str(res['risk_category']),
        "top_risk_factors": factors_list,
        "recommended_action": str(res['recommended_action']),
        "explanation": str(res['explanation']),
        "model_metadata": {
            "fdd_model": "HistGradientBoostingClassifier",
            "rul_model": "HistGradientBoostingRegressor",
            "feature_count": 82
        },
        "data_scope": {
            "dga_used": True,
            "weather_used": False
        }
    }
    return formatted_response


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Diagnostics"],
    summary="Check API health and model readiness"
)
def get_health():
    """
    Returns runtime operational status and confirms that both the FDD and RUL models are loaded.
    """
    is_loaded = risk_engine is not None and risk_engine.fdd_model is not None and risk_engine.rul_model is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "fdd_model_loaded": bool(is_loaded),
        "rul_model_loaded": bool(is_loaded),
        "feature_count": 82
    }


@app.post(
    "/predict",
    response_model=TransformerRiskResponse,
    tags=["Prediction & Risk Assessment"],
    summary="Predict health, fault diagnosis, RUL, and operational risk for a single transformer"
)
def predict_transformer(payload: TransformerPredictRequest):
    """
    Ingests a 420-step DGA time-series, extracts 82 domain features, and generates:
    * Soft-probabilistic Fault Detection & Diagnosis (Classes 1 to 4)
    * Remaining Useful Life (RUL) estimation
    * Statistical Anomaly Score (0 to 100)
    * Composite Equipment Risk Score (0 to 100) & Operational Risk Tier (LOW, MEDIUM, HIGH, CRITICAL)
    * Actionable Maintenance Recommendation
    * Explainable AI (XAI) evidence factors and localized explanation
    """
    return process_single_transformer_payload(payload)


@app.post(
    "/predict/batch",
    response_model=List[TransformerRiskResponse],
    tags=["Prediction & Risk Assessment"],
    summary="Batch prediction for multiple transformers sorted by risk score descending"
)
def predict_batch_transformers(payloads: List[TransformerPredictRequest]):
    """
    Accepts an array of transformer telemetry sequences and returns all predictions
    sorted strictly by `risk_score` in descending order for prioritized engineering review.
    """
    if not payloads:
        raise HTTPException(status_code=400, detail="Payload list cannot be empty.")
    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100 transformers per request.")

    results = []
    for payload in payloads:
        pred = process_single_transformer_payload(payload)
        results.append(pred)

    # Sort descending by risk_score
    results = sorted(results, key=lambda x: x['risk_score'], reverse=True)
    return results


@app.get(
    "/predict/demo",
    response_model=List[TransformerRiskResponse],
    tags=["Prediction & Risk Assessment"],
    summary="Fetch live demonstration batch predictions"
)
def predict_demo():
    """
    Loads 5 sample transformer CSV files from data_test, generates predictions, and returns
    them sorted by risk score. Used by the React Frontend for a live demo.
    """
    data_dir = os.path.join(SRC_DIR, '..', 'data', 'raw', 'data_test')
    test_files = [
        "2_trans_1.csv", "2_trans_10.csv", "2_trans_1003.csv", 
        "2_trans_1004.csv", "2_trans_1005.csv"
    ]
    
    results = []
    for file in test_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            continue
            
        df = pd.read_csv(filepath)
        req = TransformerPredictRequest(
            asset_id=file.replace('.csv', ''),
            data=df[['H2', 'CO', 'C2H4', 'C2H2']].to_dict(orient='records')
        )
        try:
            pred = process_single_transformer_payload(req)
            results.append(pred)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    return sorted(results, key=lambda x: x['risk_score'], reverse=True)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
