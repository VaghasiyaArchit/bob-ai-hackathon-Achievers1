# Power Grid AI: Transformer Health & Risk Prediction API

This directory contains the production-grade **FastAPI microservice** that serves real-time predictive maintenance diagnostics, fault classifications, remaining useful life estimates, and explainable AI (XAI) risk scores for the Power Grid AI platform.

---

## 🚀 How to Start the API

Run the service using `uvicorn`:

```bash
# From the repository root (power-grid-ai):
python -m uvicorn ml_model.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from within `ml_model/api`:
```bash
python main.py
```

The server will initialize the pre-trained `fdd_model.pkl` and `rul_model.pkl` bundles and bind to:  
`http://127.0.0.1:8000`

---

## 📖 Interactive Swagger & OpenAPI Documentation

Once the server is running, open your web browser:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Technical Specification**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can execute live requests, inspect request schemas, and validate response contracts directly from the browser.

---

## 🔌 API Endpoints

### 1. Health & Model Readiness
`GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "fdd_model_loaded": true,
  "rul_model_loaded": true,
  "feature_count": 82
}
```

---

### 2. Single Transformer Prediction
`POST /predict`

**Request Body**:
```json
{
  "asset_id": "2_trans_939.csv",
  "data": [
    {"H2": 0.00242, "CO": 0.01545, "C2H4": 0.00249, "C2H2": 0.00030},
    {"H2": 0.00242, "CO": 0.01547, "C2H4": 0.00249, "C2H2": 0.00030}
    // ... exactly 420 observations
  ]
}
```

**Response (`200 OK`)**:
```json
{
  "asset_id": "2_trans_939.csv",
  "predicted_fault": {
    "class": 4,
    "label": "Class 4: High-Energy Electrical Arcing / Flashover",
    "confidence": 0.9998
  },
  "fault_probabilities": {
    "class_1": 0.0001,
    "class_2": 0.0000,
    "class_3": 0.0001,
    "class_4": 0.9998
  },
  "predicted_rul": 417.27,
  "anomaly_score": 51.08,
  "risk_score": 85.12,
  "risk_category": "CRITICAL",
  "top_risk_factors": [
    "Predicted Class 4 arcing pattern (confidence 100.0%)",
    "Critically depleted remaining useful life (approx. 417 cycles)",
    "Elevated terminal Ethylene concentration (C2H4 = 0.00873)"
  ],
  "recommended_action": "Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review.",
  "explanation": "Model predicts Class 4: High-Energy Electrical Arcing / Flashover with high confidence (100.0%). Predicted Class 4 arcing pattern (confidence 100.0%) and Critically depleted remaining useful life (approx. 417 cycles) serve as strong contributing DGA signals consistent with the predicted fault pattern, while predicted RUL is approximately 417 cycles. Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review.",
  "model_metadata": {
    "fdd_model": "HistGradientBoostingClassifier",
    "rul_model": "HistGradientBoostingRegressor",
    "feature_count": 82
  },
  "data_scope": {
    "dga_used": true,
    "weather_used": false
  }
}
```

---

### 3. Batch Transformer Prioritization
`POST /predict/batch`

Accepts a list of transformer payloads:
```json
[
  { "asset_id": "2_trans_100.csv", "data": [...] },
  { "asset_id": "2_trans_939.csv", "data": [...] }
]
```

**Response (`200 OK`)**:  
Returns all processed predictions **sorted strictly by `risk_score` descending**, enabling dispatchers to prioritize critical units immediately.

---

## 🛠️ Developer Integration Guide (Frontend & Backend)

### Python Integration
```python
import requests
import pandas as pd

df = pd.read_csv("path/to/transformer.csv")
payload = {
    "asset_id": "TX-101",
    "data": df[['H2', 'CO', 'C2H4', 'C2H2']].to_dict(orient='records')
}
response = requests.post("http://127.0.0.1:8000/predict", json=payload)
data = response.json()
print("Risk Category:", data["risk_category"])
```

### Node.js / TypeScript Integration
```typescript
interface DGAObservation {
  H2: number;
  CO: number;
  C2H4: number;
  C2H2: number;
}

const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    asset_id: "TX-101",
    data: telemetrySequence // Array of 420 DGAObservation objects
  })
});
const result = await response.json();
console.log(`Asset Risk: ${result.risk_score} (${result.risk_category})`);
```

---

## 🧪 Running Automated Tests

Run the test suite verifying all 10 QA criteria (health, parity, schema, bounds, XAI):

```bash
python ml_model/api/test_api.py
```

---

## ⚠️ Important Boundary & Environmental Note
The model operates **exclusively on internal Dissolved Gas Analysis (DGA) sensor measurements** (`H2`, `CO`, `C2H4`, `C2H2`). Weather variables (ambient temperature, humidity, wind, lightning strikes) are **NOT currently included** because they are absent from the Kaggle dataset. Zero synthetic data or fake weather variables were fabricated.
