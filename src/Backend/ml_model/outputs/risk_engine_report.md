# Unified Transformer Risk Engine & Inference Architecture Report

**Project**: IBM Hackathon - Power Grid AI  
**Module**: Unified Real Data Inference & Equipment Risk Engine (TASK 4)  
**Trained Models Ingested**: `fdd_model.pkl`, `rul_model.pkl`  
**Output Prediction Catalog**: `ml_model/outputs/test_risk_predictions.csv`  

---

## 1. Pipeline Architecture

The unified transformer risk engine operates as a production inference service that ingests raw online transformer telemetry (or preprocessed feature matrices) and generates a multi-dimensional risk diagnostic for grid dispatchers and maintenance engineers.

```
       +-------------------------------------------------------------+
       |           Substation DGA Gas Telemetry (420 Steps)          |
       |       [H2, CO, C2H4, C2H2] from mineral insulating oil      |
       +------------------------------+------------------------------+
                                      |
                                      v
                       +------------------------------+
                       |    Feature Engineering       |
                       |    (82 Domain Features)      |
                       +--------------+---------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+-----------------------------+                           +-----------------------------+
|    FDD Classifier           |                           |    RUL Regressor            |
|    (HistGradientBoosting)   |                           |    (HistGradientBoosting)   |
|    Output: Fault Probs      |                           |    Output: Remaining Steps  |
+--------------+--------------+                           +--------------+--------------+
               |                                                         |
               +----------------------------+----------------------------+
                                            |
                                            v
                             +------------------------------+
                             |    Statistical Anomaly       |
                             |    Scoring Engine            |
                             |    (Empirical Cat-1 Baseline)|
                             +--------------+---------------+
                                            |
                                            v
                             +------------------------------+
                             |    Composite Risk Formula    |
                             | 40% FDD + 35% RUL + 25% Anom |
                             +--------------+---------------+
                                            |
                                            v
                             +------------------------------+
                             |    Operational Categorizer   |
                             |  LOW / MEDIUM / HIGH / CRIT  |
                             |  Action & Explanation Gen    |
                             +------------------------------+
```

---

## 2. Input Features

The engine operates on **82 domain features** derived from the 420-step DGA trajectory:
- **Per-Gas Trajectory Features** (17 features $\times$ 4 gases = 68): Terminal concentration, mean, std, min, max, median, first value, absolute change, slope, percentage change, range, coefficient of variation, and windowed statistics for the final 10%, 20%, and 50% intervals.
- **Total Dissolved Combustible Gas (TDCG)** (5 features): Terminal sum, mean, max, delta, and accumulation slope.
- **Normalized Gas Fractions & Ratios** (9 features): $\frac{H_2}{\text{TDCG}}$, $\frac{CO}{\text{TDCG}}$, $\frac{C_2H_4}{\text{TDCG}}$, $\frac{C_2H_2}{\text{TDCG}}$, $\frac{C_2H_2}{C_2H_4}$, $\frac{H_2}{C_2H_4}$, $\frac{CO}{H_2}$, and Duval hydrocarbon ratios.

---

## 3. FDD Prediction (Fault Detection & Diagnosis)

- **Model**: `HistGradientBoostingClassifier` with balanced class weights.
- **Inference Mode**: Soft probability output via `predict_proba()`.
- **Output Columns**:
  - `predicted_fault`: $\arg\max_k P(\text{Cat}_k) \in \{1, 2, 3, 4\}$
  - `fault_probability`: $\max_k P(\text{Cat}_k)$
  - `prob_fault_1`: Probability of normal degradation / cellulose aging
  - `prob_fault_2`: Probability of low-energy partial discharge (PD)
  - `prob_fault_3`: Probability of high-temperature thermal oil breakdown ($>700^\circ\text{C}$)
  - `prob_fault_4`: Probability of high-energy electrical arcing / flashover

---

## 4. RUL Prediction (Remaining Useful Life)

- **Model**: `HistGradientBoostingRegressor`.
- **Target Space**: Continuous remaining lifetime horizon $[362, 1093]$ cycles/steps.
- **Bounds Checking**: Enforces physical boundaries: $\text{predicted\_rul} = \text{clip}(\hat{y}, 362.0, 1093.0)$.
- **Output Column**: `predicted_rul`.

---

## 5. Statistical Anomaly Methodology

To guarantee complete transparency and avoid black-box synthetic anomaly artifacts, the anomaly score is computed as a robust standardized deviation against the **Category 1 (Normal Healthy Baseline)** empirical distribution established on the 1,705 training transformers:

$$\text{Raw Anomaly} = \sqrt{\frac{1}{|\mathcal{K}|} \sum_{j \in \mathcal{K}} \max\left(0, \frac{x_j - \mu_{1, j}}{\sigma_{1, j}}\right)^2}$$

Where $\mathcal{K} = \{\text{TDCG\_final}, \text{TDCG\_slope}, H_2\text{\_final}, C_2H_2\text{\_final}, C_2H_4\text{\_final}\}$.

The raw deviation is smoothly saturated onto a strict $0 - 100$ scale:

$$\text{Anomaly Score} = 100 \times \left(1 - \exp\left(-\frac{\text{Raw Anomaly}}{1.5}\right)\right)$$

- Healthy baseline units yield anomaly scores near $0 - 25$.
- Assets exhibiting severe gas surges or generation rate spikes score $>50 - 100$.

---

## 6. Composite Equipment Risk Score Formula

The overall operational risk is a weighted multi-factor composite normalized strictly to $0 - 100$:

$$\text{Risk Score} = 0.40 \times \text{FDD Risk} + 0.35 \times \text{RUL Risk} + 0.25 \times \text{Anomaly Score}$$

Where:
1. **FDD Fault Risk**: Probability-weighted physical hazard:
   $$\text{FDD Risk} = 0.0 \cdot P(\text{Cat}_1) + 55.0 \cdot P(\text{Cat}_2) + 80.0 \cdot P(\text{Cat}_3) + 100.0 \cdot P(\text{Cat}_4)$$
2. **RUL Degradation Risk**: Inverted linear lifetime depletion:
   $$\text{RUL Risk} = \text{clip}\left(\frac{1093.0 - \text{predicted\_rul}}{1093.0 - 362.0} \times 100.0, 0.0, 100.0\right)$$
3. **Anomaly Score**: As defined in Section 5.

*Note: This is an operational decision metric for prioritizing fleet inspection and dispatch, NOT a calibrated physical failure probability.*

---

## 7. Risk Thresholds & Categories

| Score Range | Operational Risk Category | Fleet Distribution in Test Set (900 Units) |
| :--- | :--- | :--- |
| **85 – 100** | **`CRITICAL`** | **1 unit (0.1%)** |
| **60 – 84** | **`HIGH`** | **63 units (7.0%)** |
| **35 – 59** | **`MEDIUM`** | **248 units (27.6%)** |
| **0 – 34** | **`LOW`** | **588 units (65.3%)** |

---

## 8. Maintenance Recommendations

Standardized actionable maintenance directives dispatched to utility asset managers:
- **CRITICAL**: *"Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review."*
- **HIGH**: *"Schedule urgent inspection/maintenance and closely monitor the transformer."*
- **MEDIUM**: *"Schedule planned inspection and continue monitoring."*
- **LOW**: *"Continue normal monitoring and scheduled maintenance."*

---

## 9. Example Real Predictions (Top 5 Ranked Assets)

| Rank | Asset ID | Risk Score | Category | Pred Fault | Prob | Pred RUL | Anomaly | Generated Explanation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `2_trans_939.csv` | **85.1** | **CRITICAL** | Class 4 | 100.0% | 417.3 | 51.1 | CRITICAL RISK (85.1/100): Imminent failure danger detected. Predicted fault is High-Energy Electrical Arcing / Severe Flashover with 100.0% confidence. Remaining Useful Life is low at 417 cycles. Key triggers: terminal Acetylene (C2H2=0.00037) indicates active high-energy electrical arcing, total combustible gas concentration is high (TDCG=0.0468). |
| **2** | `2_trans_1672.csv` | **83.7** | **HIGH** | Class 4 | 100.0% | 465.1 | 54.6 | HIGH RISK (83.7/100): Significant asset degradation detected. Predicted fault is High-Energy Electrical Arcing / Severe Flashover (100.0% confidence). Remaining Useful Life is estimated at 465 cycles. Key triggers: terminal Acetylene (C2H2=0.00039) indicates active high-energy electrical arcing, total combustible gas concentration is high (TDCG=0.0475). |
| **3** | `2_trans_1036.csv` | **82.9** | **HIGH** | Class 4 | 100.0% | 437.3 | 45.9 | HIGH RISK (82.9/100): Significant asset degradation detected. Predicted fault is High-Energy Electrical Arcing / Severe Flashover (100.0% confidence). Remaining Useful Life is estimated at 437 cycles. Key triggers: terminal Acetylene (C2H2=0.00035) indicates active high-energy electrical arcing. |
| **4** | `2_trans_1081.csv` | **81.5** | **HIGH** | Class 4 | 100.0% | 482.8 | 48.9 | HIGH RISK (81.5/100): Significant asset degradation detected. Predicted fault is High-Energy Electrical Arcing / Severe Flashover (100.0% confidence). Remaining Useful Life is estimated at 483 cycles. Key triggers: terminal Acetylene (C2H2=0.00036) indicates active high-energy electrical arcing. |
| **5** | `2_trans_2522.csv` | **81.0** | **HIGH** | Class 4 | 100.0% | 495.7 | 49.6 | HIGH RISK (81.0/100): Significant asset degradation detected. Predicted fault is High-Energy Electrical Arcing / Severe Flashover (100.0% confidence). Remaining Useful Life is estimated at 496 cycles. Key triggers: terminal Acetylene (C2H2=0.00037) indicates active high-energy electrical arcing. |

---

## 10. Known Limitations & Technical Scope

> [!IMPORTANT]
> **Environmental & Weather Data Limitation**  
> The current machine learning pipeline is trained and evaluated **exclusively on internal Dissolved Gas Analysis (DGA) chemical chromatography** (`H2`, `CO`, `C2H4`, `C2H2`).  
> Meteorological and environmental streams (ambient temperature, solar irradiance, relative humidity, precipitation, wind velocity, and storm lightning strikes) are **NOT currently included** because they are absent from the Kaggle competition dataset.  
> As instructed, no fake weather variables were fabricated. External weather APIs can be appended in future operational deployments when linked to substation geographic coordinates and timestamps.

- **RUL Right-Censoring**: Healthy units without active degradation cluster near the $1,093$-step ceiling, as the source telemetry does not track transformers past that horizon.
