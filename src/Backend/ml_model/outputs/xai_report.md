# Explainable Artificial Intelligence (XAI) Report: Transformer Risk Engine

**Project**: IBM Hackathon - Power Grid AI  
**Module**: Explainable AI (XAI) Subsystem (TASK 5)  
**Evaluated Artifacts**: `fdd_model.pkl`, `rul_model.pkl`, `test_risk_predictions.csv`  

---

## 1. Why Explainability is Critical in Power Grid Operations

High-voltage electrical substation power transformers represent multi-million dollar capital assets vital to grid stability. Catastrophic failures result in cascading blackouts, wildfire risks, and severe economic losses.

In mission-critical grid operations:
1. **Black-Box Skepticism**: Substation protection engineers and grid dispatchers will not de-energize or reroute multi-megawatt transmission lines based solely on opaque probability numbers without verifiable physical evidence.
2. **Actionable Root-Cause Triage**: A maintenance crew dispatched to a substation must know whether the risk is driven by localized high-temperature core overheating ($C_2H_4$), insulation paper breakdown ($CO$), or dielectric arcing flashover ($C_2H_2$).
3. **Auditable Decision-Support**: Compliance with utility regulatory bodies (NERC/FERC, IEEE, IEC) mandates transparent documentation for all automated load-shedding and capital maintenance prioritizations.

Our XAI subsystem provides dual-layer interpretability:
- **Global Interpretability**: Permutation feature importance ranking the primary chemical signals driving the models.
- **Local Interpretability**: Deterministic, case-by-case evidence identification detailing the top 3 physical risk factors and calibrated engineering explanations for each individual transformer.

---

## 2. Global Feature Importance: Fault Detection & Diagnosis (FDD)

Evaluated via 10-fold permutation importance on the independent 900-transformer test set:

| Rank | Feature Name | Permutation Importance (Mean $\Delta$ Macro F1) | Physical Diagnostic Meaning |
| :---: | :--- | :---: | :--- |
| **1** | `C2H4_TDCG_ratio` | **0.036667** | Ethylene gas fraction — Key IEEE/IEC Duval indicator for thermal oil breakdown ($>700^\circ\text{C}$) |
| **2** | `H2_TDCG_ratio` | **0.034111** | Hydrogen gas fraction — Primary discriminator isolating partial discharge/corona from thermal faults |
| **3** | `CO_TDCG_ratio` | **0.020222** | Carbon Monoxide fraction — Direct indicator of solid cellulose paper/pressboard degradation |
| **4** | `ratio_H2_C2H4` | **0.019333** | Hydrogen-to-Ethylene ratio — Disentangles electrical sparking from core overheating |
| **5** | `CO_recent_50pct_mean` | **0.004333** | Medium-term CO accumulation — Measures sustained thermal paper pyrolysis |
| **6** | `CO_cv` | **0.002556** | CO coefficient of variation — Quantifies instability in paper degradation rate |
| **7** | `CO_recent_20pct_mean` | **0.001444** | Recent CO generation — Captures accelerated paper insulation wear |
| **8** | `CO_pct_change` | **0.001000** | Net percentage surge in carbon monoxide over the observation window |
| **9** | `H2_cv` | **0.001000** | Hydrogen coefficient of variation — Reflects intermittent partial discharge sparking |
| **10** | `H2_std` | **0.000778** | Hydrogen standard deviation — Volatility in dielectric oil ionization |

*Refer to the high-resolution visualization at: `ml_model/outputs/fdd_feature_importance.png`.*

---

## 3. Global Feature Importance: Remaining Useful Life (RUL)

Evaluated via 10-fold permutation importance on the independent 900-transformer test set:

| Rank | Feature Name | Permutation Importance (Mean $\Delta R^2$) | Operational Lifetime Impact |
| :---: | :--- | :---: | :--- |
| **1** | `H2_final` | **0.253672** | Terminal Hydrogen concentration — Dominant signal for active ongoing dielectric stress |
| **2** | `C2H4_final` | **0.189864** | Terminal Ethylene level — Accumulation of high-temperature thermal cracking gas |
| **3** | `TDCG_final` | **0.173714** | Terminal Total Combustible Gas — Direct proxy for cumulative chemical aging |
| **4** | `C2H2_final` | **0.162759** | Terminal Acetylene level — High-energy arcing presence sharply abbreviates remaining life |
| **5** | `H2_recent_10pct_slope` | **0.077761** | Recent hydrogen surge rate — Fast gas generation indicates accelerating deterioration |
| **6** | `C2H4_recent_10pct_slope` | **0.053024** | Recent ethylene surge rate — Accelerated winding/core hot-spot expansion |
| **7** | `C2H2_recent_10pct_slope` | **0.041579** | Recent acetylene generation velocity — Indicates emerging or escalating electrical arcing |
| **8** | `CO_recent_10pct_slope` | **0.037870** | Recent CO generation slope — Accelerating paper insulation loss-of-life |
| **9** | `CO_final` | **0.023238** | Terminal Carbon Monoxide level — Overall state of solid insulation decomposition |
| **10** | `C2H4_recent_20pct_slope` | **0.015927** | Sustained 20% ethylene slope — Confirms persistent thermal runaway |

*Refer to the high-resolution visualization at: `ml_model/outputs/rul_feature_importance.png`.*

---

## 4. Local Transformer Explanations Across Risk Tiers

Every prediction in `ml_model/outputs/test_risk_predictions.csv` contains individual evidence signals and localized explanations:

### 4.1. CRITICAL Risk Example (Highest-Risk Fleet Asset: `2_trans_939.csv`)
- **Risk Score**: **85.1 / 100 [CRITICAL]**
- **Predicted Fault**: **Class 4** (High-Energy Electrical Arcing / Flashover) | **Confidence: 100.0%**
- **Predicted RUL**: **417.3 cycles** | **Anomaly Score**: **51.1 / 100**
- **Top 3 Risk Factors**:
  1. *Predicted Class 4 arcing pattern (confidence 100.0%)*
  2. *Critically depleted remaining useful life (approx. 417 cycles)*
  3. *Elevated terminal Ethylene concentration ($C_2H_4 = 0.00873$)*
- **Recommended Action**: *"Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review."*
- **XAI Explanation**:
  > *"Model predicts Class 4: High-Energy Electrical Arcing / Flashover with high confidence (100.0%). Predicted Class 4 arcing pattern (confidence 100.0%) and Critically depleted remaining useful life (approx. 417 cycles) serve as strong contributing DGA signals consistent with the predicted fault pattern, while predicted RUL is approximately 417 cycles. Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review."*

---

### 4.2. HIGH Risk Example (`2_trans_1672.csv`)
- **Risk Score**: **83.7 / 100 [HIGH]**
- **Predicted Fault**: **Class 4** (High-Energy Electrical Arcing) | **Confidence: 100.0%**
- **Predicted RUL**: **465.1 cycles** | **Anomaly Score**: **54.6 / 100**
- **Top 3 Risk Factors**:
  1. *Predicted Class 4 arcing pattern (confidence 100.0%)*
  2. *Elevated terminal Acetylene concentration ($C_2H_2 = 0.00046$)*
  3. *Critically depleted remaining useful life (approx. 465 cycles)*
- **Recommended Action**: *"Schedule urgent inspection/maintenance and closely monitor the transformer."*
- **XAI Explanation**:
  > *"Model identifies Class 4: High-Energy Electrical Arcing / Flashover pattern (100.0% confidence). Predicted Class 4 arcing pattern (confidence 100.0%) is an important indicator of significant asset degradation, accompanied by Elevated terminal Acetylene concentration (C2H2 = 0.00046), while estimated RUL is 465 cycles. Schedule urgent inspection/maintenance and closely monitor the transformer."*

---

### 4.3. MEDIUM Risk Example (`2_trans_81.csv`)
- **Risk Score**: **59.8 / 100 [MEDIUM]**
- **Predicted Fault**: **Class 4** (Emerging arcing signature) | **Confidence: 100.0%**
- **Predicted RUL**: **830.6 cycles** | **Anomaly Score**: **28.9 / 100**
- **Top 3 Risk Factors**:
  1. *Predicted Class 4 arcing pattern (confidence 100.0%)*
  2. *Elevated terminal Hydrogen concentration ($H_2 = 0.00351$)*
  3. *DGA combustible gas concentrations remain within standard limits*
- **Recommended Action**: *"Schedule planned inspection and continue monitoring."*
- **XAI Explanation**:
  > *"Model identifies Class 4: High-Energy Electrical Arcing / Flashover pattern (100.0% confidence). Predicted Class 4 arcing pattern (confidence 100.0%) represents an important indicator of moderate operational stress, with estimated remaining useful life at 831 cycles. Schedule planned inspection and continue monitoring."*

---

### 4.4. LOW Risk Example (`2_trans_2556.csv`)
- **Risk Score**: **35.0 / 100 [LOW]**
- **Predicted Fault**: **Class 1** (Normal Baseline Degradation) | **Confidence: 100.0%**
- **Predicted RUL**: **610.0 cycles** | **Anomaly Score**: **47.3 / 100**
- **Top 3 Risk Factors**:
  1. *Elevated terminal Ethylene concentration ($C_2H_4 = 0.01271$)*
  2. *Accelerating combustible gas generation slope*
  3. *Substantially degraded remaining useful life (610 cycles)*
- **Recommended Action**: *"Continue normal monitoring and scheduled maintenance."*
- **XAI Explanation**:
  > *"Model classifies asset as Class 1: Normal Baseline Degradation (Cellulose Aging) (100.0% confidence). Combustible gas measurements are consistent with the healthy Category 1 operating baseline, and predicted RUL is healthy at approximately 610 cycles. Continue normal monitoring and scheduled maintenance."*

---

## 5. Important Limitations & Operational Scope

> [!IMPORTANT]
> **Key Limitations for Engineering & Evaluation**:
> 1. **DGA Measurements Only**: The models operate exclusively on oil Dissolved Gas Analysis (`H2`, `CO`, `C2H4`, `C2H2`). Weather variables (ambient temperature, humidity, wind velocity, storm lightning strikes) are **NOT currently included** because they are absent from the Kaggle dataset.
> 2. **Evidentiary Attribution, Not Deterministic Causality**: Explanations highlight statistical feature importance and correlation consistent with DGA diagnostic guidelines (IEEE C57.104), but do not constitute guaranteed physical causation.
> 3. **RUL Target Censoring**: In the benchmark dataset, healthy transformers that did not fail within the observation window are right-censored at $1,093$ cycles/steps. Predictions near $1,093$ indicate that the asset will survive beyond the current monitoring horizon.
> 4. **Decision-Support Score**: The equipment risk score ($0 - 100$) is an operational heuristic for maintenance prioritization and inspection triage, not a calibrated physical probability of immediate failure.
