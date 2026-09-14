# Real Data Preprocessing & Feature Engineering Report

**Project**: IBM Hackathon - Power Grid AI Predictive Maintenance  
**Module**: ML / Model-Training Module  
**Task**: TASK 2 - Real Data Preprocessing (Strictly Real Kaggle Dataset)  
**Status**: Completed & Verified  

---

## 1. Overview & Dataset Accountability

The preprocessing pipeline has been rebuilt from scratch to operate strictly on the real transformer time-series and label files present in `ml_model/data/raw/`. 

- **Synthetic Data Generation**: **COMPLETELY ELIMINATED**. No artificial transformers (`TX-ALPHA`, etc.) or random values were used.
- **Weather Data**: **MARKED MISSING**. The real dataset contains solely Dissolved Gas Analysis (DGA) telemetry. No fake meteorological variables were synthesized.
- **Preserved Split**: The native train/test partition provided by the challenge was strictly maintained without reshuffling or data leakage.

| Split | Transformer Count | Raw Time Steps / Series | Total Observations | Raw Sensors | Final Engineered Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | **2,100** | Exactly 420 | 882,000 | 4 (`H2, CO, C2H4, C2H2`) | **82** |
| **TEST** | **900** | Exactly 420 | 378,000 | 4 (`H2, CO, C2H4, C2H2`) | **82** |
| **Total** | **3,000** | **420** | **1,260,000** | **4** | **82** |

---

## 2. Integrity Verification

Prior to transformation, automated assertions confirmed:
1. **100% ID Matching**: Every filename in `data_train/` (e.g., `2_trans_100.csv`) matches a unique key in `labels_fdd_train.csv` and `labels_rul_train.csv`.
2. **No Duplicates**: Zero duplicate IDs in either train or test sets.
3. **No Missing Values**: 0 nulls across all 1,260,000 raw sensor time-steps.
4. **Dimension Uniformity**: Exactly 420 rows $\times$ 4 columns in all 3,000 CSV files.

---

## 3. Feature Engineering Breakdown (82 Domain Features)

Each transformer's 420-step trajectory was condensed into one feature vector consisting of:

### 3.1. Individual Gas Dynamics (17 features $\times$ 4 gases = 68 features)
Calculated separately for **`H2`**, **`CO`**, **`C2H4`**, and **`C2H2`**:
- **Final Value**: `$g_final` (terminal state at $t=419$).
- **Basic Statistics**: `$g_mean`, `$g_std`, `$g_min`, `$g_max`, `$g_median` over the 420 steps.
- **Trend Dynamics**:
  - `$g_first`: Initial state at $t=0$.
  - `$g_abs_change`: Absolute net difference ($final - first$).
  - `$g_slope`: Overall trajectory slope ($\frac{final - first}{419}$).
  - `$g_pct_change`: Relative percentage change ($\frac{final - first}{first + \epsilon}$).
- **Variability**:
  - `$g_range`: Absolute range ($max - min$).
  - `$g_cv`: Coefficient of Variation ($\frac{std}{mean + \epsilon}$).
- **Recent Behavior**:
  - `$g_recent_10pct_mean`: Mean of the final 42 steps.
  - `$g_recent_20pct_mean`: Mean of the final 84 steps.
  - `$g_recent_50pct_mean`: Mean of the final 210 steps.
  - `$g_recent_10pct_slope`: Rate of change over the final 42 steps.
  - `$g_recent_20pct_slope`: Rate of change over the final 84 steps.

### 3.2. DGA-Derived & Multi-Gas Aggregate Features (14 features)
- **Total Dissolved Combustible Gas (TDCG)**:
  - $\text{TDCG} = H_2 + CO + C_2H_4 + C_2H_2$
  - `TDCG_final`, `TDCG_mean`, `TDCG_max`, `TDCG_abs_change`, `TDCG_slope`.
- **Normalized Gas Fractions (Relative Concentrations)**:
  - `H2_TDCG_ratio` = $\frac{H_2}{\text{TDCG} + \epsilon}$
  - `CO_TDCG_ratio` = $\frac{CO}{\text{TDCG} + \epsilon}$
  - `C2H4_TDCG_ratio` = $\frac{C_2H_4}{\text{TDCG} + \epsilon}$
  - `C2H2_TDCG_ratio` = $\frac{C_2H_2}{\text{TDCG} + \epsilon}$
- **Pairwise Diagnostic Ratios (IEEE C57.104 / IEC 60599)**:
  - `ratio_C2H2_C2H4` = $\frac{C_2H_2}{C_2H_4 + \epsilon}$ (Arcing vs Thermal Fault discriminator)
  - `ratio_H2_C2H4` = $\frac{H_2}{C_2H_4 + \epsilon}$ (Corona/PD vs Thermal Oil Breakdown)
  - `ratio_CO_H2` = $\frac{CO}{H_2 + \epsilon}$ (Cellulose Pyrolysis vs Low-energy Sparking)
- **Duval Hydrocarbon Proportions**:
  - `C2H4_hydrocarbon_ratio` = $\frac{C_2H_4}{C_2H_4 + C_2H_2 + \epsilon}$
  - `C2H2_hydrocarbon_ratio` = $\frac{C_2H_2}{C_2H_4 + C_2H_2 + \epsilon}$

*Note on Safeguards: All denominators are strictly protected with $\epsilon = 1.0 \times 10^{-12}$ to prevent division-by-zero or numerical instability.*

---

## 4. Data Quality & Outlier Handling

- **Missing Values**: 0 NaNs or missing records detected.
- **Infinite Values**: 0 infinite values found after all ratio calculations.
- **Outlier Policy**: Extreme gas surges are **NOT clipped or deleted**. In transformer diagnostic engineering, extreme spikes in $C_2H_2$ or $C_2H_4$ represent critical physical fault phenomena (electrical flashover, localized thermal runaway). Deleting or Winsorizing these spikes would destroy the primary discriminative signal for FDD fault categories 2, 3, and 4.
- **Constant Features**: 0 constant features found; all 82 features exhibit non-zero variance.

---

## 5. Leakage Prevention Architecture

Strict guardrails were implemented to eliminate all vectors of data leakage:
1. **Target Segregation**:
   - `target_fdd` (`category`) is strictly excluded from feature matrices.
   - `target_rul` (`predicted`) is strictly excluded from feature matrices.
   - `train_fdd.csv` and `test_fdd.csv` contain solely the 82 sensor features and `target_fdd` (zero RUL presence).
   - `train_rul.csv` and `test_rul.csv` contain solely the 82 sensor features and `target_rul` (zero FDD presence).
2. **Metadata Insulation**:
   - `asset_id` is retained purely as an external join identifier and is excluded from model feature sets.
3. **Split Insulation**:
   - Preprocessing scaling parameters (`StandardScaler`) were fitted **exclusively on the training feature matrix**. Test data was never observed during fitting.
   - Scaler bundle serialized to `ml_model/models/scaler.pkl`.

---

## 6. Generated Processed Datasets

All processed artifacts are saved under `ml_model/data/processed/`:

| Dataset File | Rows | Columns | Description |
| :--- | :--- | :--- | :--- |
| `train_features.csv` | 2,100 | 85 | Master training dataset (`asset_id` + 82 features + `target_fdd` + `target_rul`) |
| `test_features.csv` | 900 | 85 | Master testing dataset (`asset_id` + 82 features + `target_fdd` + `target_rul`) |
| `train_fdd.csv` | 2,100 | 84 | FDD-specific training set (`asset_id` + 82 features + `target_fdd`) |
| `test_fdd.csv` | 900 | 84 | FDD-specific testing set (`asset_id` + 82 features + `target_fdd`) |
| `train_rul.csv` | 2,100 | 84 | RUL-specific training set (`asset_id` + 82 features + `target_rul`) |
| `test_rul.csv` | 900 | 84 | RUL-specific testing set (`asset_id` + 82 features + `target_rul`) |

---

## 7. Assumptions & Limitations

- **Sampling Uniformity**: The 420 time steps are assumed to be regularly sampled periodic observations (e.g., hourly online DGA telemetry).
- **Missing Weather Streams**: Ambient temperature, solar radiation, and humidity are absent from the raw dataset. External weather variables must be integrated via regional substation mappings if needed for subsequent dispatch simulations.
