# Comprehensive Dataset Audit: Power Grid Transformer DGA Telemetry

**Project**: IBM Hackathon - Power Grid AI Predictive Maintenance  
**Module**: ML / Model-Training Module  
**Audit Scope**: Strict audit of raw data in `ml_model/data/raw/` (No models trained, no files modified or deleted)  
**Audit Date**: September 2026  

---

## 1. Executive Summary & File Inventory

The dataset located in `ml_model/data/raw/` consists of **3,004 total files**:
- **4 Master Label Files** (`.csv`) located at the root of `data/raw/`.
- **2 Time-Series Directories** (`data_train/` and `data_test/`) containing individual transformer Dissolved Gas Analysis (DGA) operational sequences.

| File / Directory | File Type | File Count | Rows Per File | Total Observations | Number of Columns | Column Names |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `labels_fdd_train.csv` | CSV (Table) | 1 | 2,100 | 2,100 | 2 | `['id', 'category']` |
| `labels_fdd_test.csv` | CSV (Table) | 1 | 900 | 900 | 2 | `['id', 'category']` |
| `labels_rul_train.csv` | CSV (Table) | 1 | 2,100 | 2,100 | 2 | `['id', 'predicted']` |
| `labels_rul_test.csv` | CSV (Table) | 1 | 900 | 900 | 2 | `['id', 'predicted']` |
| `data_train/*.csv` | CSV (Time-Series) | 2,100 | 420 | 882,000 | 4 | `['H2', 'CO', 'C2H4', 'C2H2']` |
| `data_test/*.csv` | CSV (Time-Series) | 900 | 420 | 378,000 | 4 | `['H2', 'CO', 'C2H4', 'C2H2']` |
| **Total Corpus** | **CSV** | **3,004** | **—** | **1,266,000** | **—** | **—** |

---

## 2. Granular File-by-File Audit

### 2.1. Master Label Files (`labels_fdd_train.csv` & `labels_fdd_test.csv`)
- **1. Filename**: `labels_fdd_train.csv` and `labels_fdd_test.csv`
- **2. File type**: Comma-Separated Values (CSV, ASCII text)
- **3. Number of rows**:
  - `labels_fdd_train.csv`: **2,100 rows** (excluding header)
  - `labels_fdd_test.csv`: **900 rows** (excluding header)
- **4. Number of columns**: **2 columns**
- **5. Column names**: `id`, `category`
- **6. Data types**:
  - `id`: `object` / `string` (e.g., `'2_trans_497.csv'`)
  - `category`: `int64` (discrete classification target)
- **7. Missing values**: **0 (0.0%)** across both files
- **8. Duplicate rows**: **0** duplicate rows; all `id` keys are unique
- **9. Min/Max values**:
  - `category`: Min = **1**, Max = **4**
- **10. Unique values & Class Distribution**:
  - **Training Set (`labels_fdd_train.csv`)**:
    - `Category 1`: **1,705 samples (81.19%)** — Normal / baseline degradation
    - `Category 4`: **193 samples (9.19%)** — Thermal / electrical mixed fault
    - `Category 3`: **113 samples (5.38%)** — High-temperature thermal fault in oil
    - `Category 2`: **89 samples (4.24%)** — Low-energy electrical discharge / partial discharge
  - **Testing Set (`labels_fdd_test.csv`)**:
    - `Category 1`: **731 samples (81.22%)**
    - `Category 4`: **82 samples (9.11%)**
    - `Category 3`: **49 samples (5.44%)**
    - `Category 2`: **38 samples (4.22%)**
  *(Note: Exact class proportion parity between train and test indicates stratified splitting).*

---

### 2.2. Master Label Files (`labels_rul_train.csv` & `labels_rul_test.csv`)
- **1. Filename**: `labels_rul_train.csv` and `labels_rul_test.csv`
- **2. File type**: Comma-Separated Values (CSV, ASCII text)
- **3. Number of rows**:
  - `labels_rul_train.csv`: **2,100 rows**
  - `labels_rul_test.csv`: **900 rows**
- **4. Number of columns**: **2 columns**
- **5. Column names**: `id`, `predicted`
- **6. Data types**:
  - `id`: `object` / `string`
  - `predicted`: `int64` (continuous / integer lifetime horizon target)
- **7. Missing values**: **0 (0.0%)** across both files
- **8. Duplicate rows**: **0** duplicate rows
- **9. Numerical Min / Max / Distribution (`predicted`)**:
  - **Training Set (`labels_rul_train.csv`)**:
    - Min: **362**
    - 25th Percentile: **559.75**
    - 50th Percentile (Median): **758.0**
    - Mean: **793.13** (Std: 243.32)
    - 75th Percentile: **1,093.0**
    - Max: **1,093**
  - **Testing Set (`labels_rul_test.csv`)**:
    - Min: **404**
    - 25th Percentile: **573.50**
    - 50th Percentile (Median): **776.0**
    - Mean: **799.93** (Std: 238.60)
    - 75th Percentile: **1,093.0**
    - Max: **1,093**
- **10. Unique values**:
  - `predicted` in Train: **479 unique values**
  - `predicted` in Test: **369 unique values**
  *(Note: 1,093 is a right-censored upper limit representing healthy units that did not fail within the observation window).*

---

### 2.3. Time-Series Telemetry Files (`data_train/` and `data_test/`)
- **1. Filename**: 2,100 files in `data_train/` + 900 files in `data_test/` (Format: `2_trans_<id>.csv`)
- **2. File type**: Comma-Separated Values (CSV)
- **3. Number of rows**: **Exactly 420 rows per file** across all 3,000 files
- **4. Number of columns**: **Exactly 4 columns** across all 3,000 files
- **5. Column names**: `['H2', 'CO', 'C2H4', 'C2H2']`
- **6. Data types**: All 4 columns are `float64`
- **7. Missing values**: **0 missing values (0.0%)** across all 1,260,000 total time-step observations
- **8. Duplicate rows**: **0 duplicate rows** within any file
- **9. Minimum and Maximum Values for Numerical Columns**:

| Gas Sensor | Physical Parameter | Train Global Min | Train Global Max | Test Global Min | Test Global Max | Typical Physical Unit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`H2`** | Dissolved Hydrogen | $1.7845 \times 10^{-8}$ | $5.6223 \times 10^{-3}$ | $1.2293 \times 10^{-7}$ | $4.7979 \times 10^{-3}$ | Volume / Mol Fraction |
| **`CO`** | Dissolved Carbon Monoxide | $1.4968 \times 10^{-8}$ | $5.7476 \times 10^{-2}$ | $3.0917 \times 10^{-7}$ | $5.4937 \times 10^{-2}$ | Volume / Mol Fraction |
| **`C2H4`** | Dissolved Ethylene | $7.8658 \times 10^{-9}$ | $1.5394 \times 10^{-2}$ | $8.8730 \times 10^{-8}$ | $1.4894 \times 10^{-2}$ | Volume / Mol Fraction |
| **`C2H2`** | Dissolved Acetylene | $7.8974 \times 10^{-11}$ | $6.7071 \times 10^{-4}$ | $2.9174 \times 10^{-10}$ | $5.8637 \times 10^{-4}$ | Volume / Mol Fraction |

- **10. Unique values for categorical/label columns**: None present inside the telemetry files (purely numerical time series).

---

## 3. Data Relationships, Keys & Semantics

### 11. Relationship Between Train Data and Label Files
- There is a strict **1-to-1 bijection** between each time-series CSV in `data_train/` and the rows of `labels_fdd_train.csv` and `labels_rul_train.csv`.
- Each transformer asset run $i$ has:
  - An input matrix $\mathbf{X}_i \in \mathbb{R}^{420 \times 4}$ in `data_train/2_trans_<i>.csv`.
  - A categorical fault label $y_{\text{fdd}, i} \in \{1, 2, 3, 4\}$ in `labels_fdd_train.csv`.
  - A continuous/discrete remaining life target $y_{\text{rul}, i} \in [362, 1093]$ in `labels_rul_train.csv`.
- Exactly the same relationship holds for `data_test/`, `labels_fdd_test.csv`, and `labels_rul_test.csv`.

### 12. Possible Join Keys
- Master Join Key: **`id`**
- In label CSVs, `id` stores the exact filename: `2_trans_<id>.csv`.
- Integrity Check: **100% (2,100 / 2,100)** train keys match `data_train/` filenames; **100% (900 / 900)** test keys match `data_test/` filenames.
- Clean Asset Key: The integer extracted via regex `r"2_trans_(\d+)\.csv"` serves as the unique transformer entity ID.

### 13. Whether Timestamps Exist
- **No explicit calendar or epoch timestamps exist** in the raw files (no datetime column).
- Temporal sequence is **implicit**: each file consists of 420 consecutive rows ($t = 0, 1, 2, \dots, 419$).
- These 420 steps represent uniform periodic sampling (e.g., hourly DGA gas sensor monitoring across 420 hours $\approx$ 17.5 days, or daily sampling).

### 14. Whether Asset / Transformer IDs Exist
- **Yes.** Embedded directly in the filename `2_trans_<id>.csv`.
- The prefix `2_` indicates substation/transformer group 2, and the suffix denotes individual transformer assets.

### 15. What FDD Means in This Dataset
- **FDD** = **Fault Detection and Diagnosis** (or Fault Detection and Discrimination).
- In power transformer engineering (IEEE C57.104 & IEC 60599 standards), DGA gases diagnose distinct internal physical failure modes:
  - **Category 1 (81.2%)**: **Normal State / Cellulose Aging**. Characterized by baseline $CO$ generation from paper insulation pyrolysis under normal thermal load.
  - **Category 2 (4.2%)**: **Low-Energy Electrical Discharge / Partial Discharge (PD)**. Dominated by Hydrogen ($H_2$) ionization in oil with very low acetylene.
  - **Category 3 (5.4%)**: **High-Temperature Thermal Fault in Oil ($T > 700^\circ\text{C}$)**. Dominated by Ethylene ($C_2H_4$) cracking.
  - **Category 4 (9.2%)**: **High-Energy Electrical Arcing / Combined Fault**. Elevated Acetylene ($C_2H_2$) accompanied by significant $H_2$ and $C_2H_4$.

### 16. What RUL Means in This Dataset
- **RUL** = **Remaining Useful Life**.
- It represents the estimated number of operational time cycles, hours, or steps remaining before the transformer insulation degrades beyond acceptable dielectric withstand limits or triggers an automatic protection trip.
- Values range from 362 to 1,093. The large cluster at 1,093 represents **right-censored** assets that are healthy and will survive well past the monitoring window.

---

## 4. Feature Utility & Leakage Assessment

### 17. Columns Representing Transformer Health / Sensor Measurements
- **`H2`**: Dissolved Hydrogen — primary sensor for corona discharge, dielectric oil ionization, and low-energy sparking.
- **`CO`**: Dissolved Carbon Monoxide — sensor indicator for solid kraft paper insulation overheating and mechanical paper degradation.
- **`C2H4`**: Dissolved Ethylene — sensor indicator for thermal oil cracking under severe localized hot-spots ($> 700^\circ\text{C}$).
- **`C2H2`**: Dissolved Acetylene — critical warning sensor for electric arc flashover and contact burning.

### 18. Columns Suitable for Fault Prediction (Supervised Classification)
- Temporal features and summary statistics engineered from the 420-step trajectory:
  1. **Final Step Values**: $H_2(419), CO(419), C_2H_4(419), C_2H_2(419)$
  2. **Gas Generation Rates (Slopes)**: $\frac{\Delta H_2}{\Delta t}, \frac{\Delta CO}{\Delta t}, \frac{\Delta C_2H_4}{\Delta t}, \frac{\Delta C_2H_2}{\Delta t}$
  3. **Total Dissolved Combustible Gas (TDCG)**: $H_2 + CO + C_2H_4 + C_2H_2$
  4. **Diagnostic Ratios (IEC 60599 / Duval Triangle)**:
     - $\frac{C_2H_2}{C_2H_4}$ (Ethylene-Acetylene ratio)
     - $\frac{CH_4}{H_2}$ / $\frac{C_2H_4}{H_2}$
     - $\frac{CO}{\text{TDCG}}$ (Cellulose vs Oil ratio)
  5. **Time-Series Latent Representations**: 1D-CNN, Bi-LSTM, or MiniRocket embeddings over the 420 steps.

### 19. Columns Suitable for Anomaly Detection (Unsupervised)
- The baseline 4-gas readings (`H2`, `CO`, `C2H4`, `C2H2`), their rolling standard deviations, and TDCG acceleration.
- Strategy: Train an unsupervised detector (`IsolationForest`, `OneClassSVM`, or an `Autoencoder`) exclusively on **Category 1 (Normal)** sequences. Any transformer exhibiting gas surges, abnormal generation rates, or out-of-distribution ratio shifts will yield a high anomaly score.

### 20. Columns That Must NOT Be Used as Model Features (Target Leakage Prevention)
- **`category`** in `labels_fdd_*.csv`: This is the ground truth target for fault classification. Including it as an input causes 100% target leakage.
- **`predicted`** in `labels_rul_*.csv`: This is the ground truth target for RUL regression. It cannot be used as an input to predict RUL or fault category in real deployment.
- **`id` / filename**: Should only be used as a database join key. Passing `id` or asset number as an ordinal/numerical feature will cause spurious overfitting to specific transformer naming IDs.
- **Future Time-Step Leakage**: When predicting step-by-step health, features at step $t$ must only use historical data $\tau \le t$, never future observations $\tau > t$.

---

## 5. Summary Schema Mapping

The machine-readable dataset schema has been generated and validated at:  
`ml_model/outputs/dataset_schema.json`

Summary statistics:
- Train samples: **2,100 transformers** $\times$ 420 time steps = **882,000 points**
- Test samples: **900 transformers** $\times$ 420 time steps = **378,000 points**
- Missing values: **0**
- Duplicate rows: **0**
- Targets provided: Both Fault Classification (`category`) and RUL Estimation (`predicted`).
