"""
Data Preprocessing & Feature Engineering Module
Power Grid AI - Real Transformer DGA Telemetry Pipeline

Strictly processes the actual Kaggle dataset in ml_model/data/raw/:
- data_train/ (2,100 CSVs, each 420 rows x 4 DGA gases: H2, CO, C2H4, C2H2)
- data_test/  (900 CSVs, each 420 rows x 4 DGA gases: H2, CO, C2H4, C2H2)
- labels_fdd_train.csv & labels_fdd_test.csv (category target: 1, 2, 3, 4)
- labels_rul_train.csv & labels_rul_test.csv (predicted RUL target: 362 - 1093)

NO SYNTHETIC DATA IS GENERATED.
NO FAKE WEATHER DATA IS FABRICATED.
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-12
EXPECTED_ROWS = 420
EXPECTED_COLS = ['H2', 'CO', 'C2H4', 'C2H2']


def verify_raw_data_integrity(raw_dir: str):
    """
    Verifies 100% integrity of raw input files before any transformation.
    Halts execution if any structural assumption is violated.
    """
    print("[VERIFY] Auditing raw data files integrity...")
    train_dir = os.path.join(raw_dir, 'data_train')
    test_dir = os.path.join(raw_dir, 'data_test')
    
    fdd_train_path = os.path.join(raw_dir, 'labels_fdd_train.csv')
    fdd_test_path = os.path.join(raw_dir, 'labels_fdd_test.csv')
    rul_train_path = os.path.join(raw_dir, 'labels_rul_train.csv')
    rul_test_path = os.path.join(raw_dir, 'labels_rul_test.csv')
    
    # 1. Check label file existence
    for path in [fdd_train_path, fdd_test_path, rul_train_path, rul_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Required label file missing: {path}")
            
    fdd_train = pd.read_csv(fdd_train_path)
    fdd_test = pd.read_csv(fdd_test_path)
    rul_train = pd.read_csv(rul_train_path)
    rul_test = pd.read_csv(rul_test_path)
    
    # 2. Check shapes
    if len(fdd_train) != 2100 or len(rul_train) != 2100:
        raise ValueError(f"[CRITICAL ERROR] Unexpected train label row count: FDD={len(fdd_train)}, RUL={len(rul_train)}")
    if len(fdd_test) != 900 or len(rul_test) != 900:
        raise ValueError(f"[CRITICAL ERROR] Unexpected test label row count: FDD={len(fdd_test)}, RUL={len(rul_test)}")
        
    # 3. Check for duplicates in label IDs
    if fdd_train['id'].duplicated().any() or rul_train['id'].duplicated().any():
        raise ValueError("[CRITICAL ERROR] Duplicate IDs detected in training label files!")
    if fdd_test['id'].duplicated().any() or rul_test['id'].duplicated().any():
        raise ValueError("[CRITICAL ERROR] Duplicate IDs detected in testing label files!")
        
    # 4. Check matching IDs between FDD and RUL
    if set(fdd_train['id']) != set(rul_train['id']):
        raise ValueError("[CRITICAL ERROR] Discrepancy between FDD and RUL training IDs!")
    if set(fdd_test['id']) != set(rul_test['id']):
        raise ValueError("[CRITICAL ERROR] Discrepancy between FDD and RUL testing IDs!")
        
    # 5. Check time-series files exist for all IDs
    train_files = set(os.listdir(train_dir))
    test_files = set(os.listdir(test_dir))
    
    missing_train = set(fdd_train['id']) - train_files
    missing_test = set(fdd_test['id']) - test_files
    
    if missing_train:
        raise FileNotFoundError(f"[CRITICAL ERROR] {len(missing_train)} train CSV files missing from {train_dir}")
    if missing_test:
        raise FileNotFoundError(f"[CRITICAL ERROR] {len(missing_test)} test CSV files missing from {test_dir}")
        
    print(f"[VERIFY] Integrity check PASSED: 2,100 Train & 900 Test transformers verified with 100% label matching.")


def extract_transformer_features(df_series: pd.DataFrame, asset_id: str) -> dict:
    """
    Extracts physically meaningful DGA features from a single transformer's 420-step telemetry.
    Strictly verifies row count and columns.
    """
    if len(df_series) != EXPECTED_ROWS:
        raise ValueError(f"Transformer {asset_id} has {len(df_series)} rows; expected exactly {EXPECTED_ROWS} rows.")
    if list(df_series.columns) != EXPECTED_COLS:
        raise ValueError(f"Transformer {asset_id} columns {list(df_series.columns)} do not match expected {EXPECTED_COLS}")
        
    feats = {'asset_id': asset_id}
    n_rows = len(df_series)
    
    # 1. Total Dissolved Combustible Gas (TDCG) time series
    tdcg_series = df_series[EXPECTED_COLS].sum(axis=1)
    
    # 2. Per-gas time-series statistics and dynamics
    for g in EXPECTED_COLS:
        s = df_series[g]
        val_first = float(s.iloc[0])
        val_last = float(s.iloc[-1])
        val_mean = float(s.mean())
        val_std = float(s.std())
        val_min = float(s.min())
        val_max = float(s.max())
        val_med = float(s.median())
        
        abs_change = val_last - val_first
        slope = abs_change / max(1, n_rows - 1)
        pct_change = abs_change / (val_first + EPSILON)
        val_range = val_max - val_min
        cv = val_std / (val_mean + EPSILON)
        
        # Recent behavior windows (final 10%, 20%, 50% of the 420 steps)
        r10 = s.iloc[-int(n_rows * 0.10):]
        r20 = s.iloc[-int(n_rows * 0.20):]
        r50 = s.iloc[-int(n_rows * 0.50):]
        
        # A. Final Value
        feats[f'{g}_final'] = val_last
        # B. Basic Statistics
        feats[f'{g}_mean'] = val_mean
        feats[f'{g}_std'] = val_std
        feats[f'{g}_min'] = val_min
        feats[f'{g}_max'] = val_max
        feats[f'{g}_median'] = val_med
        # C. Trend Information
        feats[f'{g}_first'] = val_first
        feats[f'{g}_abs_change'] = abs_change
        feats[f'{g}_slope'] = slope
        feats[f'{g}_pct_change'] = pct_change
        # D. Variability
        feats[f'{g}_range'] = val_range
        feats[f'{g}_cv'] = cv
        # E. Recent Behavior
        feats[f'{g}_recent_10pct_mean'] = float(r10.mean())
        feats[f'{g}_recent_20pct_mean'] = float(r20.mean())
        feats[f'{g}_recent_50pct_mean'] = float(r50.mean())
        feats[f'{g}_recent_10pct_slope'] = float((r10.iloc[-1] - r10.iloc[0]) / max(1, len(r10) - 1))
        feats[f'{g}_recent_20pct_slope'] = float((r20.iloc[-1] - r20.iloc[0]) / max(1, len(r20) - 1))
        
    # 3. TDCG Aggregate Features
    feats['TDCG_final'] = float(tdcg_series.iloc[-1])
    feats['TDCG_mean'] = float(tdcg_series.mean())
    feats['TDCG_max'] = float(tdcg_series.max())
    feats['TDCG_abs_change'] = float(tdcg_series.iloc[-1] - tdcg_series.iloc[0])
    feats['TDCG_slope'] = feats['TDCG_abs_change'] / max(1, n_rows - 1)
    
    # 4. Standard DGA Ratios and Gas Fractions (with strict epsilon safeguards)
    final_tdcg = feats['TDCG_final'] + EPSILON
    for g in EXPECTED_COLS:
        feats[f'{g}_TDCG_ratio'] = feats[f'{g}_final'] / final_tdcg
        
    feats['ratio_C2H2_C2H4'] = feats['C2H2_final'] / (feats['C2H4_final'] + EPSILON)
    feats['ratio_H2_C2H4'] = feats['H2_final'] / (feats['C2H4_final'] + EPSILON)
    feats['ratio_CO_H2'] = feats['CO_final'] / (feats['H2_final'] + EPSILON)
    
    # 5. Hydrocarbon Proportions (Duval component)
    hc_tot = feats['C2H4_final'] + feats['C2H2_final'] + EPSILON
    feats['C2H4_hydrocarbon_ratio'] = feats['C2H4_final'] / hc_tot
    feats['C2H2_hydrocarbon_ratio'] = feats['C2H2_final'] / hc_tot
    
    return feats


def process_split(data_dir: str, fdd_df: pd.DataFrame, rul_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Extracts features for all transformers in a split and merges verified targets.
    """
    print(f"[PROCESS] Processing {split_name} split ({len(fdd_df)} transformers)...")
    records = []
    
    # Merge label targets on id
    labels = pd.merge(fdd_df, rul_df, on='id')
    
    for idx, row in labels.iterrows():
        asset_id = row['id']
        file_path = os.path.join(data_dir, asset_id)
        df_series = pd.read_csv(file_path)
        
        feats = extract_transformer_features(df_series, asset_id)
        feats['target_fdd'] = int(row['category'])
        feats['target_rul'] = int(row['predicted'])
        records.append(feats)
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(labels):
            print(f"  Processed {idx + 1}/{len(labels)} {split_name} transformers...")
            
    df_out = pd.DataFrame(records)
    return df_out


def run_preprocessing_pipeline():
    """
    Orchestrates the end-to-end real data preprocessing pipeline.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Strict raw data verification
    verify_raw_data_integrity(raw_dir)
    
    # 2. Load ground-truth labels
    fdd_train = pd.read_csv(os.path.join(raw_dir, 'labels_fdd_train.csv'))
    fdd_test = pd.read_csv(os.path.join(raw_dir, 'labels_fdd_test.csv'))
    rul_train = pd.read_csv(os.path.join(raw_dir, 'labels_rul_train.csv'))
    rul_test = pd.read_csv(os.path.join(raw_dir, 'labels_rul_test.csv'))
    
    # 3. Extract features preserving the train/test split
    train_df = process_split(os.path.join(raw_dir, 'data_train'), fdd_train, rul_train, 'TRAIN')
    test_df = process_split(os.path.join(raw_dir, 'data_test'), fdd_test, rul_test, 'TEST')
    
    # 4. Separate feature names (excluding metadata and targets)
    non_feature_cols = ['asset_id', 'target_fdd', 'target_rul']
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
    
    print(f"\n[FEATURES] Total engineered features per transformer: {len(feature_cols)}")
    
    # 5. Data Quality Checks
    print("\n[QUALITY CHECK]:")
    for name, df in [('Train', train_df), ('Test', test_df)]:
        n_nan = df[feature_cols].isnull().sum().sum()
        n_inf = np.isinf(df[feature_cols].values).sum()
        print(f"  {name} feature matrix NaN count: {n_nan}")
        print(f"  {name} feature matrix Inf count: {n_inf}")
        if n_nan > 0 or n_inf > 0:
            raise ValueError(f"[DATA QUALITY ERROR] Invalid numerical values found in {name} features!")
            
    # 6. Fit Scaler ONLY on Train Feature Matrix (Strict Leakage Prevention)
    print("\n[SCALING] Fitting StandardScaler on training feature matrix ONLY...")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    scaler_bundle = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'non_feature_cols': non_feature_cols,
        'n_features': len(feature_cols),
        'expected_rows_per_series': EXPECTED_ROWS
    }
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler_bundle, scaler_path)
    print(f"[SCALING] Preprocessing scaler bundle saved to {scaler_path}")
    
    # 7. Save Master Processed Files
    train_features_path = os.path.join(processed_dir, 'train_features.csv')
    test_features_path = os.path.join(processed_dir, 'test_features.csv')
    train_df.to_csv(train_features_path, index=False)
    test_df.to_csv(test_features_path, index=False)
    print(f"[SAVE] train_features.csv saved: {train_df.shape}")
    print(f"[SAVE] test_features.csv saved: {test_df.shape}")
    
    # 8. Save Model-Specific Datasets with Segregated Targets (Zero Cross-Target Leakage)
    # FDD datasets: asset_id, feature_cols, target_fdd
    train_fdd = train_df[['asset_id'] + feature_cols + ['target_fdd']]
    test_fdd = test_df[['asset_id'] + feature_cols + ['target_fdd']]
    train_fdd.to_csv(os.path.join(processed_dir, 'train_fdd.csv'), index=False)
    test_fdd.to_csv(os.path.join(processed_dir, 'test_fdd.csv'), index=False)
    print(f"[SAVE] train_fdd.csv saved: {train_fdd.shape}")
    print(f"[SAVE] test_fdd.csv saved: {test_fdd.shape}")
    
    # RUL datasets: asset_id, feature_cols, target_rul
    train_rul = train_df[['asset_id'] + feature_cols + ['target_rul']]
    test_rul = test_df[['asset_id'] + feature_cols + ['target_rul']]
    train_rul.to_csv(os.path.join(processed_dir, 'train_rul.csv'), index=False)
    test_rul.to_csv(os.path.join(processed_dir, 'test_rul.csv'), index=False)
    print(f"[SAVE] train_rul.csv saved: {train_rul.shape}")
    print(f"[SAVE] test_rul.csv saved: {test_rul.shape}")
    
    # 9. Print Comprehensive Summary
    print("\n" + "="*60)
    print("TASK 2 PREPROCESSING SUMMARY (REAL KAGGLE DATASET)")
    print("="*60)
    print(f"1. Number of training transformers : {len(train_df)}")
    print(f"2. Number of testing transformers  : {len(test_df)}")
    print(f"3. Number of final features        : {len(feature_cols)}")
    print(f"4. Final feature names             : {feature_cols}")
    print(f"5. FDD class distribution (Train)  : {train_df['target_fdd'].value_counts().to_dict()}")
    print(f"   FDD class distribution (Test)   : {test_df['target_fdd'].value_counts().to_dict()}")
    print(f"6. RUL statistics (Train)          : min={train_df['target_rul'].min()}, max={train_df['target_rul'].max()}, mean={train_df['target_rul'].mean():.2f}, median={train_df['target_rul'].median():.2f}")
    print(f"   RUL statistics (Test)           : min={test_df['target_rul'].min()}, max={test_df['target_rul'].max()}, mean={test_df['target_rul'].mean():.2f}, median={test_df['target_rul'].median():.2f}")
    print(f"7. Confirmation                    : NO SYNTHETIC DATA WAS USED.")
    print(f"8. Confirmation                    : NO WEATHER DATA WAS FABRICATED (Marked MISSING).")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_preprocessing_pipeline()
