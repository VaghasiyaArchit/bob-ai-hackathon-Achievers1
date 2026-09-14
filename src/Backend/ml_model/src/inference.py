"""
Unified Transformer Inference & Explainable AI (XAI) Risk Engine
Power Grid AI - Real Data Production-Grade Diagnostic Engine

Integrates:
1. Fault Detection & Diagnosis (FDD) - 4-Class Classification
2. Remaining Useful Life (RUL) - Continuous Lifetime Estimation
3. DGA Statistical Anomaly Scoring (0-100)
4. Explainable AI (XAI): Top 3 Evidence Risk Factors & Deterministic Local Explanations
5. Multi-Factor Equipment Risk Engine & Actionable Dispatch Logic
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd

EPSILON = 1e-12
EXPECTED_ROWS = 420
EXPECTED_COLS = ['H2', 'CO', 'C2H4', 'C2H2']

# Category 1 (Normal Healthy Baseline) empirical parameters established on the 1,705 training transformers
CAT1_BASELINE_MEANS = {
    'TDCG_final': 0.03769550,
    'TDCG_slope': 3.5523788e-05,
    'H2_final': 0.00235273,
    'C2H2_final': 0.00028980,
    'C2H4_final': 0.00732725
}

CAT1_BASELINE_STDS = {
    'TDCG_final': 0.01016366,
    'TDCG_slope': 1.4685010e-05,
    'H2_final': 0.00101181,
    'C2H2_final': 0.00011875,
    'C2H4_final': 0.00287352
}

# Physical fault descriptions for the 4 classes
FAULT_DESCRIPTIONS = {
    1: "Class 1: Normal Baseline Degradation (Cellulose Aging)",
    2: "Class 2: Low-Energy Partial Discharge (PD / Corona)",
    3: "Class 3: High-Temperature Thermal Oil Breakdown (> 700°C)",
    4: "Class 4: High-Energy Electrical Arcing / Flashover"
}

# Severity weights for expected fault risk (0 to 100 scale)
FAULT_SEVERITIES = {
    1: 0.0,    # Normal operation
    2: 55.0,   # Moderate dielectric stress
    3: 80.0,   # Serious hot-spot overheating
    4: 100.0   # Extreme hazard / arcing
}

RUL_MIN = 362.0
RUL_MAX = 1093.0


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


class TransformerRiskEngine:
    """
    Unified production inference engine for transformer DGA diagnostics,
    anomaly detection, RUL forecasting, and operational risk assessment.
    """
    def __init__(self, fdd_model_path: str = None, rul_model_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        
        self.fdd_model_path = fdd_model_path or os.path.join(models_dir, 'fdd_model.pkl')
        self.rul_model_path = rul_model_path or os.path.join(models_dir, 'rul_model.pkl')
        
        if not os.path.exists(self.fdd_model_path):
            raise FileNotFoundError(f"FDD model bundle missing at: {self.fdd_model_path}")
        if not os.path.exists(self.rul_model_path):
            raise FileNotFoundError(f"RUL model bundle missing at: {self.rul_model_path}")
            
        print(f"[ENGINE] Loading FDD model from {self.fdd_model_path}...")
        fdd_bundle = joblib.load(self.fdd_model_path)
        self.fdd_model = fdd_bundle['model']
        self.feature_names = fdd_bundle['feature_names']
        
        print(f"[ENGINE] Loading RUL model from {self.rul_model_path}...")
        rul_bundle = joblib.load(self.rul_model_path)
        self.rul_model = rul_bundle['model']
        
        print(f"[ENGINE] Risk Engine ready with {len(self.feature_names)} domain features.")

    def compute_anomaly_score(self, df_features: pd.DataFrame) -> pd.Series:
        """
        Calculates a transparent statistical anomaly score (0-100) based on
        multidimensional deviation from the healthy Category 1 baseline.
        """
        z_sq_terms = []
        for col, mean_val in CAT1_BASELINE_MEANS.items():
            std_val = CAT1_BASELINE_STDS[col]
            # Only positive deviations (elevated gas levels or generation rates) drive anomaly
            z_term = np.maximum(0.0, (df_features[col] - mean_val) / std_val)
            z_sq_terms.append(z_term**2)
            
        raw_anomaly = np.sqrt(np.mean(z_sq_terms, axis=0))
        # Non-linear saturation mapping to strictly [0, 100]
        anomaly_score = 100.0 * (1.0 - np.exp(-raw_anomaly / 1.5))
        return np.clip(anomaly_score, 0.0, 100.0)

    def compute_rul_risk(self, predicted_rul: np.ndarray) -> np.ndarray:
        """
        Converts predicted RUL into a normalized degradation risk score (0-100).
        Lower RUL produces higher risk.
        Formula: clip((1093 - predicted_rul) / (1093 - 362) * 100, 0, 100)
        """
        rul_risk = (RUL_MAX - predicted_rul) / (RUL_MAX - RUL_MIN) * 100.0
        return np.clip(rul_risk, 0.0, 100.0)

    def compute_fdd_risk(self, prob_matrix: np.ndarray) -> np.ndarray:
        """
        Computes expected fault risk score (0-100) as the probability-weighted
        sum of physical fault severities:
        Severity weights: Cat 1 = 0, Cat 2 = 55, Cat 3 = 80, Cat 4 = 100.
        """
        severities = np.array([
            FAULT_SEVERITIES[1],
            FAULT_SEVERITIES[2],
            FAULT_SEVERITIES[3],
            FAULT_SEVERITIES[4]
        ])
        fdd_risk = np.dot(prob_matrix, severities)
        return np.clip(fdd_risk, 0.0, 100.0)

    def assign_risk_category(self, score: float) -> str:
        """
        Maps final composite risk score (0-100) to operational risk category:
        85-100 = CRITICAL
        60-84  = HIGH
        35-59  = MEDIUM
        0-34   = LOW
        """
        if score >= 85.0:
            return 'CRITICAL'
        elif score >= 60.0:
            return 'HIGH'
        elif score >= 35.0:
            return 'MEDIUM'
        else:
            return 'LOW'

    def assign_recommended_action(self, category: str) -> str:
        """
        Provides standardized operational maintenance directive based on risk category.
        """
        if category == 'CRITICAL':
            return "Immediate inspection and maintenance recommended. Prioritize this transformer for engineering review."
        elif category == 'HIGH':
            return "Schedule urgent inspection/maintenance and closely monitor the transformer."
        elif category == 'MEDIUM':
            return "Schedule planned inspection and continue monitoring."
        else:
            return "Continue normal monitoring and scheduled maintenance."

    def extract_top_risk_factors(self, row: dict) -> list:
        """
        Identifies the top 3 most influential physical evidence risk factors for the transformer.
        Deterministic, empirical, and grounded strictly in the asset's feature values.
        """
        fault_id = int(row['predicted_fault'])
        c2h2 = row['C2H2_final']
        c2h4 = row['C2H4_final']
        h2 = row['H2_final']
        tdcg = row['TDCG_final']
        tdcg_slope = row['TDCG_slope']
        rul = row['predicted_rul']
        anom = row['anomaly_score']
        
        candidates = []
        
        # 1. Fault specific strong signals
        if fault_id == 4:
            candidates.append((90, f"Predicted Class 4 arcing pattern (confidence {row['fault_probability']*100:.1f}%)"))
        elif fault_id == 3:
            candidates.append((85, f"Predicted Class 3 thermal breakdown pattern (confidence {row['fault_probability']*100:.1f}%)"))
        elif fault_id == 2:
            candidates.append((75, f"Predicted Class 2 partial discharge pattern (confidence {row['fault_probability']*100:.1f}%)"))

        # 2. Chemical thresholds
        if c2h2 >= 0.00032:
            candidates.append((88, f"Elevated terminal Acetylene concentration (C2H2 = {c2h2:.5f})"))
        if c2h4 >= 0.0085:
            candidates.append((82, f"Elevated terminal Ethylene concentration (C2H4 = {c2h4:.5f})"))
        if h2 >= 0.0030:
            candidates.append((78, f"Elevated terminal Hydrogen concentration (H2 = {h2:.5f})"))
        if tdcg >= 0.0420:
            candidates.append((80, f"High total combustible gas accumulation (TDCG = {tdcg:.4f})"))
        if tdcg_slope >= 4.0e-5:
            candidates.append((72, "Accelerating combustible gas generation slope"))
            
        # 3. RUL depletion
        if rul <= 500:
            candidates.append((86, f"Critically depleted remaining useful life (approx. {rul:.0f} cycles)"))
        elif rul <= 700:
            candidates.append((65, f"Substantially degraded remaining useful life ({rul:.0f} cycles)"))
            
        # 4. Statistical anomaly
        if anom >= 50.0:
            candidates.append((70, f"Substantial multi-gas statistical deviation ({anom:.1f}/100 anomaly score)"))
        elif anom >= 35.0:
            candidates.append((55, f"Moderate statistical deviation from Category 1 baseline ({anom:.1f}/100)"))

        # Baseline factors if asset is low risk
        if len(candidates) < 3:
            if fault_id == 1:
                candidates.append((10, "Operating within normal Category 1 cellulose aging limits"))
            if tdcg < 0.040:
                candidates.append((8, "DGA combustible gas concentrations remain within standard limits"))
            if rul > 700:
                candidates.append((6, f"Healthy remaining useful life estimate ({rul:.0f} cycles)"))
            if anom < 30.0:
                candidates.append((5, "DGA sensor trajectories consistent with healthy Category 1 baseline"))

        # Sort by priority score and take top 3
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        top_3 = [item[1] for item in candidates[:3]]
        return top_3

    def generate_explanation(self, row: dict, top_factors: list) -> str:
        """
        Generates a deterministic, professional, non-dogmatic XAI explanation
        using model evidence, key DGA markers, and the top 3 risk factors.
        """
        fault_id = int(row['predicted_fault'])
        fault_name = FAULT_DESCRIPTIONS.get(fault_id, f"Class {fault_id}")
        prob = row['fault_probability'] * 100.0
        rul = row['predicted_rul']
        category = row['risk_category']
        action = row['recommended_action']
        
        f1_str = top_factors[0] if len(top_factors) > 0 else "observed gas pattern"
        f2_str = top_factors[1] if len(top_factors) > 1 else "elevated trajectory"
        
        if category == 'CRITICAL':
            return (
                f"Model predicts {fault_name} with high confidence ({prob:.1f}%). "
                f"{f1_str} and {f2_str} serve as strong contributing DGA signals consistent with the predicted fault pattern, "
                f"while predicted RUL is approximately {rul:.0f} cycles. {action}"
            )
        elif category == 'HIGH':
            return (
                f"Model identifies {fault_name} pattern ({prob:.1f}% confidence). "
                f"{f1_str} is an important indicator of significant asset degradation, "
                f"accompanied by {f2_str}, while estimated RUL is {rul:.0f} cycles. {action}"
            )
        elif category == 'MEDIUM':
            return (
                f"Model identifies {fault_name} pattern ({prob:.1f}% confidence). "
                f"{f1_str} represents an important indicator of moderate operational stress, "
                f"with estimated remaining useful life at {rul:.0f} cycles. {action}"
            )
        else:
            return (
                f"Model classifies asset as {fault_name} ({prob:.1f}% confidence). "
                f"Combustible gas measurements are consistent with the healthy Category 1 operating baseline, "
                f"and predicted RUL is healthy at approximately {rul:.0f} cycles. {action}"
            )

    def predict_from_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Executes unified prediction and XAI explanation pipeline over an extracted feature DataFrame.
        """
        if 'asset_id' in df_features.columns:
            asset_ids = df_features['asset_id'].tolist()
        else:
            asset_ids = [f"asset_{i}" for i in range(len(df_features))]
            
        X = df_features[self.feature_names].copy()
        
        # 1. Fault Detection & Diagnosis (FDD)
        probs = self.fdd_model.predict_proba(X)
        pred_faults = self.fdd_model.predict(X)
        
        prob_f1 = probs[:, 0]
        prob_f2 = probs[:, 1]
        prob_f3 = probs[:, 2]
        prob_f4 = probs[:, 3]
        max_probs = np.max(probs, axis=1)
        
        # 2. Remaining Useful Life (RUL)
        pred_ruls = self.rul_model.predict(X)
        pred_ruls = np.clip(pred_ruls, RUL_MIN, RUL_MAX)
        
        # 3. Anomaly Score (0-100)
        anomaly_scores = self.compute_anomaly_score(X)
        
        # 4. Component Risks
        fdd_risks = self.compute_fdd_risk(probs)
        rul_risks = self.compute_rul_risk(pred_ruls)
        
        # 5. Composite Equipment Risk Score (0-100)
        composite_risks = (0.40 * fdd_risks + 0.35 * rul_risks + 0.25 * anomaly_scores).round(2)
        composite_risks = np.clip(composite_risks, 0.0, 100.0)
        
        # Build Results
        results = []
        for i in range(len(X)):
            score = float(composite_risks[i])
            cat = self.assign_risk_category(score)
            action = self.assign_recommended_action(cat)
            
            row_dict = {
                'asset_id': asset_ids[i],
                'predicted_fault': int(pred_faults[i]),
                'fault_probability': round(float(max_probs[i]), 4),
                'prob_fault_1': round(float(prob_f1[i]), 4),
                'prob_fault_2': round(float(prob_f2[i]), 4),
                'prob_fault_3': round(float(prob_f3[i]), 4),
                'prob_fault_4': round(float(prob_f4[i]), 4),
                'predicted_rul': round(float(pred_ruls[i]), 2),
                'anomaly_score': round(float(anomaly_scores.iloc[i] if isinstance(anomaly_scores, pd.Series) else anomaly_scores[i]), 2),
                'risk_score': score,
                'risk_category': cat,
                'recommended_action': action,
                # Key feature references for explanation & risk factor extraction
                'C2H2_final': float(X['C2H2_final'].iloc[i]),
                'C2H4_final': float(X['C2H4_final'].iloc[i]),
                'H2_final': float(X['H2_final'].iloc[i]),
                'TDCG_final': float(X['TDCG_final'].iloc[i]),
                'TDCG_slope': float(X['TDCG_slope'].iloc[i])
            }
            
            top_factors = self.extract_top_risk_factors(row_dict)
            formatted_top_factors = f"1. {top_factors[0]}; 2. {top_factors[1]}; 3. {top_factors[2]}"
            row_dict['top_3_risk_factors'] = formatted_top_factors
            row_dict['explanation'] = self.generate_explanation(row_dict, top_factors)
            
            # Clean temporary feature references
            for k in ['C2H2_final', 'C2H4_final', 'H2_final', 'TDCG_final', 'TDCG_slope']:
                del row_dict[k]
                
            results.append(row_dict)
            
        df_out = pd.DataFrame(results)
        return df_out

    def predict_files(self, file_paths: list) -> pd.DataFrame:
        """
        Processes a list of raw 420-step transformer CSV files and runs unified inference with XAI.
        """
        feature_rows = []
        for fp in file_paths:
            asset_id = os.path.basename(fp)
            df_series = pd.read_csv(fp)
            feats = extract_transformer_features(df_series, asset_id)
            feature_rows.append(feats)
            
        df_features = pd.DataFrame(feature_rows)
        return self.predict_from_features(df_features)
