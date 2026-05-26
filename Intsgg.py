"""
NBFC Collection Allocation — ML Model Audit Agent
===================================================
This agent runs automatically in the background and does four things:

  1. OUTCOME SIMULATION
     Generates synthetic historical outcomes (what actually happened after
     each allocation) to give the model something to learn from.
     In production: replace this with your real outcomes table.

  2. FEATURE RELEVANCE AUDIT
     Trains an XGBoost model on historical outcomes and uses SHAP to rank
     every feature by its actual predictive importance. Compares against
     the current rule-based engine's assumed weights and flags divergences.

  3. COEFFICIENT RECOMMENDATION
     For each scoring formula in the current engine (contactability score,
     digital affinity, etc.), the agent re-derives statistically optimal
     weights using logistic regression + permutation importance and outputs
     a recommended new weight table with confidence intervals.

  4. NEW ATTRIBUTE DISCOVERY
     Tests candidate new attributes (e.g. outstanding_amount bucket,
     missed EMI count, max_delay, call consistency score) against outcomes
     and recommends which ones to add to the model and at what weight.

  5. AUTOMATED REPORT
     Writes a plain-language audit report every time it runs, with a
     diff of what changed vs the current model and a recommended action plan.

Run schedule (production):
  - Daily via cron / APScheduler / Airflow
  - Or call agent.run() from your orchestration layer

Usage:
  pip install pandas scikit-learn xgboost lightgbm shap scipy tabulate
  python ml_model_agent.py
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# Force UTF-8 output on Windows so box-drawing / special chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback — chars will be replaced with '?'

# ─────────────────────────────────────────────────────────────────────────────
# AGENT CONFIGURATION
# Tune these to match your production thresholds.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_CONFIG = {
    # Weight divergence threshold — if learned weight differs from current
    # weight by more than this fraction, flag it as a recommended change
    "weight_divergence_threshold": 0.15,

    # Minimum SHAP importance for a NEW feature to be recommended for inclusion
    "min_shap_for_new_feature": 0.03,

    # Minimum AUC improvement a new feature must provide to be added
    "min_auc_improvement": 0.005,

    # Number of cross-validation folds
    "cv_folds": 5,

    # Report output path
    "report_dir": "agent_reports",

    # State file (agent stores last-run recommendations here)
    "state_file": "agent_state.json",

    # Significance level for statistical tests
    "alpha": 0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# CURRENT MODEL WEIGHTS (from v2 engine)
# These are what the agent will audit and compare against.
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_MODEL_WEIGHTS = {
    "contactability_score": {
        "call_pickup_rate":         0.35,
        "conversation_rate":        0.25,
        "number_active":            0.20,
        "switchoff_penalty":       -0.15,   # negative = penalty
        "guarantor_reachable":      0.05,
    },
    "digital_affinity": {
        "upi_pct":                  1.00,
        "enach_pct":                0.90,
        "bbps_pct":                 0.70,
        "qr_pct":                   0.35,
        "cash_pct":                 0.05,
    },
    "composite_score": {
        "contactability_score":     0.40,
        "digital_affinity":         0.25,
        "delay_score":              0.20,
        "on_time_pct":              0.15,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE NEW ATTRIBUTES TO TEST
# The agent will evaluate whether these should enter the model.
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATE_NEW_ATTRIBUTES = [
    "outstanding_amount_log",        # log-scaled outstanding
    "missed_emi_count",              # raw missed EMI count
    "max_delay_days",                # worst single delay episode
    "bounce_rate",                   # bounce_count / total_emis
    "call_connect_efficiency",       # connected_calls / total_dialled
    "payment_mode_entropy",          # diversity of payment modes (high = inconsistent)
    "field_visit_density",           # field_visits / months_on_book
    "delay_trend",                   # is delay getting worse? (simulated)
    "digital_shift_trend",           # moving toward or away from digital? (simulated)
    "guarantor_leverage_score",      # combined guarantor signal
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA GENERATION
# Replicates v2 borrower data + generates synthetic historical outcomes.
# In production: load from your loan management / call log / CRM database.
# ─────────────────────────────────────────────────────────────────────────────

def build_base_dataset() -> pd.DataFrame:
    """Rebuild the 48-borrower dataset from v2 + extend with synthetic outcomes."""

    RAW = [
        ("LN-0012","Rajesh Kumar",        60,30, 8, 2, 0,  88,12,14, 0, 1,1,  2, 5,94, 0,0, 0,0,10,  4200, 12),
        ("LN-0047","Priya Sharma",         0,90, 8, 2, 0,  95,15,16, 0, 1,1,  1, 3,97, 0,0, 0,0,12,  3100, 14),
        ("LN-0083","Amit Singh",          30,20,40, 8, 2,  76,10,13, 1, 1,1,  4, 9,85, 1,0, 0,0, 8,  8600, 10),
        ("LN-0115","Sunita Patel",        55,25,15, 4, 1,  82,11,14, 0, 1,1,  3, 7,90, 0,0, 0,0, 9,  5500, 11),
        ("LN-0129","Vikram Nair",         20,10,60, 8, 2,  70, 9,13, 1, 1,1,  5,10,78, 1,0, 0,0, 7,  9200, 9),
        ("LN-0144","Kavitha Rao",         10,75,12, 2, 1,  90,14,16, 0, 1,1,  2, 6,92, 0,0, 0,0,11,  4800, 13),
        ("LN-0189","Meena Joshi",          5,85, 8, 1, 1,  93,15,16, 0, 1,1,  1, 4,96, 0,0, 0,0,13,  2900, 15),
        ("LN-0698","Renu Sharma",         65,20,10, 4, 1,  87,12,14, 0, 1,1,  3, 6,91, 0,0, 0,0,10,  5100, 12),
        ("LN-0572","Neha Chandra",        70,15,10, 4, 1,  80,11,14, 0, 1,1,  4, 8,83, 1,0, 0,0, 8,  7400, 10),
        ("LN-0156","Deepak Mehta",        15,10,45,20,10,  65, 8,13, 1, 1,1,  6,12,75, 2,1, 1,0, 6, 14200, 8),
        ("LN-0168","Ananya Iyer",         40,10,25,15,10,  79,10,13, 0, 1,1,  3, 8,88, 1,0, 0,0, 7,  9800, 11),
        ("LN-0172","Suresh Gupta",        10, 5,40,25,20,  60, 7,12, 2, 1,1,  7,14,72, 2,1, 1,0, 5, 17600, 7),
        ("LN-0307","Pooja Singh",         20,10,40,20,10,  58, 7,12, 2, 1,1,  7,13,73, 2,1, 0,0, 5, 13400, 8),
        ("LN-0558","Sanjay Patil",        25,10,40,15,10,  66, 8,13, 1, 1,1,  6,11,77, 1,0, 1,0, 6, 12800, 9),
        ("LN-0617","Vivek Agarwal",       30,10,40,12, 8,  72, 9,13, 1, 1,1,  5,10,80, 1,0, 0,0, 7, 10600, 10),
        ("LN-0628","Archana Pillai",      20,10,45,15,10,  63, 8,13, 1, 1,1,  7,12,74, 2,1, 0,0, 6, 13900, 8),
        ("LN-0589","Ajay Kumar",          15, 5,45,20,15,  55, 6,11, 2, 1,1,  8,15,70, 2,1, 1,0, 4, 18200, 7),
        ("LN-0671","Biswas Roy",          20,10,40,15,15,  52, 6,11, 2, 0,1, 10,18,67, 2,1, 1,0, 4, 20100, 6),
        ("LN-0203","Ramesh Tiwari",        5, 0,20,55,20,  45, 4,10, 3, 0,1, 12,22,60, 3,1, 2,1, 3, 28500, 8),
        ("LN-0218","Lalita Yadav",        10, 5,35,30,20,  50, 5,10, 2, 1,1, 10,19,65, 2,1, 1,0, 4, 22300, 9),
        ("LN-0235","Prakash Verma",        5, 0,15,50,30,  40, 3, 9, 3, 0,1, 15,25,55, 3,2, 2,1, 2, 34700, 7),
        ("LN-0247","Geeta Mishra",        10, 0,35,30,25,  55, 5,10, 2, 1,1,  9,17,68, 2,1, 1,0, 4, 19800, 9),
        ("LN-0261","Naresh Pandey",        5, 0,15,45,35,  42, 4, 9, 3, 0,1, 11,21,62, 3,2, 2,1, 2, 30200, 7),
        ("LN-0279","Sarita Chandra",      10, 5,30,30,25,  52, 5,10, 2, 1,1,  8,16,70, 2,1, 1,0, 4, 18900, 10),
        ("LN-0292","Anil Kumar",           5, 0,10,40,45,  38, 3, 9, 4, 0,1, 13,24,58, 3,2, 2,1, 2, 32400, 6),
        ("LN-0334","Rekha Pillai",        10, 5,30,30,25,  48, 4,10, 3, 0,1,  9,18,66, 2,1, 1,0, 3, 21500, 8),
        ("LN-0478","Mahesh Gupta",         5, 0,20,45,30,  35, 3, 9, 4, 0,1, 16,28,52, 3,2, 2,1, 2, 37800, 6),
        ("LN-0504","Pankaj Mishra",        5, 0,20,45,30,  38, 3, 9, 3, 0,1, 17,29,50, 3,2, 2,1, 2, 39200, 5),
        ("LN-0603","Sangeeta Singh",       5, 0,15,45,35,  42, 3, 9, 3, 0,1, 11,20,63, 3,2, 2,1, 2, 29800, 7),
        ("LN-0642","Sunil Patil",          5, 0,15,45,35,  36, 3, 9, 4, 0,1, 13,23,58, 3,2, 2,1, 2, 33100, 6),
        ("LN-0712","Kishore Gupta",        5, 0,20,45,30,  46, 4,10, 2, 1,1,  9,17,68, 2,1, 1,0, 3, 20900, 9),
        ("LN-0319","Harish Agarwal",       2, 0, 5,20,73,  35, 2, 8, 4, 0,1, 14,26,56, 4,2, 3,2, 1, 41200, 8),
        ("LN-0443","Leela Nair",           0, 0, 5,20,75,  25, 2, 8, 5, 0,1, 19,32,44, 4,2, 3,2, 1, 48700, 7),
        ("LN-0491","Seema Joshi",          2, 0, 3,15,80,  20, 1, 7, 5, 0,1, 21,35,42, 5,3, 3,2, 1, 52300, 6),
        ("LN-0348","Ganesh Patil",         0, 0, 2, 8,90,  20, 1, 7, 5, 0,1, 22,38,40, 5,3, 4,3, 0, 58400, 7),
        ("LN-0377","Mohan Lal",            0, 0, 3,10,87,  22, 1, 7, 5, 0,1, 18,33,45, 4,2, 5,4, 0, 49200, 8),
        ("LN-0405","Ratan Singh",          0, 0, 2, 8,90,  18, 1, 6, 6, 0,1, 20,36,42, 5,3, 3,2, 0, 54600, 7),
        ("LN-0362","Shanti Devi",          5, 0, 5,20,70,  15, 0, 6, 6, 0,0, 25,42,35, 5,3, 3,2, 0, 67300, 6),
        ("LN-0391","Champa Bai",           0, 0, 2, 8,90,  10, 0, 5, 7, 0,0, 30,50,30, 6,4, 4,3, 0, 78900, 5),
        ("LN-0418","Savita Kumari",        0, 0, 2,10,88,  12, 0, 5, 7, 0,0, 28,46,32, 6,3, 5,4, 0, 71200, 5),
        ("LN-0429","Dinesh Sharma",        0, 0, 1, 5,94,   8, 0, 4, 8, 0,0, 35,58,25, 7,4, 6,5, 0, 92100, 4),
        ("LN-0457","Brijesh Yadav",        0, 0, 1, 5,94,  10, 0, 5, 8, 0,0, 32,54,28, 6,4, 5,4, 0, 85400, 4),
        ("LN-0462","Usha Tiwari",          0, 0, 2, 8,90,  14, 0, 5, 6, 0,0, 24,40,38, 5,3, 4,3, 0, 63800, 5),
        ("LN-0518","Jyoti Verma",          0, 0, 2, 8,90,  15, 0, 5, 7, 0,0, 26,44,36, 5,3, 4,3, 0, 69500, 5),
        ("LN-0531","Rohit Pandey",         0, 0, 1, 5,94,  12, 0, 4, 7, 0,0, 29,48,30, 6,4, 5,4, 0, 80200, 4),
        ("LN-0547","Kamala Rao",           0, 0, 1, 4,95,   9, 0, 4, 9, 0,0, 33,55,26, 7,4, 6,5, 0, 94700, 4),
        ("LN-0656","Madhuri Devi",         0, 0, 2, 8,90,  13, 0, 5, 7, 0,0, 27,45,34, 5,3, 4,3, 0, 72600, 5),
        ("LN-0684","Tara Singh",           0, 0, 1, 3,96,   6, 0, 4,10, 0,0, 38,62,20, 8,5, 7,5, 0,105800, 4),
    ]

    COLS = [
        "loan_id","name",
        "upi_pct","enach_pct","bbps_pct","qr_pct","cash_pct",
        "call_pickup_rate","connected_calls","total_dialled","switchoff_freq",
        "guarantor_reachable","number_active",
        "avg_delay_days","max_delay_days","on_time_pct","bounce_count","missed_emi_count",
        "field_visits_done","paid_after_field","paid_after_call",
        "outstanding_amount","months_on_book"
    ]

    df = pd.DataFrame(RAW, columns=COLS)

    # ── Derived base features ──────────────────────────────────────────────
    df["digital_affinity"] = (
        df["upi_pct"]*1.00 + df["enach_pct"]*0.90 +
        df["bbps_pct"]*0.70 + df["qr_pct"]*0.35 + df["cash_pct"]*0.05
    ).round().astype(int)

    df["conversation_rate"] = (df["connected_calls"] / df["total_dialled"].replace(0,1) * 100).round()
    df["cash_dependency_pct"] = df["cash_pct"] + df["qr_pct"]

    df["contactability_score"] = (
        df["call_pickup_rate"]   * 0.35 +
        df["conversation_rate"]  * 0.25 +
        df["number_active"]      * 20   +
        df["guarantor_reachable"]* 5    -
        (df["switchoff_freq"]    * 4).clip(upper=20)
    ).clip(lower=0, upper=100).round().astype(int)

    df["delay_score"] = (100 - df["avg_delay_days"]*2.0).clip(lower=0).round().astype(int)

    df["composite_score"] = (
        df["contactability_score"]*0.40 +
        df["digital_affinity"]    *0.25 +
        df["delay_score"]         *0.20 +
        df["on_time_pct"]         *0.15
    ).round().astype(int)

    # ── CANDIDATE NEW ATTRIBUTES ────────────────────────────────────────────
    df["outstanding_amount_log"]    = np.log1p(df["outstanding_amount"])
    df["bounce_rate"]               = (df["bounce_count"] / df["months_on_book"].replace(0,1)).round(3)
    df["call_connect_efficiency"]   = df["conversation_rate"] / 100.0

    # Payment mode entropy — high entropy = customer uses many modes inconsistently
    def payment_entropy(row):
        probs = np.array([row["upi_pct"], row["enach_pct"], row["bbps_pct"],
                          row["qr_pct"], row["cash_pct"]]) / 100.0
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-10))
    df["payment_mode_entropy"] = df.apply(payment_entropy, axis=1).round(3)

    df["field_visit_density"]      = (df["field_visits_done"] / df["months_on_book"].replace(0,1)).round(3)
    df["guarantor_leverage_score"] = (df["guarantor_reachable"] * 50 +
                                       (df["guarantor_reachable"] * df["contactability_score"] * 0.5)).round()

    # Simulated trend features (in production, derive from time-series data)
    np.random.seed(42)
    df["delay_trend"]          = np.random.choice([-1, 0, 1], size=len(df),
                                   p=[0.3, 0.4, 0.3])   # -1=improving, 0=stable, 1=worsening
    df["digital_shift_trend"]  = np.random.choice([-1, 0, 1], size=len(df),
                                   p=[0.2, 0.5, 0.3])   # -1=moving to cash, 1=moving digital

    # ── SYNTHETIC OUTCOME LABELS ────────────────────────────────────────────
    # "collected_successfully" = 1 if payment was recovered within 30 days of allocation
    # Based on realistic assumptions:
    #   - Digital high-CS customers almost always pay after AI/Tele
    #   - Medium CS pays after tele ~60-70%
    #   - Low CS field-dependent pays after visit ~70-80%
    #   - Inactive numbers rarely resolve
    def generate_outcome(row):
        cs = row["contactability_score"]
        da = row["digital_affinity"]
        delay = row["avg_delay_days"]
        active = row["number_active"]
        fv = row["field_visits_done"]
        paf = row["paid_after_field"]

        # Base probability
        if active == 0:
            p = 0.25 + (paf / max(fv,1)) * 0.30
        elif cs >= 65 and da >= 50:
            p = 0.82 - delay * 0.008
        elif cs >= 40:
            p = 0.62 - delay * 0.010
        elif fv >= 2 and (paf / max(fv,1)) >= 0.6:
            p = 0.70
        else:
            p = 0.35 - delay * 0.005

        p = np.clip(p, 0.05, 0.95)
        return int(np.random.random() < p)

    np.random.seed(99)
    df["collected_successfully"] = df.apply(generate_outcome, axis=1)

    # Days to collection (if collected)
    def days_to_collect(row):
        if row["collected_successfully"] == 0:
            return np.nan
        cs = row["contactability_score"]
        if cs >= 65:  return np.random.randint(1, 8)
        if cs >= 40:  return np.random.randint(5, 20)
        return np.random.randint(10, 35)

    df["days_to_collection"] = df.apply(days_to_collect, axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE RELEVANCE AUDIT (XGBoost + SHAP)
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_FEATURES = [
    # Current model features
    "call_pickup_rate", "conversation_rate", "number_active",
    "switchoff_freq", "guarantor_reachable",
    "upi_pct", "enach_pct", "bbps_pct", "qr_pct", "cash_pct",
    "avg_delay_days", "on_time_pct",
    "contactability_score", "digital_affinity", "delay_score", "composite_score",
    # Candidate new features
    "outstanding_amount_log", "missed_emi_count", "max_delay_days",
    "bounce_rate", "call_connect_efficiency", "payment_mode_entropy",
    "field_visit_density", "delay_trend", "digital_shift_trend",
    "guarantor_leverage_score", "cash_dependency_pct",
]

def run_feature_audit(df: pd.DataFrame) -> dict:
    """
    Train XGBoost on all features, compute SHAP values,
    return ranked feature importance with current-model comparison.
    """
    print("\n  [Agent] Running feature relevance audit (XGBoost + SHAP)...")

    X = df[AUDIT_FEATURES].fillna(0)
    y = df["collected_successfully"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0
    )

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=AGENT_CONFIG["cv_folds"], shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

    model.fit(X_scaled, y)

    # SHAP importance
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_scaled)
    shap_mean  = np.abs(shap_vals).mean(axis=0)
    shap_df    = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "shap_importance": shap_mean,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1
    shap_df["shap_pct"] = (shap_df["shap_importance"] / shap_df["shap_importance"].sum() * 100).round(2)

    # Tag current vs candidate
    current_features = [
        "call_pickup_rate","conversation_rate","number_active","switchoff_freq",
        "guarantor_reachable","upi_pct","enach_pct","bbps_pct","qr_pct","cash_pct",
        "avg_delay_days","on_time_pct","contactability_score","digital_affinity",
        "delay_score","composite_score"
    ]
    shap_df["status"] = shap_df["feature"].apply(
        lambda f: "CURRENT" if f in current_features else "CANDIDATE"
    )

    return {
        "shap_df": shap_df,
        "cv_auc_mean": cv_aucs.mean(),
        "cv_auc_std":  cv_aucs.std(),
        "model": model,
        "scaler": scaler,
        "X": X,
        "y": y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — COEFFICIENT RECOMMENDATION
# Uses logistic regression to derive statistically optimal weights for each
# scoring formula and compares against current weights.
# ─────────────────────────────────────────────────────────────────────────────

def recommend_coefficients(df: pd.DataFrame) -> dict:
    """
    For each scoring sub-model, fit a logistic regression and extract
    normalised coefficients as recommended weights.
    Returns dict of {formula_name: {feature: {current, recommended, delta, flag}}}
    """
    print("  [Agent] Deriving optimal coefficients via logistic regression...")

    recommendations = {}
    y = df["collected_successfully"]

    for formula_name, current_weights in CURRENT_MODEL_WEIGHTS.items():
        features = list(current_weights.keys())

        # Map feature names to actual df columns
        col_map = {
            "switchoff_penalty": "switchoff_freq",   # penalty in engine
        }
        actual_cols = [col_map.get(f, f) for f in features]

        # Check all columns exist
        valid = [(f, c) for f, c in zip(features, actual_cols) if c in df.columns]
        if not valid:
            continue

        feat_names, col_names = zip(*valid)
        X_sub = df[list(col_names)].fillna(0)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sub)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_sc, y)

        # Normalise coefficients to sum to 1.0 (like weights)
        coefs     = lr.coef_[0]
        coef_abs  = np.abs(coefs)
        norm_coefs = coef_abs / coef_abs.sum()

        # Bootstrap confidence intervals
        n_boot = 500
        boot_coefs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(X_sc), len(X_sc), replace=True)
            lr_b = LogisticRegression(max_iter=500, random_state=None)
            try:
                lr_b.fit(X_sc[idx], y.iloc[idx])
                bc = np.abs(lr_b.coef_[0])
                boot_coefs.append(bc / bc.sum())
            except Exception:
                pass

        boot_arr = np.array(boot_coefs)
        ci_low   = np.percentile(boot_arr, 2.5,  axis=0)
        ci_high  = np.percentile(boot_arr, 97.5, axis=0)

        formula_rec = {}
        for i, fname in enumerate(feat_names):
            cur  = abs(current_weights[fname])   # current model weight
            rec  = round(float(norm_coefs[i]), 3)
            delta= round(rec - cur, 3)
            flag = abs(delta) > AGENT_CONFIG["weight_divergence_threshold"]
            formula_rec[fname] = {
                "current_weight":     cur,
                "recommended_weight": rec,
                "delta":              delta,
                "ci_low":             round(float(ci_low[i]),  3),
                "ci_high":            round(float(ci_high[i]), 3),
                "flag_change":        flag,
            }

        recommendations[formula_name] = formula_rec

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — NEW ATTRIBUTE DISCOVERY
# Tests each candidate feature individually and in combination.
# ─────────────────────────────────────────────────────────────────────────────

def discover_new_attributes(df: pd.DataFrame, audit_result: dict) -> list:
    """
    For each candidate new attribute:
      1. Measure its SHAP importance (already computed)
      2. Measure AUC improvement when added to a baseline model
      3. Check statistical significance (permutation test)
    Returns list of recommended new attributes with details.
    """
    print("  [Agent] Testing candidate new attributes for model inclusion...")

    shap_df = audit_result["shap_df"]
    X_base  = audit_result["X"]
    y       = audit_result["y"]

    candidates = [f for f in CANDIDATE_NEW_ATTRIBUTES if f in X_base.columns]
    base_features = [f for f in AUDIT_FEATURES if f not in candidates]
    base_features = [f for f in base_features if f in X_base.columns]

    scaler = StandardScaler()
    X_b_sc = scaler.fit_transform(X_base[base_features].fillna(0))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                    use_label_encoder=False, eval_metric="logloss",
                                    random_state=42, verbosity=0)
    base_auc = cross_val_score(base_model, X_b_sc, y, cv=cv, scoring="roc_auc").mean()

    recommendations = []
    for feat in candidates:
        shap_row = shap_df[shap_df["feature"] == feat]
        if shap_row.empty:
            continue
        shap_imp = shap_row.iloc[0]["shap_importance"]
        shap_pct = shap_row.iloc[0]["shap_pct"]
        rank     = shap_row.iloc[0]["rank"]

        # AUC with this feature added
        X_aug    = X_base[base_features + [feat]].fillna(0)
        X_aug_sc = scaler.fit_transform(X_aug)
        aug_auc  = cross_val_score(base_model, X_aug_sc, y, cv=cv, scoring="roc_auc").mean()
        auc_gain = aug_auc - base_auc

        # Permutation test: is this feature's importance statistically significant?
        base_model.fit(X_aug_sc, y)
        perm = permutation_importance(base_model, X_aug_sc, y,
                                       n_repeats=20, random_state=42)
        feat_idx = base_features.index(feat) if feat in base_features else len(base_features)
        perm_mean = perm.importances_mean[-1]
        perm_std  = perm.importances_std[-1]
        t_stat    = perm_mean / (perm_std + 1e-10)
        p_value   = stats.t.sf(t_stat, df=20)  # one-sided

        recommend = (
            shap_imp >= AGENT_CONFIG["min_shap_for_new_feature"] and
            auc_gain >= AGENT_CONFIG["min_auc_improvement"]
        )

        # Suggested weight if added
        suggested_weight = round(shap_pct / 100.0 * 0.5, 3)  # scale to sensible range

        recommendations.append({
            "feature":          feat,
            "shap_importance":  round(shap_imp, 4),
            "shap_pct":         shap_pct,
            "rank_overall":     int(rank),
            "auc_gain":         round(auc_gain, 4),
            "perm_importance":  round(perm_mean, 4),
            "p_value":          round(p_value, 4),
            "significant":      p_value < AGENT_CONFIG["alpha"],
            "recommend_add":    recommend,
            "suggested_weight": suggested_weight,
        })

    recommendations.sort(key=lambda x: x["shap_importance"], reverse=True)
    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — THRESHOLD OPTIMISATION
# Checks whether the current allocation thresholds are optimal.
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_THRESHOLDS = {
    "ai_calling": {
        "contactability_score_min": 65,
        "digital_affinity_min":     50,
        "avg_delay_max":             8,
    },
    "telecalling": {
        "contactability_score_min": 30,
    },
    "field_direct": {
        "contactability_score_max": 15,
        "avg_delay_min":            20,
    },
}

def optimise_thresholds(df: pd.DataFrame) -> dict:
    """
    Grid-search optimal thresholds for each channel boundary.
    Objective: maximise collection rate while minimising field visit %.
    """
    print("  [Agent] Optimising allocation thresholds...")

    results = {}
    y = df["collected_successfully"]

    # ── AI Calling: optimise contactability threshold ────────────────────────
    best_ai_cs, best_ai_score = 65, 0
    for cs_thresh in range(40, 80, 5):
        ai_mask = (df["contactability_score"] >= cs_thresh) & (df["digital_affinity"] >= 50)
        if ai_mask.sum() < 3:
            continue
        recovery = y[ai_mask].mean()
        field_pct = (df["contactability_score"] < cs_thresh).mean()
        # Combined score: reward recovery, penalise unnecessary field visits
        score = recovery * 0.7 - field_pct * 0.3
        if score > best_ai_score:
            best_ai_score = score
            best_ai_cs = cs_thresh

    results["ai_contactability_threshold"] = {
        "current": 65, "recommended": best_ai_cs,
        "change": best_ai_cs - 65,
        "flag": abs(best_ai_cs - 65) >= 5
    }

    # ── Telecalling lower bound ───────────────────────────────────────────────
    best_tele_cs, best_tele_score = 30, 0
    for cs_thresh in range(15, 50, 5):
        tele_mask = (df["contactability_score"] >= cs_thresh) & \
                    (df["contactability_score"] < best_ai_cs)
        if tele_mask.sum() < 3:
            continue
        recovery = y[tele_mask].mean()
        score = recovery
        if score > best_tele_score:
            best_tele_score = score
            best_tele_cs = cs_thresh

    results["tele_contactability_lower_bound"] = {
        "current": 30, "recommended": best_tele_cs,
        "change": best_tele_cs - 30,
        "flag": abs(best_tele_cs - 30) >= 5
    }

    # ── Field visit delay threshold ───────────────────────────────────────────
    best_delay, best_delay_score = 20, 0
    for delay_thresh in range(10, 35, 3):
        field_mask = (df["avg_delay_days"] >= delay_thresh) & \
                     (df["contactability_score"] < 35)
        if field_mask.sum() < 2:
            continue
        recovery = y[field_mask].mean()
        score = recovery
        if score > best_delay_score:
            best_delay_score = score
            best_delay = delay_thresh

    results["field_delay_threshold"] = {
        "current": 20, "recommended": best_delay,
        "change": best_delay - 20,
        "flag": abs(best_delay - 20) >= 3
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODEL DRIFT DETECTION
# Compares current run against last stored state to detect drift.
# ─────────────────────────────────────────────────────────────────────────────

def detect_drift(current_auc: float, new_feature_recs: list) -> dict:
    """
    Load previous agent state and compare AUC and recommendations.
    Flags if significant drift detected.
    """
    state_file = Path(AGENT_CONFIG["state_file"])
    drift_report = {"previous_auc": None, "auc_change": None, "drift_detected": False}

    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
        prev_auc = state.get("last_auc", None)
        if prev_auc:
            auc_change = current_auc - prev_auc
            drift_report["previous_auc"] = round(prev_auc, 4)
            drift_report["auc_change"]   = round(auc_change, 4)
            drift_report["drift_detected"] = abs(auc_change) > 0.05

    # Save current state
    new_state = {
        "last_run": datetime.datetime.now().isoformat(),
        "last_auc": round(current_auc, 4),
        "recommended_new_features": [r["feature"] for r in new_feature_recs if r["recommend_add"]],
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(new_state, f, indent=2)

    return drift_report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    df, audit_result, coeff_recs, new_attr_recs, threshold_recs, drift_report
) -> str:
    """Generate a plain-language audit report and save to file."""

    os.makedirs(AGENT_CONFIG["report_dir"], exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(AGENT_CONFIG["report_dir"], f"model_audit_{ts}.txt")

    lines = []
    def h(title): lines.append("\n" + "="*90 + f"\n  {title}\n" + "="*90)
    def p(text=""):  lines.append(f"  {text}")

    lines.append("#"*90)
    lines.append("  NBFC COLLECTION ALLOCATION -- ML MODEL AUDIT REPORT")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    lines.append("#"*90)

    # ── Executive Summary ────────────────────────────────────────────────────
    h("EXECUTIVE SUMMARY")
    auc = audit_result["cv_auc_mean"]
    p(f"Model AUC (cross-validated):   {auc:.4f}  ±{audit_result['cv_auc_std']:.4f}")
    if drift_report["previous_auc"]:
        p(f"Previous run AUC:              {drift_report['previous_auc']}")
        p(f"AUC change:                    {drift_report['auc_change']:+.4f}")
        if drift_report["drift_detected"]:
            p("!! DRIFT DETECTED -- AUC shifted >5%. Review data pipeline and feature distributions.")
    p()

    flagged_weights = sum(
        1 for fm in coeff_recs.values() for v in fm.values() if v["flag_change"]
    )
    new_recs = [r for r in new_attr_recs if r["recommend_add"]]
    flagged_thresholds = sum(1 for v in threshold_recs.values() if v["flag"])

    p(f"Weight changes recommended:    {flagged_weights}")
    p(f"New attributes recommended:    {len(new_recs)}")
    p(f"Threshold changes recommended: {flagged_thresholds}")

    priority = "LOW"
    if flagged_weights > 3 or len(new_recs) >= 2 or flagged_thresholds >= 2:
        priority = "HIGH"
    elif flagged_weights > 1 or len(new_recs) >= 1:
        priority = "MEDIUM"
    p(f"\n  Action Priority: {priority}")

    # ── Feature Importance Ranking ───────────────────────────────────────────
    h("FEATURE IMPORTANCE RANKING (SHAP — XGBoost)")
    shap_df = audit_result["shap_df"]
    p("Top 15 features by SHAP importance:")
    p()
    top15 = shap_df.head(15)
    for _, row in top15.iterrows():
        bar  = "|" * int(row["shap_pct"] / 1.5)
        tag  = "* CANDIDATE" if row["status"] == "CANDIDATE" else "  current  "
        p(f"  {int(row['rank']):>2}. {row['feature']:<35} {row['shap_pct']:>5.1f}%  {bar}  [{tag}]")

    # ── Coefficient Recommendations ──────────────────────────────────────────
    h("COEFFICIENT RECOMMENDATIONS")
    for formula_name, formula_recs in coeff_recs.items():
        p(f"Formula: {formula_name}")
        p()
        rows = []
        for feat, vals in formula_recs.items():
            flag_str = "!! CHANGE" if vals["flag_change"] else "  ok    "
            rows.append([
                feat,
                f"{vals['current_weight']:.3f}",
                f"{vals['recommended_weight']:.3f}",
                f"{vals['delta']:+.3f}",
                f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}]",
                flag_str
            ])
        table = tabulate(rows,
            headers=["Feature","Current","Recommended","Delta","95% CI","Status"],
            tablefmt="outline")
        for line in table.split("\n"):
            p("  " + line)
        p()

    # ── New Attribute Recommendations ────────────────────────────────────────
    h("NEW ATTRIBUTE RECOMMENDATIONS")
    if new_recs:
        p("The following candidate features are recommended for inclusion:\n")
        rows = []
        for r in new_attr_recs:
            rec_str = "[ADD]" if r["recommend_add"] else " skip"
            sig_str = "Yes" if r["significant"] else "No"
            rows.append([
                r["feature"],
                f"{r['shap_importance']:.4f}",
                f"{r['shap_pct']:.1f}%",
                r["rank_overall"],
                f"{r['auc_gain']:+.4f}",
                f"{r['p_value']:.3f}",
                sig_str,
                f"{r['suggested_weight']:.3f}",
                rec_str,
            ])
        table = tabulate(rows,
            headers=["Feature","SHAP","SHAP%","Rank","AUC Gain","p-val","Sig?","Sug. Weight","Action"],
            tablefmt="outline")
        for line in table.split("\n"):
            p("  " + line)
    else:
        p("No new attributes meet the inclusion threshold at this time.")

    # ── Threshold Recommendations ─────────────────────────────────────────────
    h("ALLOCATION THRESHOLD RECOMMENDATIONS")
    for thresh_name, vals in threshold_recs.items():
        flag_str = "!! CHANGE RECOMMENDED" if vals["flag"] else "  no change needed"
        p(f"{thresh_name}:")
        p(f"  Current value:     {vals['current']}")
        p(f"  Recommended value: {vals['recommended']}  (Δ {vals['change']:+d})")
        p(f"  Status:            {flag_str}")
        p()

    # ── Recommended New Scoring Formula ──────────────────────────────────────
    h("RECOMMENDED UPDATED SCORING FORMULAS")
    p("CONTACTABILITY SCORE (recommended weights):")
    cm = coeff_recs.get("contactability_score", {})
    for f, v in cm.items():
        w = v["recommended_weight"]
        p(f"  {f:<28} × {w:.3f}")
    p()
    p("DIGITAL AFFINITY (recommended weights):")
    dm = coeff_recs.get("digital_affinity", {})
    for f, v in dm.items():
        w = v["recommended_weight"]
        p(f"  {f:<28} × {w:.3f}  (mapped to 0–100 scale)")
    p()
    p("COMPOSITE SCORE (recommended weights):")
    comp = coeff_recs.get("composite_score", {})
    for f, v in comp.items():
        w = v["recommended_weight"]
        p(f"  {f:<28} × {w:.3f}")

    if new_recs:
        p()
        p("NEW VARIABLES TO ADD TO COMPOSITE SCORE:")
        for r in new_recs:
            p(f"  {r['feature']:<28} × {r['suggested_weight']:.3f}  (SHAP rank #{r['rank_overall']})")

    # ── Action Plan ───────────────────────────────────────────────────────────
    h("RECOMMENDED ACTION PLAN")
    action_num = 1
    for formula_name, formula_recs in coeff_recs.items():
        flagged = {f: v for f, v in formula_recs.items() if v["flag_change"]}
        if flagged:
            p(f"  {action_num}. Update weights in '{formula_name}':")
            for feat, vals in flagged.items():
                p(f"     {feat}: {vals['current_weight']:.3f} → {vals['recommended_weight']:.3f}  (Δ {vals['delta']:+.3f})")
            action_num += 1

    for r in new_recs:
        p(f"  {action_num}. Add new feature '{r['feature']}' to model")
        p(f"     SHAP importance: {r['shap_pct']:.1f}% | AUC gain: {r['auc_gain']:+.4f} | Suggested weight: {r['suggested_weight']:.3f}")
        action_num += 1

    for thresh_name, vals in threshold_recs.items():
        if vals["flag"]:
            p(f"  {action_num}. Update threshold '{thresh_name}': {vals['current']} → {vals['recommended']}")
            action_num += 1

    if drift_report["drift_detected"]:
        p(f"  {action_num}. Investigate model drift — AUC changed {drift_report['auc_change']:+.4f}")
        action_num += 1

    lines.append("\n" + "#"*90)
    lines.append("  END OF REPORT")
    lines.append("#"*90 + "\n")

    report_text = "\n".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath, report_text


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — AUTO-PATCHER
# Writes a patched version of the engine weights if agent recommends changes.
# ─────────────────────────────────────────────────────────────────────────────

def auto_patch_weights(coeff_recs: dict, threshold_recs: dict) -> str:
    """
    Generate a patched weights file (not auto-applied — requires human approval).
    Returns path to the patch file.
    """
    patch = {
        "generated_at":      datetime.datetime.now().isoformat(),
        "status":            "PENDING_REVIEW",
        "requires_approval": True,
        "weight_patches": {},
        "threshold_patches": {},
    }

    for formula, recs in coeff_recs.items():
        flagged = {f: v["recommended_weight"] for f, v in recs.items() if v["flag_change"]}
        if flagged:
            patch["weight_patches"][formula] = flagged

    for thresh_name, vals in threshold_recs.items():
        if vals["flag"]:
            patch["threshold_patches"][thresh_name] = {
                "old": vals["current"],
                "new": vals["recommended"],
            }

    patch_path = os.path.join(AGENT_CONFIG["report_dir"], "pending_patch.json")
    os.makedirs(AGENT_CONFIG["report_dir"], exist_ok=True)
    with open(patch_path, "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2)

    return patch_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CollectionModelAgent:
    """
    Background ML agent for continuous model auditing and improvement.

    Usage:
        agent = CollectionModelAgent()
        agent.run()

    In production — schedule via APScheduler:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(agent.run, 'cron', hour=2)   # 2 AM daily
        scheduler.start()
    """

    def __init__(self, config: dict = None):
        self.config = config or AGENT_CONFIG
        self.last_run = None
        self.last_report = None

    def run(self):
        print("\n" + "#"*70)
        print("  COLLECTION MODEL AGENT -- AUDIT RUN STARTED")
        print(f"  {datetime.datetime.now().strftime('%d %b %Y %H:%M:%S')}")
        print("#"*70)

        # Step 1: Load data
        print("\n  [Agent] Loading borrower dataset + outcomes...")
        df = build_base_dataset()
        print(f"  [Agent] Dataset loaded: {len(df)} borrowers, "
              f"{df['collected_successfully'].sum()} successful outcomes")

        # Step 2: Feature audit
        audit_result = run_feature_audit(df)
        print(f"  [Agent] Feature audit complete. "
              f"CV AUC: {audit_result['cv_auc_mean']:.4f} ±{audit_result['cv_auc_std']:.4f}")

        # Step 3: Coefficient recommendations
        coeff_recs = recommend_coefficients(df)
        flagged_total = sum(
            1 for fm in coeff_recs.values() for v in fm.values() if v["flag_change"]
        )
        print(f"  [Agent] Coefficient audit complete. {flagged_total} changes flagged.")

        # Step 4: New attribute discovery
        new_attr_recs = discover_new_attributes(df, audit_result)
        rec_new = [r for r in new_attr_recs if r["recommend_add"]]
        print(f"  [Agent] Attribute discovery complete. "
              f"{len(rec_new)} new attributes recommended.")

        # Step 5: Threshold optimisation
        threshold_recs = optimise_thresholds(df)
        flagged_thresh = sum(1 for v in threshold_recs.values() if v["flag"])
        print(f"  [Agent] Threshold optimisation complete. "
              f"{flagged_thresh} threshold changes recommended.")

        # Step 6: Drift detection
        drift_report = detect_drift(audit_result["cv_auc_mean"], new_attr_recs)
        if drift_report["drift_detected"]:
            print(f"  [Agent] !! MODEL DRIFT DETECTED! AUC change: "
                  f"{drift_report['auc_change']:+.4f}")
        else:
            print("  [Agent] No significant model drift detected.")

        # Step 7: Write report
        report_path, report_text = write_report(
            df, audit_result, coeff_recs, new_attr_recs, threshold_recs, drift_report
        )
        print(f"\n  [Agent] Report saved: {report_path}")

        # Step 8: Auto-patch (pending human approval)
        patch_path = auto_patch_weights(coeff_recs, threshold_recs)
        print(f"  [Agent] Patch file (pending approval): {patch_path}")

        self.last_run    = datetime.datetime.now()
        self.last_report = report_path

        # Print report to console
        print(report_text)

        print("#"*70)
        print("  AGENT RUN COMPLETE")
        print("#"*70 + "\n")

        return {
            "audit":      audit_result,
            "coeff_recs": coeff_recs,
            "new_attrs":  new_attr_recs,
            "thresholds": threshold_recs,
            "drift":      drift_report,
            "report":     report_path,
            "patch":      patch_path,
        }

    def schedule_daily(self, hour: int = 2):
        """
        Schedule agent to run daily at given hour (requires APScheduler).

        pip install apscheduler

        from apscheduler.schedulers.background import BackgroundScheduler
        agent = CollectionModelAgent()
        agent.schedule_daily(hour=2)
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            scheduler.add_job(self.run, "cron", hour=hour, minute=0)
            scheduler.start()
            print(f"  [Agent] Scheduled to run daily at {hour:02d}:00")
            return scheduler
        except ImportError:
            print("  [Agent] APScheduler not installed.")
            print("  Run: pip install apscheduler")
            print("  Then call agent.schedule_daily(hour=2)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = CollectionModelAgent()
    results = agent.run()
