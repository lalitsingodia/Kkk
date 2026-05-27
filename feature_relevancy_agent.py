# =============================================================================
# FEATURE RELEVANCY AGENT — NBFC Credit Risk EWS
# Periodic Feature Health Check + Report Generator
# Compatible with v2_advanced_model.py pipeline
# =============================================================================
#
# WHAT THIS AGENT DOES:
#   1. Loads the trained model + current feature set
#   2. Computes importance scores for ALL currently used features
#   3. Scans the database/new data for candidate NEW features
#   4. Evaluates new feature importance scores in isolation + combined
#   5. Runs drift detection on existing features
#   6. Generates a full PDF/Excel report with recommendations
#
# HOW TO RUN:
#   python feature_relevancy_agent.py --data ../data/train-test.xlsx
#                                     --model ../models/rf_model_v2.pkl
#                                     --period monthly
#                                     --output ../outputs/reports/
#
# SCHEDULE (run via cron or task scheduler):
#   Monthly:  0 0 1 * * python feature_relevancy_agent.py --period monthly
#   Quarterly:0 0 1 */3 * python feature_relevancy_agent.py --period quarterly
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scheduled runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, pointbiserialr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed. SHAP analysis will be skipped.")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, Image, PageBreak, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[WARN] reportlab not installed. PDF report will be skipped. Run: pip install reportlab")

# ── Styling ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

TARGET = 'Loan Status'

# =============================================================================
# CLI ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Relevancy Agent for NBFC EWS")
    parser.add_argument('--data',   default='../data/train-test.xlsx', help='Path to current dataset')
    parser.add_argument('--model',  default='../models/rf_model_v2.pkl', help='Path to trained model (.pkl)')
    parser.add_argument('--period', default='monthly', choices=['monthly', 'quarterly', 'adhoc'],
                        help='Report period type')
    parser.add_argument('--output', default='../outputs/reports/', help='Output directory for reports')
    parser.add_argument('--new_data', default=None,
                        help='Optional: path to newer data for drift detection')
    parser.add_argument('--candidate_features', default=None,
                        help='Optional: JSON list of candidate column names to evaluate')
    return parser.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

def cramers_v(x, y):
    cm_cv = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm_cv)[0]
    n = cm_cv.sum().sum()
    phi2 = chi2 / n
    r, k = cm_cv.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return 0.0 if denom == 0 else np.sqrt(phi2corr / denom)


# =============================================================================
# STEP 1: LOAD DATA + MODEL
# =============================================================================

class FeatureRelevancyAgent:

    def __init__(self, data_path, model_path, output_dir, period,
                 new_data_path=None, candidate_features=None):

        self.data_path          = data_path
        self.model_path         = model_path
        self.output_dir         = output_dir
        self.period             = period
        self.new_data_path      = new_data_path
        self.candidate_features = candidate_features or []
        self.run_ts             = datetime.now()
        self.report_name        = f"feature_relevancy_{period}_{self.run_ts.strftime('%Y%m')}"
        self.img_dir            = os.path.join(output_dir, 'images', self.report_name)
        self.findings           = []   # Will hold all textual findings for the report
        self.recommendations    = []

        ensure_dir(output_dir)
        ensure_dir(self.img_dir)

        print("\n" + "="*65)
        print("  FEATURE RELEVANCY AGENT — NBFC EWS")
        print(f"  Run Time  : {self.run_ts.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Period    : {period.upper()}")
        print(f"  Dataset   : {data_path}")
        print(f"  Model     : {model_path}")
        print("="*65)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self):
        print("\n[AGENT] Loading dataset and model...")

        # Dataset
        if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        print(f"  Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        # Model
        self.model = joblib.load(self.model_path)
        print(f"  Model loaded: {type(self.model).__name__}")

        # Newer data for drift (optional)
        if self.new_data_path:
            if self.new_data_path.endswith(('.xlsx','.xls')):
                self.df_new = pd.read_excel(self.new_data_path)
            else:
                self.df_new = pd.read_csv(self.new_data_path)
            print(f"  New data loaded: {self.df_new.shape[0]} rows for drift detection")
        else:
            self.df_new = None
            print("  No new data provided — drift detection will use temporal split.")

    # ── Reconstruct Feature Set from Model ───────────────────────────────────

    def get_model_features(self):
        """Extract feature names that the model was trained on."""
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'feature_importances_'):
            # Fall back — try to recover from the data columns minus target
            all_cols = [c for c in self.df.columns if c != TARGET]
            print("  [WARN] Model has no feature_names_in_. Using dataset columns as proxy.")
            return all_cols
        return []

    # =========================================================================
    # STEP 2: CURRENT FEATURE IMPORTANCE ANALYSIS
    # =========================================================================

    def analyze_current_features(self):
        print("\n[AGENT] Analyzing current feature importances...")

        model_features = self.get_model_features()

        # ── 2A. Model-native importances ────────────────────────────────────
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feat_count = len(importances)

            # Align feature names
            if len(model_features) == feat_count:
                names = model_features
            else:
                names = [f"Feature_{i}" for i in range(feat_count)]

            self.feat_imp_df = pd.DataFrame({
                'Feature': names,
                'Importance': importances,
                'Importance_%': (importances / importances.sum() * 100).round(2)
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

            self.feat_imp_df['Rank'] = self.feat_imp_df.index + 1
            self.feat_imp_df['Status'] = self.feat_imp_df['Importance'].apply(
                lambda x: '🔴 Low' if x < 0.005 else ('🟡 Medium' if x < 0.02 else '🟢 High')
            )

        elif hasattr(self.model, 'coef_'):
            # Logistic Regression
            coefs = np.abs(self.model.coef_[0])
            self.feat_imp_df = pd.DataFrame({
                'Feature': model_features[:len(coefs)],
                'Importance': coefs,
                'Importance_%': (coefs / coefs.sum() * 100).round(2)
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            self.feat_imp_df['Rank'] = self.feat_imp_df.index + 1
            self.feat_imp_df['Status'] = self.feat_imp_df['Importance'].apply(
                lambda x: '🔴 Low' if x < 0.01 else ('🟡 Medium' if x < 0.1 else '🟢 High')
            )
        else:
            print("  [WARN] Model type not supported for native importance extraction.")
            self.feat_imp_df = pd.DataFrame()
            return

        print(f"\n  Top 15 Current Features by Importance:")
        print(self.feat_imp_df[['Rank','Feature','Importance_%','Status']].head(15).to_string(index=False))

        low_imp = self.feat_imp_df[self.feat_imp_df['Status'] == '🔴 Low']
        if len(low_imp) > 0:
            self.findings.append(f"⚠️  {len(low_imp)} features have very low importance (<0.5%) and may be candidates for removal: {low_imp['Feature'].tolist()}")
            self.recommendations.append(f"CONSIDER DROPPING: {low_imp['Feature'].tolist()} — importance below threshold.")

        # ── 2B. Plot: Top 25 current feature importances ───────────────────
        top_n = min(25, len(self.feat_imp_df))
        fig, ax = plt.subplots(figsize=(13, 7))
        colors_map = self.feat_imp_df.head(top_n)['Status'].map(
            {'🟢 High': '#27ae60', '🟡 Medium': '#f39c12', '🔴 Low': '#e74c3c'}
        )
        bars = ax.barh(
            self.feat_imp_df.head(top_n)['Feature'][::-1],
            self.feat_imp_df.head(top_n)['Importance_%'][::-1],
            color=colors_map[::-1], edgecolor='white', height=0.75
        )
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'Current Model — Top {top_n} Feature Importances\n({self.period.title()} Review | {self.run_ts.strftime("%b %Y")})', fontsize=13)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Low threshold (0.5%)')
        ax.axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, label='Medium threshold (2%)')
        ax.legend(fontsize=9)
        # Add value labels
        for bar, val in zip(bars, self.feat_imp_df.head(top_n)['Importance_%'][::-1]):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=8)
        plt.tight_layout()
        self.current_imp_plot = save_fig(fig, os.path.join(self.img_dir, 'current_feature_importances.png'))

        # ── 2C. Correlation of current numeric features with target ─────────
        print("\n  Computing correlation with target...")
        self.corr_with_target = {}
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if TARGET in num_cols:
            num_cols.remove(TARGET)

        for col in num_cols:
            try:
                corr_val, p_val = pointbiserialr(
                    self.df[col].fillna(self.df[col].median()),
                    self.df[TARGET].fillna(0)
                )
                self.corr_with_target[col] = {'correlation': round(corr_val, 4), 'p_value': round(p_val, 4)}
            except Exception:
                pass

        corr_df = pd.DataFrame(self.corr_with_target).T
        if not corr_df.empty:
            corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
            corr_df['Significant'] = corr_df['p_value'] < 0.05
            self.corr_df = corr_df

            # Plot
            fig, ax = plt.subplots(figsize=(13, 6))
            colors_corr = ['#e74c3c' if v < 0 else '#2980b9' for v in corr_df['correlation'].head(20)]
            ax.barh(corr_df.index[:20][::-1], corr_df['correlation'].head(20)[::-1],
                    color=colors_corr[::-1], edgecolor='white')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title('Feature Correlation with Loan Default (Target)\n(Point Biserial Correlation)')
            ax.set_xlabel('Correlation Coefficient')
            plt.tight_layout()
            save_fig(fig, os.path.join(self.img_dir, 'correlation_with_target.png'))

            insig = corr_df[corr_df['Significant'] == False]
            if len(insig) > 0:
                self.findings.append(f"ℹ️  {len(insig)} numeric features have statistically insignificant correlation with target (p > 0.05): {insig.index.tolist()}")

        # ── 2D. SHAP Analysis ───────────────────────────────────────────────
        if SHAP_AVAILABLE and hasattr(self.model, 'feature_importances_'):
            try:
                print("  Computing SHAP values...")
                # Build a proxy X from df for SHAP (only model features that exist)
                shap_cols = [c for c in model_features if c in self.df.columns]
                X_shap = self.df[shap_cols].select_dtypes(include=np.number).fillna(0)

                explainer = shap.TreeExplainer(self.model)
                sample = X_shap.sample(min(300, len(X_shap)), random_state=42)
                shap_vals = explainer.shap_values(sample)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # class 1 for binary

                self.shap_importance = pd.DataFrame({
                    'Feature': sample.columns,
                    'SHAP_Mean_Abs': np.abs(shap_vals).mean(axis=0)
                }).sort_values('SHAP_Mean_Abs', ascending=False)

                fig, ax = plt.subplots(figsize=(13, 6))
                ax.barh(
                    self.shap_importance.head(20)['Feature'][::-1],
                    self.shap_importance.head(20)['SHAP_Mean_Abs'][::-1],
                    color='#8e44ad', edgecolor='white'
                )
                ax.set_title('SHAP Feature Importance (Top 20) — Mean |SHAP Value|')
                ax.set_xlabel('Mean |SHAP Value|')
                plt.tight_layout()
                save_fig(fig, os.path.join(self.img_dir, 'shap_importance.png'))
                print("  SHAP analysis complete.")
            except Exception as e:
                print(f"  [WARN] SHAP failed: {e}")
                self.shap_importance = None
        else:
            self.shap_importance = None

    # =========================================================================
    # STEP 3: CANDIDATE FEATURE EVALUATION
    # =========================================================================

    def evaluate_candidate_features(self):
        """
        Evaluate new candidate features from the database/dataset.
        These are columns present in the data but NOT in the model's current feature set.
        """
        print("\n[AGENT] Evaluating candidate/new features...")

        model_features = self.get_model_features()
        all_data_cols = [c for c in self.df.columns if c != TARGET]

        # Candidate = in data but not in model, OR explicitly passed
        auto_candidates = [c for c in all_data_cols if c not in model_features]
        candidates = list(set(auto_candidates + self.candidate_features))
        candidates = [c for c in candidates if c in self.df.columns]

        if not candidates:
            print("  No candidate features found. All dataset columns are already in the model.")
            self.candidate_results = pd.DataFrame()
            return

        print(f"  Candidate features found: {candidates}")
        self.candidate_results = []

        for col in candidates:
            result = {'Feature': col, 'Type': 'numeric' if self.df[col].dtype in [np.float64, np.int64] else 'categorical'}

            col_data = self.df[col].copy()

            # ── Encode if categorical ────────────────────────────────────
            if result['Type'] == 'categorical':
                le = LabelEncoder()
                col_encoded = le.fit_transform(col_data.astype(str).fillna('Missing'))
            else:
                col_encoded = col_data.fillna(col_data.median())

            # ── Univariate AUC ───────────────────────────────────────────
            try:
                auc = roc_auc_score(self.df[TARGET].fillna(0), col_encoded)
                auc = max(auc, 1 - auc)  # Flip if below 0.5
                result['Univariate_AUC'] = round(auc, 4)
            except Exception:
                result['Univariate_AUC'] = None

            # ── Correlation / Association with target ────────────────────
            try:
                corr_val, p_val = pointbiserialr(col_encoded, self.df[TARGET].fillna(0))
                result['Correlation_with_Target'] = round(abs(corr_val), 4)
                result['p_value'] = round(p_val, 4)
                result['Significant'] = p_val < 0.05
            except Exception:
                result['Correlation_with_Target'] = None
                result['p_value'] = None
                result['Significant'] = False

            # ── Missing Rate ─────────────────────────────────────────────
            result['Missing_%'] = round(self.df[col].isnull().mean() * 100, 2)

            # ── Unique Values ─────────────────────────────────────────────
            result['Unique_Values'] = self.df[col].nunique()

            # ── Default Rate Spread (categorical) ────────────────────────
            if result['Type'] == 'categorical' and self.df[col].nunique() <= 30:
                rates = self.df.groupby(col)[TARGET].mean()
                result['Default_Rate_Spread'] = round(rates.max() - rates.min(), 4)
            else:
                result['Default_Rate_Spread'] = None

            # ── RF importance when added to current features ──────────────
            try:
                base_cols = [c for c in model_features if c in self.df.columns]
                X_base = self.df[base_cols].select_dtypes(include=np.number).fillna(0)
                X_with_cand = X_base.copy()
                X_with_cand[col] = col_encoded.values

                y = self.df[TARGET].fillna(0)

                rf_probe = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores_base = cross_val_score(rf_probe, X_base, y, cv=cv, scoring='roc_auc')
                scores_new = cross_val_score(rf_probe, X_with_cand, y, cv=cv, scoring='roc_auc')

                result['Base_AUC_CV'] = round(scores_base.mean(), 4)
                result['With_Feature_AUC_CV'] = round(scores_new.mean(), 4)
                result['AUC_Gain'] = round(scores_new.mean() - scores_base.mean(), 4)
                result['Recommendation'] = (
                    '✅ ADD — Improves AUC' if result['AUC_Gain'] > 0.003 else
                    '🟡 MONITOR — Marginal gain' if result['AUC_Gain'] > 0 else
                    '❌ SKIP — No improvement'
                )
            except Exception as e:
                result['Base_AUC_CV'] = None
                result['With_Feature_AUC_CV'] = None
                result['AUC_Gain'] = None
                result['Recommendation'] = '⚠️ Could not evaluate'

            self.candidate_results.append(result)
            print(f"    {col}: Univariate AUC={result.get('Univariate_AUC','N/A')} | AUC Gain={result.get('AUC_Gain','N/A')} | → {result.get('Recommendation','N/A')}")

        self.candidate_results = pd.DataFrame(self.candidate_results)

        if not self.candidate_results.empty and 'AUC_Gain' in self.candidate_results.columns:
            add_features = self.candidate_results[
                self.candidate_results['Recommendation'].str.startswith('✅')
            ]['Feature'].tolist()
            skip_features = self.candidate_results[
                self.candidate_results['Recommendation'].str.startswith('❌')
            ]['Feature'].tolist()

            if add_features:
                self.findings.append(f"🆕 {len(add_features)} new candidate features show meaningful AUC improvement: {add_features}")
                self.recommendations.append(f"ADD TO NEXT MODEL VERSION: {add_features}")
            if skip_features:
                self.findings.append(f"ℹ️  {len(skip_features)} candidate features show no improvement and can be skipped.")

        # ── Plot: Candidate feature comparison ───────────────────────────
        if not self.candidate_results.empty and 'AUC_Gain' in self.candidate_results.columns:
            plot_df = self.candidate_results.dropna(subset=['AUC_Gain']).sort_values('AUC_Gain', ascending=False)
            if len(plot_df) > 0:
                fig, ax = plt.subplots(figsize=(13, max(5, len(plot_df)*0.5 + 1)))
                bar_colors = ['#27ae60' if v > 0.003 else ('#f39c12' if v > 0 else '#e74c3c')
                              for v in plot_df['AUC_Gain']]
                ax.barh(plot_df['Feature'][::-1], plot_df['AUC_Gain'][::-1],
                        color=bar_colors[::-1], edgecolor='white')
                ax.axvline(0, color='black', linewidth=1)
                ax.axvline(0.003, color='green', linestyle='--', alpha=0.6, label='Add threshold (+0.003)')
                ax.set_title('Candidate Feature Evaluation — AUC Gain over Baseline')
                ax.set_xlabel('AUC Gain')
                ax.legend()
                plt.tight_layout()
                save_fig(fig, os.path.join(self.img_dir, 'candidate_feature_auc_gain.png'))

    # =========================================================================
    # STEP 4: FEATURE DRIFT DETECTION
    # =========================================================================

    def detect_drift(self):
        """
        Detect distributional drift in current features.
        Uses temporal split if no new_data provided.
        """
        print("\n[AGENT] Running drift detection...")

        if self.df_new is not None:
            df_old = self.df
            df_new = self.df_new
            split_label = "Old vs New Dataset"
        else:
            # Temporal split: if date column exists, split by date; else 50/50
            date_cols = [c for c in self.df.columns if 'date' in c.lower() or 'Date' in c]
            if date_cols:
                date_col = date_cols[0]
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                    median_date = self.df[date_col].median()
                    df_old = self.df[self.df[date_col] <= median_date]
                    df_new = self.df[self.df[date_col] > median_date]
                    split_label = f"Before vs After {median_date.strftime('%b %Y')}"
                except Exception:
                    half = len(self.df) // 2
                    df_old = self.df.iloc[:half]
                    df_new = self.df.iloc[half:]
                    split_label = "First Half vs Second Half"
            else:
                half = len(self.df) // 2
                df_old = self.df.iloc[:half]
                df_new = self.df.iloc[half:]
                split_label = "First Half vs Second Half"

        print(f"  Drift split: {split_label} | Old: {len(df_old)} rows | New: {len(df_new)} rows")

        self.drift_results = []
        model_features = self.get_model_features()
        check_cols = [c for c in model_features if c in df_old.columns and c in df_new.columns]

        for col in check_cols:
            result = {'Feature': col}
            old_vals = df_old[col].dropna()
            new_vals = df_new[col].dropna()

            if old_vals.dtype in [np.float64, np.int64, float, int]:
                # KS test for numerical
                ks_stat_val, p_val = ks_2samp(old_vals, new_vals)
                result['Test'] = 'KS'
                result['Stat'] = round(ks_stat_val, 4)
                result['p_value'] = round(p_val, 4)
                result['Old_Mean'] = round(old_vals.mean(), 4)
                result['New_Mean'] = round(new_vals.mean(), 4)
                result['Mean_Shift_%'] = round(abs(new_vals.mean() - old_vals.mean()) / (old_vals.mean() + 1e-9) * 100, 2)
            else:
                # Chi-squared for categorical
                try:
                    ct = pd.crosstab(
                        pd.concat([old_vals.astype(str).assign(period='old') if hasattr(old_vals,'assign') else old_vals.astype(str)]),
                        columns='count'
                    )
                    # Simpler approach for categorical drift
                    old_dist = old_vals.astype(str).value_counts(normalize=True)
                    new_dist = new_vals.astype(str).value_counts(normalize=True)
                    all_cats = set(old_dist.index) | set(new_dist.index)
                    psi = sum(
                        (new_dist.get(c, 1e-4) - old_dist.get(c, 1e-4)) *
                        np.log((new_dist.get(c, 1e-4) / old_dist.get(c, 1e-4)))
                        for c in all_cats
                    )
                    result['Test'] = 'PSI'
                    result['Stat'] = round(psi, 4)
                    result['p_value'] = None
                    result['Old_Mean'] = 'N/A'
                    result['New_Mean'] = 'N/A'
                    result['Mean_Shift_%'] = None
                except Exception:
                    result['Test'] = 'N/A'
                    result['Stat'] = None
                    result['p_value'] = None
                    result['Old_Mean'] = 'N/A'
                    result['New_Mean'] = 'N/A'
                    result['Mean_Shift_%'] = None

            # Drift flag
            if result['Test'] == 'KS':
                result['Drift'] = '🔴 HIGH' if p_val < 0.01 else ('🟡 MODERATE' if p_val < 0.05 else '🟢 STABLE')
            elif result['Test'] == 'PSI' and result['Stat'] is not None:
                result['Drift'] = '🔴 HIGH' if result['Stat'] > 0.2 else ('🟡 MODERATE' if result['Stat'] > 0.1 else '🟢 STABLE')
            else:
                result['Drift'] = '⚪ UNKNOWN'

            self.drift_results.append(result)

        self.drift_df = pd.DataFrame(self.drift_results)

        if not self.drift_df.empty:
            high_drift = self.drift_df[self.drift_df['Drift'] == '🔴 HIGH']
            moderate_drift = self.drift_df[self.drift_df['Drift'] == '🟡 MODERATE']

            print(f"\n  Drift Summary: {len(high_drift)} HIGH drift | {len(moderate_drift)} MODERATE drift")
            if len(high_drift) > 0:
                print(f"  HIGH DRIFT features: {high_drift['Feature'].tolist()}")
                self.findings.append(f"🚨 DRIFT ALERT: {len(high_drift)} features show significant distributional drift: {high_drift['Feature'].tolist()}")
                self.recommendations.append(f"URGENT: Retrain model or investigate data pipeline for drifted features: {high_drift['Feature'].tolist()}")

            # ── Plot: Drift heatmap ───────────────────────────────────────
            drift_plot_df = self.drift_df[self.drift_df['Stat'].notna()].copy()
            if len(drift_plot_df) > 0:
                drift_plot_df = drift_plot_df.sort_values('Stat', ascending=False).head(20)
                drift_colors = drift_plot_df['Drift'].map(
                    {'🔴 HIGH': '#e74c3c', '🟡 MODERATE': '#f39c12', '🟢 STABLE': '#27ae60', '⚪ UNKNOWN': '#95a5a6'}
                )
                fig, ax = plt.subplots(figsize=(13, max(5, len(drift_plot_df)*0.5 + 1)))
                ax.barh(drift_plot_df['Feature'][::-1], drift_plot_df['Stat'].astype(float)[::-1],
                        color=drift_colors.values[::-1], edgecolor='white')
                ax.set_title(f'Feature Drift Detection — KS/PSI Statistics\n({split_label})')
                ax.set_xlabel('Drift Statistic (KS or PSI)')
                ax.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='PSI moderate (0.1)')
                ax.axvline(0.2, color='red', linestyle='--', alpha=0.5, label='PSI high (0.2)')
                ax.legend(fontsize=9)
                plt.tight_layout()
                save_fig(fig, os.path.join(self.img_dir, 'feature_drift.png'))

    # =========================================================================
    # STEP 5: FEATURE REDUNDANCY CHECK
    # =========================================================================

    def check_redundancy(self):
        """Check for highly correlated (redundant) features."""
        print("\n[AGENT] Checking feature redundancy...")

        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if TARGET in num_cols:
            num_cols.remove(TARGET)

        if len(num_cols) < 2:
            self.redundant_pairs = []
            return

        corr_matrix = self.df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.redundant_pairs = [
            {'Feature_A': col, 'Feature_B': upper[col][upper[col] > 0.85].index.tolist(),
             'Correlation': upper[col][upper[col] > 0.85].values.tolist()}
            for col in upper.columns if upper[col][upper[col] > 0.85].any()
        ]

        if self.redundant_pairs:
            pairs_flat = [(p['Feature_A'], b, c)
                         for p in self.redundant_pairs
                         for b, c in zip(p['Feature_B'], p['Correlation'])]
            self.findings.append(f"⚠️  {len(pairs_flat)} highly correlated feature pairs found (|r| > 0.85). Consider dropping one from each pair.")
            for a, b, c in pairs_flat:
                self.recommendations.append(f"REDUNDANCY: '{a}' ↔ '{b}' (r={c:.2f}) — consider removing one.")

        # Plot correlation matrix for current numeric features
        if len(num_cols) <= 30:
            fig, ax = plt.subplots(figsize=(14, 11))
            mask = np.triu(np.ones_like(corr_matrix.loc[num_cols, num_cols], dtype=bool))
            sns.heatmap(corr_matrix.loc[num_cols, num_cols], mask=mask, annot=len(num_cols)<=20,
                        fmt='.2f', cmap='RdYlGn', center=0, linewidths=0.3, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Feature Correlation Matrix (Current Features)')
            plt.tight_layout()
            save_fig(fig, os.path.join(self.img_dir, 'feature_correlation_matrix.png'))

    # =========================================================================
    # STEP 6: GENERATE EXCEL REPORT
    # =========================================================================

    def generate_excel_report(self):
        print("\n[AGENT] Generating Excel report...")
        excel_path = os.path.join(self.output_dir, f"{self.report_name}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            # Sheet 1: Summary
            summary_data = {
                'Item': [
                    'Report Generated', 'Period', 'Dataset Path', 'Model Path',
                    'Dataset Rows', 'Dataset Columns', 'Target Variable',
                    'Total Current Features (Model)',
                    'Features with High Importance (>2%)',
                    'Features with Low Importance (<0.5%)',
                    'Candidate Features Evaluated',
                    'Features Recommended to ADD',
                    'High Drift Features',
                    'Redundant Feature Pairs',
                    'Total Findings',
                    'Total Recommendations'
                ],
                'Value': [
                    self.run_ts.strftime('%Y-%m-%d %H:%M:%S'),
                    self.period.upper(),
                    self.data_path,
                    self.model_path,
                    self.df.shape[0],
                    self.df.shape[1],
                    TARGET,
                    len(self.feat_imp_df) if hasattr(self, 'feat_imp_df') else 'N/A',
                    len(self.feat_imp_df[self.feat_imp_df['Status']=='🟢 High']) if hasattr(self, 'feat_imp_df') and not self.feat_imp_df.empty else 'N/A',
                    len(self.feat_imp_df[self.feat_imp_df['Status']=='🔴 Low']) if hasattr(self, 'feat_imp_df') and not self.feat_imp_df.empty else 'N/A',
                    len(self.candidate_results) if hasattr(self, 'candidate_results') else 0,
                    len([r for r in self.recommendations if r.startswith('ADD')]),
                    len([f for f in self.findings if 'DRIFT ALERT' in f]),
                    len(self.redundant_pairs) if hasattr(self, 'redundant_pairs') else 0,
                    len(self.findings),
                    len(self.recommendations)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Current Feature Importances
            if hasattr(self, 'feat_imp_df') and not self.feat_imp_df.empty:
                self.feat_imp_df.to_excel(writer, sheet_name='Current_Importances', index=False)

            # Sheet 3: Candidate Feature Evaluation
            if hasattr(self, 'candidate_results') and not self.candidate_results.empty:
                self.candidate_results.to_excel(writer, sheet_name='Candidate_Features', index=False)

            # Sheet 4: Drift Detection
            if hasattr(self, 'drift_df') and not self.drift_df.empty:
                self.drift_df.to_excel(writer, sheet_name='Drift_Detection', index=False)

            # Sheet 5: Correlation with Target
            if hasattr(self, 'corr_df') and not self.corr_df.empty:
                self.corr_df.to_excel(writer, sheet_name='Correlation_with_Target')

            # Sheet 6: SHAP Importance
            if self.shap_importance is not None:
                self.shap_importance.to_excel(writer, sheet_name='SHAP_Importance', index=False)

            # Sheet 7: Findings & Recommendations
            findings_df = pd.DataFrame({
                'Finding': self.findings
            })
            rec_df = pd.DataFrame({'Recommendation': self.recommendations})
            findings_df.to_excel(writer, sheet_name='Findings', index=False)
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)

        print(f"  Excel report saved: {excel_path}")
        return excel_path

    # =========================================================================
    # STEP 7: GENERATE PDF REPORT
    # =========================================================================

    def generate_pdf_report(self):
        if not PDF_AVAILABLE:
            print("  [SKIP] PDF generation skipped — reportlab not installed.")
            return None

        print("\n[AGENT] Generating PDF report...")
        pdf_path = os.path.join(self.output_dir, f"{self.report_name}.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                topMargin=1.5*cm, bottomMargin=1.5*cm,
                                leftMargin=2*cm, rightMargin=2*cm)

        styles = getSampleStyleSheet()
        story = []

        # ── Custom styles ────────────────────────────────────────────────
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                      fontSize=20, spaceAfter=6,
                                      textColor=colors.HexColor('#1a252f'))
        subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'],
                                         fontSize=11, textColor=colors.HexColor('#555'),
                                         spaceAfter=12)
        h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
                                   fontSize=14, textColor=colors.HexColor('#1a252f'),
                                   spaceBefore=18, spaceAfter=6)
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                   fontSize=11, textColor=colors.HexColor('#2c3e50'),
                                   spaceBefore=12, spaceAfter=4)
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                     fontSize=9.5, leading=14, spaceAfter=5)
        finding_style = ParagraphStyle('Finding', parent=styles['Normal'],
                                        fontSize=9, leading=13,
                                        backColor=colors.HexColor('#fef9e7'),
                                        borderPadding=4, spaceAfter=4)
        rec_style = ParagraphStyle('Rec', parent=styles['Normal'],
                                    fontSize=9, leading=13,
                                    backColor=colors.HexColor('#eafaf1'),
                                    borderPadding=4, spaceAfter=4)

        def tbl_style(header_color='#1a252f'):
            return TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_color)),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,0), 8),
                ('FONTSIZE',   (0,1), (-1,-1), 7.5),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#ccc')),
                ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING', (0,0), (-1,-1), 4),
                ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ])

        # ── Cover Page ───────────────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("NBFC Credit Risk EWS", subtitle_style))
        story.append(Paragraph("Feature Relevancy & Health Report", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e74c3c')))
        story.append(Spacer(1, 0.3*cm))

        meta_data = [
            ['Report Period', self.period.upper()],
            ['Generated On', self.run_ts.strftime('%d %B %Y, %H:%M')],
            ['Dataset', os.path.basename(self.data_path)],
            ['Model', os.path.basename(self.model_path)],
            ['Dataset Size', f"{self.df.shape[0]:,} rows × {self.df.shape[1]} columns"],
            ['Total Findings', str(len(self.findings))],
            ['Total Recommendations', str(len(self.recommendations))],
        ]
        meta_tbl = Table(meta_data, colWidths=[5*cm, 11*cm])
        meta_tbl.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#ddd')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(meta_tbl)
        story.append(PageBreak())

        # ── Section 1: Executive Summary ─────────────────────────────────
        story.append(Paragraph("1. Executive Summary", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        if self.findings:
            story.append(Paragraph("Key Findings:", h2_style))
            for i, finding in enumerate(self.findings, 1):
                story.append(Paragraph(f"{i}. {finding}", finding_style))
        else:
            story.append(Paragraph("No significant issues found. Feature set appears healthy.", body_style))

        story.append(Spacer(1, 0.5*cm))
        if self.recommendations:
            story.append(Paragraph("Recommendations:", h2_style))
            for i, rec in enumerate(self.recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", rec_style))

        story.append(PageBreak())

        # ── Section 2: Current Feature Importances ────────────────────────
        story.append(Paragraph("2. Current Feature Importance Analysis", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        if os.path.exists(os.path.join(self.img_dir, 'current_feature_importances.png')):
            story.append(Image(os.path.join(self.img_dir, 'current_feature_importances.png'),
                               width=17*cm, height=8*cm))
            story.append(Spacer(1, 0.3*cm))

        if hasattr(self, 'feat_imp_df') and not self.feat_imp_df.empty:
            top15 = self.feat_imp_df.head(15)
            tbl_data = [['Rank', 'Feature', 'Importance %', 'Status']] + \
                       [[str(r['Rank']), r['Feature'], f"{r['Importance_%']:.2f}%", r['Status']]
                        for _, r in top15.iterrows()]
            tbl = Table(tbl_data, colWidths=[1.5*cm, 8*cm, 3*cm, 3.5*cm])
            tbl.setStyle(tbl_style())
            story.append(Paragraph("Top 15 Features by Model Importance:", h2_style))
            story.append(tbl)

        if os.path.exists(os.path.join(self.img_dir, 'correlation_with_target.png')):
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("Feature Correlation with Target (Point Biserial):", h2_style))
            story.append(Image(os.path.join(self.img_dir, 'correlation_with_target.png'),
                               width=17*cm, height=6.5*cm))

        if self.shap_importance is not None and os.path.exists(os.path.join(self.img_dir, 'shap_importance.png')):
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("SHAP Feature Importance:", h2_style))
            story.append(Image(os.path.join(self.img_dir, 'shap_importance.png'),
                               width=17*cm, height=6.5*cm))

        story.append(PageBreak())

        # ── Section 3: Candidate Feature Evaluation ───────────────────────
        story.append(Paragraph("3. Candidate / New Feature Evaluation", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        if hasattr(self, 'candidate_results') and not self.candidate_results.empty:
            if os.path.exists(os.path.join(self.img_dir, 'candidate_feature_auc_gain.png')):
                story.append(Image(os.path.join(self.img_dir, 'candidate_feature_auc_gain.png'),
                                   width=17*cm, height=7*cm))
                story.append(Spacer(1, 0.3*cm))

            show_cols = ['Feature', 'Type', 'Univariate_AUC', 'Correlation_with_Target',
                         'Missing_%', 'AUC_Gain', 'Recommendation']
            show_cols = [c for c in show_cols if c in self.candidate_results.columns]
            cand_display = self.candidate_results[show_cols]

            tbl_data = [show_cols] + cand_display.fillna('N/A').values.tolist()
            col_widths = [4*cm] + [2.5*cm] * (len(show_cols)-2) + [4*cm]
            tbl = Table(tbl_data, colWidths=col_widths[:len(show_cols)])
            tbl.setStyle(tbl_style('#27ae60'))
            story.append(tbl)
        else:
            story.append(Paragraph("No candidate features found in the dataset beyond current model features.", body_style))

        story.append(PageBreak())

        # ── Section 4: Drift Detection ────────────────────────────────────
        story.append(Paragraph("4. Feature Drift Detection", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        if os.path.exists(os.path.join(self.img_dir, 'feature_drift.png')):
            story.append(Image(os.path.join(self.img_dir, 'feature_drift.png'),
                               width=17*cm, height=7*cm))
            story.append(Spacer(1, 0.3*cm))

        if hasattr(self, 'drift_df') and not self.drift_df.empty:
            show_cols = ['Feature', 'Test', 'Stat', 'p_value', 'Old_Mean', 'New_Mean', 'Drift']
            show_cols = [c for c in show_cols if c in self.drift_df.columns]
            drift_display = self.drift_df[show_cols].sort_values('Drift', key=lambda x: x.map({'🔴 HIGH':0,'🟡 MODERATE':1,'🟢 STABLE':2,'⚪ UNKNOWN':3}))
            tbl_data = [show_cols] + drift_display.fillna('N/A').values.tolist()
            col_widths = [4.5*cm] + [2.2*cm] * (len(show_cols)-1)
            tbl = Table(tbl_data, colWidths=col_widths[:len(show_cols)])
            tbl.setStyle(tbl_style('#c0392b'))
            story.append(tbl)

        story.append(PageBreak())

        # ── Section 5: Redundancy Check ───────────────────────────────────
        story.append(Paragraph("5. Feature Redundancy Check", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        if os.path.exists(os.path.join(self.img_dir, 'feature_correlation_matrix.png')):
            story.append(Image(os.path.join(self.img_dir, 'feature_correlation_matrix.png'),
                               width=17*cm, height=11*cm))

        if self.redundant_pairs:
            story.append(Paragraph("Highly Correlated Pairs (|r| > 0.85):", h2_style))
            pairs_data = [['Feature A', 'Feature B', 'Correlation']]
            for p in self.redundant_pairs:
                for b, c in zip(p['Feature_B'], p['Correlation']):
                    pairs_data.append([p['Feature_A'], b, f"{c:.3f}"])
            tbl = Table(pairs_data, colWidths=[6*cm, 6*cm, 4*cm])
            tbl.setStyle(tbl_style('#8e44ad'))
            story.append(tbl)
        else:
            story.append(Paragraph("✅ No highly correlated feature pairs found. Feature set is non-redundant.", body_style))

        # ── Section 6: Action Plan ─────────────────────────────────────────
        story.append(PageBreak())
        story.append(Paragraph("6. Recommended Action Plan", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#ccc')))

        action_plan = [
            ['Priority', 'Action', 'Affected Features', 'Rationale']
        ]

        # Build action plan from findings
        has_drift = any('DRIFT ALERT' in f for f in self.findings)
        has_low_imp = any('low importance' in f.lower() for f in self.findings)
        has_add = any(r.startswith('ADD TO NEXT') for r in self.recommendations)
        has_redundant = bool(self.redundant_pairs) if hasattr(self, 'redundant_pairs') else False

        if has_drift:
            drifted = [f for f in self.findings if 'DRIFT ALERT' in f]
            action_plan.append(['🔴 URGENT', 'Investigate drifted features / retrain model', 'See Drift Sheet', 'Distributional shift detected'])
        if has_low_imp:
            action_plan.append(['🟡 NEXT SPRINT', 'Evaluate removing low-importance features', 'Importance < 0.5%', 'Reduce noise, improve speed'])
        if has_add:
            action_plan.append(['🟢 NEXT VERSION', 'Add new features to model', 'See Candidate Sheet', 'Proven AUC improvement'])
        if has_redundant:
            action_plan.append(['🟡 NEXT SPRINT', 'Remove one of each correlated pair', 'See Redundancy Section', 'Reduce multicollinearity'])
        if not any([has_drift, has_low_imp, has_add, has_redundant]):
            action_plan.append(['✅ NO ACTION', 'Model feature set is healthy', 'All features', 'Continue monitoring'])

        tbl = Table(action_plan, colWidths=[2.5*cm, 5.5*cm, 5*cm, 4*cm])
        tbl.setStyle(tbl_style('#1a252f'))
        story.append(tbl)

        # Footer note
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            f"This report was automatically generated by the Feature Relevancy Agent on "
            f"{self.run_ts.strftime('%d %B %Y')}. Next scheduled run: "
            f"{'Next Month' if self.period=='monthly' else 'Next Quarter'}.",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                           textColor=colors.HexColor('#888'), alignment=TA_CENTER)
        ))

        doc.build(story)
        print(f"  PDF report saved: {pdf_path}")
        return pdf_path

    # =========================================================================
    # MAIN RUN
    # =========================================================================

    def run(self):
        """Execute the full agent pipeline."""
        self.load()
        self.analyze_current_features()
        self.evaluate_candidate_features()
        self.detect_drift()
        self.check_redundancy()

        excel_path = self.generate_excel_report()
        pdf_path   = self.generate_pdf_report()

        # ── Final Console Summary ─────────────────────────────────────────
        print("\n" + "="*65)
        print("  FEATURE RELEVANCY AGENT — COMPLETE")
        print("="*65)
        print(f"\n  📊 Findings    : {len(self.findings)}")
        print(f"  ✅ Recs        : {len(self.recommendations)}")
        print(f"  📁 Excel       : {excel_path}")
        if pdf_path:
            print(f"  📄 PDF         : {pdf_path}")
        print(f"  🖼️  Images      : {self.img_dir}")

        if self.findings:
            print("\n  ── KEY FINDINGS ──────────────────────────────────────")
            for i, f in enumerate(self.findings, 1):
                print(f"  {i}. {f}")
        if self.recommendations:
            print("\n  ── RECOMMENDATIONS ───────────────────────────────────")
            for i, r in enumerate(self.recommendations, 1):
                print(f"  {i}. {r}")

        print("\n" + "="*65)
        return {'excel': excel_path, 'pdf': pdf_path}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    args = parse_args()

    # Parse candidate features if passed as JSON string
    candidates = []
    if args.candidate_features:
        try:
            candidates = json.loads(args.candidate_features)
        except json.JSONDecodeError:
            # Treat as comma-separated
            candidates = [c.strip() for c in args.candidate_features.split(',')]

    agent = FeatureRelevancyAgent(
        data_path          = args.data,
        model_path         = args.model,
        output_dir         = args.output,
        period             = args.period,
        new_data_path      = args.new_data,
        candidate_features = candidates
    )

    agent.run()
