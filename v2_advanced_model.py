import warnings

import shap

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import(
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import joblib
import pickle

sns.set_theme(style = "whitegrid", palette = "muted")
plt.rcParams['figure.figsize']=(10,5)
plt.rcParams['axes.titlesize']=13
plt.rcParams['axes.labelsize']=11

TARGET = 'Loan Status'

df = pd.read_excel("../data/train-test.xlsx")

employee_ids = df['Lead Generator'].copy()
drop_cols = ['Loan No']

for col in drop_cols:
    if col in df.columns:
        df.drop(columns = [col], inplace = True)

binary_cols = { 'Gender':[0,1], 'Martial Status':[0,1], 'Guarantor Presence':[0,1], TARGET:[0,1]}
for col, valid in binary_cols.items():
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x if x in valid else np.nan)

num_cols = df.select_dtypes(include = np.number).columns.tolist()
cat_cols = df.select_dtypes(include = 'object').columns.tolist()

if TARGET in num_cols: num_cols.remove(TARGET)
for col in cat_cols: df[col].fillna(df[col].mode()[0],inplace = True)

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

for col in num_cols:
    if df[col].skew()>1:
        df[col] = np.log1p(df[col])

if 'EMI' in df.columns and 'Loan Amount' in df.columns:
     df['EMI_to_Loan'] = df['EMI']/(df['Loan Amount'] + 1)

if 'LTV' in df.columns:
    df['High_LTV'] = (df['LTV'] > 0.8).astype(int)

if 'CIBIL' in df.columns:
    df['CIBIL_bucket'] = pd.cut(df['CIBIL'], bins = [-1,0,650,750,900], labels = ['New','Poor','Fair','Excellent'])
    df['CIBIL_bucket'] = df['CIBIL_bucket'].astype(str)

x = df.drop(columns = [TARGET])
y = df[TARGET]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

emp_train, emp_test = train_test_split(employee_ids, test_size = 0.2, random_state = 42, stratify = y)


emp_stats = pd.DataFrame({'Lead Generator': emp_train.values, TARGET : y_train.values})
emp_counts = emp_stats.groupby('Lead Generator')[TARGET].count()
valid_employees = emp_counts[emp_counts >= 20].index
emp_default_rate = (emp_stats[emp_stats['Lead Generator'].isin(valid_employees)].groupby('Lead Generator')[TARGET].mean())
x_train['Emp_Default_Rate'] = emp_train.map(emp_default_rate)
x_test['Emp_Default_Rate'] = emp_test.map(emp_default_rate)
global_default_rate = y_train.mean()
x_train['Emp_Default_Rate'] = (x_train['Emp_Default_Rate'].fillna(global_default_rate))
x_test['Emp_Default_Rate'] = (x_test['Emp_Default_Rate'].fillna(global_default_rate))

if 'Lead Generator' in x_train.columns:
    x_train.drop(columns = ['Lead Generator'], inplace = True)
if 'Lead Generator' in x_test.columns:
    x_test.drop(columns = ['Lead Generator'], inplace = True)

pd_default_rate = df.groupby('PD Employee ID')[TARGET].mean()
df['PD_Default_Rate'] = df['PD Employee ID'].map(pd_default_rate)

if 'PD Employee ID' in x_train.columns:
    x_train.drop(columns = ['PD Employee ID'], inplace = True)
if 'PD Employee ID' in x_test.columns:
    x_test.drop(columns = ['PD Employee ID'], inplace = True)

df['Sanction Date'] = pd.to_datetime(df['Sanction Date'])
df['Sanction_Month'] = df['Sanction Date'].dt.month
df['Sanction_Day'] = df['Sanction Date'].dt.day
def sanction_bucket(day):
    if day <= 5:
        return 'Start_Month'
    elif day <= 10:
        return 'Early_Month'
    elif day <= 15:
        return 'Mid_Month'
    elif day <= 20:
        return 'Late_Mid'
    elif day <= 25:
        return 'Month_End'
    else:
        return 'Extreme Month End'
df['Sanction_Bucket'] = df['Sanction_Day'].apply(sanction_bucket)

if 'Sanction Date' in x_train.columns:
    x_train.drop(columns = ['Sanction Date'], inplace = True)
if 'Sanction Date' in x_test.columns:
    x_test.drop(columns = ['Sanction Date'], inplace = True)

cat_cols = ['Loan_Purpose','Industry','Occupation','Education','Location','Sanction_Bucket', 'Sanction_Month']
for col in cat_cols:
    if col in df.columns:
        df[col] = (df[col].astype(str).str.strip().str.title())
for col in num_cols:
    if df[col].skew()>1:
        df[col] = np.log1p(df[col])
cat_cols = x_train.select_dtypes(include = ['object', 'category']).columns
for col in cat_cols:
    x_train[col] = x_train[col].astype(str)
    x_test[col] = x_test[col].astype(str)
    x_train[col] = x_train[col].replace('nan','Missing')
    x_test[col] = x_test[col].replace('nan','Missing')
    x_train[col] = x_train[col].fillna('Missing')
    x_test[col] = x_test[col].fillna('Missing')

if 'Education' in x_train.columns:
    edu_order = ['No Education', '10th', '12th', 'Graduate', 'Post Graduate', 'other']
    enc = OrdinalEncoder(categories = [edu_order])
    x_train[['Education']] = enc.fit_transform(x_train[['Education']])
    x_test[['Education']] = enc.transform(x_test[['Education']])

low_card = ['Loan Purpose', 'Property type']
for col in low_card:
    if col in x_train.columns:
        x_train = pd.get_dummies(x_train, columns = [col], drop_first = True)
        x_test = pd.get_dummies(x_test, columns = [col], drop_first = True)

x_test = x_test.reindex(columns = x_train.columns, fill_value = 0)

high_card = ['Occupation','Location']
for col in high_card:
    if col in x_train.columns:
        freq = x_train [col].value_counts(normalize = True)
        x_train[col] = x_train[col].map(freq)
        x_test[col] = x_test[col].map(freq).fillna(0)

numeric_df = df.select_dtypes(include = [np.number])
corr_matrix = numeric_df.corr(method = 'pearson')

plt.figure(figsize = (18,12))

sns.heatmap(corr_matrix, annot=True, fmt = ".2f", cmap = 'coolwarm', center = 0, linewidths = 0.5)
plt.title('Correlation Matrix')
plt.show()

target_corr = corr_matrix[TARGET].sort_values(ascending = False)
print("\nCorrelation with target variable")
print(target_corr)
plt.figure(figsize = (8,10))
target_corr.drop(TARGET).sort_values().plot(kind ='barh')
plt.title("Feature Correlation with Loan Status")
plt.xlabel("Pearson Correlation")
plt.show()

x_train = x_train.replace([np.inf,-np.inf], np.nan)
x_test = x_test.replace([np.inf,-np.inf], np.nan)

for col in x_train.columns:
    if x_train[col].dtype in ['float64', 'int64']:
        median_val = x_train[col].median()
        x_train[col] = x_train[col].fillna(median_val)
    else:
        mode_val = x_train[col].mode()[0]
        x_train[col] = x_train[col].fillna(mode_val)

    if x_test[col].dtype in ['float64', 'int64']:
        median_val = x_train[col].median()
        x_test[col] = x_test[col].fillna(median_val)
    else:
        mode_val = x_train[col].mode()[0]
        x_test[col] = x_test[col].fillna(mode_val)

x_train = x_train.apply(pd.to_numeric, errors = 'coerce')
x_test = x_test.apply(pd.to_numeric, errors = 'coerce')

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

x_train_bal = x_train.copy()
y_train_bal = y_train.copy()

def ks_stat(y_true,y_proba):
    fpr, tpr, _ = roc_curve(y_true,y_proba)
    return np.max(tpr - fpr)

def evaluate_model(model, name):
    model.fit(x_train_bal, y_train_bal)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    ks = ks_stat(y_test, y_proba)
    print(f"\n{name}")
    print(f"AUC: {auc:.4f} | KS:{ks:.4f}")
    print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()
    return auc, model

models = {
    "Logistic Regression": LogisticRegression(max_iter = 1000, class_weight = 'balanced'),
    "Random Forest": RandomForestClassifier(n_estimators = 200, class_weight = 'balanced'),
    "XGBoost": XGBClassifier(eval_metric = 'logloss', use_label_encoder = False, scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum())
}

results = {}
for name, model in models.items():
    auc,trained_model = evaluate_model(model, name)
    results[name] = (auc, trained_model)

best_model_name = max(results, key = lambda x: results[x][0])
best_model = results[best_model_name][1]

print(f"\nBest Model:{best_model_name}")

explainer = shap.TreeExplainer(best_model)
sample_data = x_test.sample(500,random_state=42)
shap_values = explainer.shap_values(sample_data)
shap.summary_plot(shap_values, sample_data,max_display=15)
shap.summary_plot(shap_values, sample_data, plot_type = 'bar',max_display=15)
plt.figure(figsize=(10,6))
plt.savefig("../outputs/feature_importance/shap_summary_v2.png",bbox_inches='tight')

from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(x_train, y_train)
y_proba = calibrated_model.predict_proba(x_test)[:,1]

thresholds = np.arange(0.01,1.00,0.01)
best_threshold = 0.5
lowest_cost = float('inf')
cost_fn = 200000
cost_fp = 5000
for threshold in thresholds:
    y_pred_temp = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    if total_cost < lowest_cost:
        lowest_cost = total_cost
        best_threshold = threshold

print(f"Optimal Thresold:{best_threshold: .3f}")
print(f"Optimal Cost:{lowest_cost: .3f}")

y_pred_opt = (y_proba >= best_threshold).astype(int)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_opt))

cm = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm).plot()
plt.title("Final Confusion Matrix")
plt.show()
plt.savefig("../outputs/confusion_matrices/random_forest_v2.png")
risk_score = y_proba * 100

low = np.percentile(risk_score,30)
high = np.percentile(risk_score,70)

def bucket(x):
    if x < low: return "Low"
    elif x < high: return "Medium"
    else: return "High"

ews = pd.DataFrame({
    "Risk Score": risk_score,
    "Risk Bucket": [bucket(x) for x in risk_score],
    "Actual": y_test.values
})

print("\nDEFAULT RATE BY RISK BUCKET:")
print(ews.groupby("Risk Bucket")["Actual"].mean())

location_risk = df.groupby('Location')[TARGET].agg(['mean','count'])
location_risk = location_risk.sort_values('mean', ascending = False)
print("\nRisk By Location")
print(location_risk.head(10))

industry_risk = df.groupby('Industry')[TARGET].agg(['mean','count'])
industry_risk = industry_risk.sort_values('mean', ascending = False)
print("\nRisk Bucket by Industry")
print(industry_risk.head(10))

loan_purpose_risk = df.groupby('Loan Purpose')[TARGET].agg(['mean','count'])
loan_purpose_risk = loan_purpose_risk.sort_values('mean', ascending = False)
print("\nRisk Bucket by Loan Purpose")
print(loan_purpose_risk)

location_risk.to_excel("../outputs/risk_outputs/location_risk_v2.xlsx")
industry_risk.to_excel("../outputs/risk_outputs/industry_risk_v2.xlsx")
loan_purpose_risk.to_excel("../outputs/risk_outputs/loan_purpose_risk_v2.xlsx")

month_risk = df.groupby('Sanction_Month')[TARGET].agg(['mean','count'])
print("\nSanction Month Risk")
print(month_risk.sort_values(by = 'mean',ascending = False))

bucket_risk = df.groupby('Sanction_Bucket')[TARGET].agg(['mean','count'])
print("\nSanction Bucket Risk")
print(bucket_risk.sort_values(by = 'mean',ascending = False))

month_risk.to_excel("../outputs/risk_outputs/month_risk_v2.xlsx", index=False)
bucket_risk.to_excel("../outputs/risk_outputs/bucket_risk_v2.xlsx", index=False)
importances = best_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': x_train.columns, 'Importances': importances})
feat_imp = feat_imp.sort_values('Importances', ascending = False)
feat_imp.to_excel("../outputs/feature_importance/feature_importance_v2.xlsx")
ews.to_excel("../outputs/risk_outputs/ews_output_v2.xlsx", index=False)
print(feat_imp.head(20))

joblib.dump(best_model,"../models/rf_model_v2.pkl")