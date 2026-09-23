"""
==============================================================================
Customer Churn Prediction and Retention Intelligence System
==============================================================================
Dataset  : Telco Customer Churn (Kaggle)
Source   : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Author   : Data Science / ML Engineering Team
File     : customer_churn_project.py
Run with : streamlit run customer_churn_project.py
==============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import shap

import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED          = 42
DATA_PATH            = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL           = "Churn"
ID_COL               = "customerID"
TEST_SIZE            = 0.20
CV_FOLDS             = 5
LOW_RISK_THRESHOLD   = 0.35   # prob <  0.35  -> Low Risk
HIGH_RISK_THRESHOLD  = 0.65   # prob >= 0.65  -> High Risk

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e293b 0%,#0f172a 100%);border-right:1px solid rgba(99,102,241,.3);}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.page-title{background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.4rem;font-weight:800;letter-spacing:-.5px;margin-bottom:.25rem;}
.kpi-card{background:linear-gradient(135deg,rgba(30,41,59,.95) 0%,rgba(15,23,42,.95) 100%);border:1px solid rgba(99,102,241,.3);border-radius:16px;padding:1.4rem 1.6rem;text-align:center;box-shadow:0 4px 24px rgba(99,102,241,.12);}
.kpi-value{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;}
.kpi-label{font-size:.82rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-top:.3rem;font-weight:500;}
.kpi-delta{font-size:.78rem;color:#64748b;margin-top:.15rem;}
.section-header{color:#e2e8f0;font-size:1.25rem;font-weight:700;border-left:4px solid #6366f1;padding-left:.75rem;margin:1.5rem 0 .75rem 0;}
.insight-box{background:linear-gradient(135deg,rgba(99,102,241,.12) 0%,rgba(6,182,212,.08) 100%);border:1px solid rgba(99,102,241,.35);border-radius:12px;padding:1rem 1.25rem;margin:.5rem 0;color:#cbd5e1;font-size:.9rem;line-height:1.6;}
.insight-box strong{color:#a5b4fc;}
.rec-card{background:linear-gradient(135deg,rgba(15,23,42,.9),rgba(30,41,59,.9));border:1px solid rgba(99,102,241,.25);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.75rem;color:#e2e8f0;}
.rec-card .rec-title{font-size:.95rem;font-weight:700;color:#a5b4fc;margin-bottom:.3rem;}
.rec-card .rec-body{font-size:.87rem;color:#94a3b8;line-height:1.55;}
.alert-warning{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);border-radius:10px;padding:.75rem 1rem;color:#fde68a;font-size:.88rem;}
hr.fancy{border:none;border-top:1px solid rgba(99,102,241,.25);margin:1.5rem 0;}
</style>
"""

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(15,23,42,0.0)",
    plot_bgcolor="rgba(30,41,59,0.5)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(99,102,241,0.15)", linecolor="rgba(99,102,241,0.3)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.15)", linecolor="rgba(99,102,241,0.3)"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(15,23,42,0.5)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
)

COLORS = {
    "primary": "#6366f1", "secondary": "#06b6d4", "accent": "#8b5cf6",
    "danger": "#f87171", "warning": "#fbbf24", "success": "#34d399",
    "churn_no": "#34d399", "churn_yes": "#f87171",
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_raw_data(path):
    return pd.read_csv(path)


def inspect_dataset(df):
    info = {}
    info["n_rows"]      = len(df)
    info["n_cols"]      = len(df.columns)
    info["columns"]     = df.columns.tolist()
    info["dtypes"]      = df.dtypes.to_dict()
    info["missing"]     = df.isnull().sum().to_dict()
    info["duplicates"]  = int(df.duplicated().sum())
    info["target_dist"] = df[TARGET_COL].value_counts().to_dict()
    info["churn_rate"]  = round(df[TARGET_COL].value_counts(normalize=True).get("Yes", 0) * 100, 2)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    info["numerical_cols"]   = [c for c in num_cols if c not in [ID_COL, TARGET_COL]]
    info["categorical_cols"] = [c for c in cat_cols if c not in [ID_COL, TARGET_COL]]
    return info


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def clean_data(df_raw):
    df  = df_raw.copy()
    log = []

    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        log.append(f"Removed {before - len(df)} duplicate rows.")
    else:
        log.append("No duplicate rows found.")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    log.append("Stripped whitespace from all string columns.")

    blank_tc = (df["TotalCharges"] == "").sum()
    if blank_tc > 0:
        log.append(f"TotalCharges: {blank_tc} blank strings found (new customers, tenure≈0). Replaced with 0.")
    df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    log.append("TotalCharges converted to float64; blanks/NaN filled with 0.")

    df["Churn_Binary"] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    log.append("Target encoded: Yes->1, No->0 into 'Churn_Binary'.")
    log.append("SeniorCitizen already binary int (0/1). No change needed.")

    remaining_na = df.isnull().sum().sum()
    if remaining_na > 0:
        df = df.dropna()
        log.append(f"Dropped rows with remaining NaN values.")
    else:
        log.append("No missing values remain after cleaning.")

    log.append(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns.")
    return df, log


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def engineer_features(df):
    df  = df.copy()
    log = []

    def tenure_grp(t):
        if t <= 12:   return "0-12 months"
        elif t <= 24: return "13-24 months"
        elif t <= 48: return "25-48 months"
        else:         return "49+ months"

    df["TenureGroup"] = df["tenure"].apply(tenure_grp)
    log.append("TenureGroup: Binned tenure into 4 cohorts (0-12, 13-24, 25-48, 49+).")

    svc_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["ServiceCount"] = df[svc_cols].apply(
        lambda r: sum(1 for v in r if v in ["Yes","DSL","Fiber optic"]), axis=1
    )
    log.append("ServiceCount: Total active services subscribed.")

    df["ChargePerService"] = np.where(
        df["ServiceCount"] > 0,
        df["MonthlyCharges"] / df["ServiceCount"],
        df["MonthlyCharges"]
    )
    log.append("ChargePerService: MonthlyCharges / ServiceCount.")

    df["IsMonthToMonth"]   = (df["Contract"] == "Month-to-month").astype(int)
    log.append("IsMonthToMonth: 1 if Month-to-month contract.")

    df["HasProtection"] = (
        (df["TechSupport"] == "Yes") | (df["OnlineSecurity"] == "Yes")
    ).astype(int)
    log.append("HasProtection: 1 if TechSupport or OnlineSecurity is Yes.")

    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    log.append("IsElectronicCheck: 1 if payment method is Electronic check.")

    log.append("All features use only pre-prediction information. No target leakage.")
    return df, log


# ═════════════════════════════════════════════════════════════════════════════
# EDA STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_eda_stats(df):
    stats = {}
    cat_vars = [
        "gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","TenureGroup"
    ]
    for col in cat_vars:
        if col in df.columns:
            grp = df.groupby(col)["Churn_Binary"].agg(["mean","count"]).reset_index()
            grp.columns = [col,"ChurnRate","Count"]
            grp["ChurnRate"] *= 100
            stats[f"churn_by_{col}"] = grp

    stats["corr_matrix"] = df.select_dtypes(include=[np.number]).drop(
        columns=["Churn_Binary"], errors="ignore"
    ).corr()

    stats["service_churn"] = df.groupby("ServiceCount")["Churn_Binary"].agg(
        ["mean","count"]
    ).reset_index()
    stats["service_churn"].columns = ["ServiceCount","ChurnRate","Count"]
    stats["service_churn"]["ChurnRate"] *= 100
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# KPIs
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_kpis(df):
    k = {}
    k["total_customers"]    = len(df)
    k["churned"]            = int(df["Churn_Binary"].sum())
    k["retained"]           = k["total_customers"] - k["churned"]
    k["churn_rate"]         = round(k["churned"] / k["total_customers"] * 100, 2)
    k["avg_tenure"]         = round(df["tenure"].mean(), 1)
    k["avg_monthly_charge"] = round(df["MonthlyCharges"].mean(), 2)
    k["avg_total_charge"]   = round(df["TotalCharges"].mean(), 2)
    k["total_monthly_rev"]  = round(df["MonthlyCharges"].sum(), 2)
    k["mtm_customers"]      = int((df["Contract"] == "Month-to-month").sum())
    k["mtm_pct"]            = round(k["mtm_customers"] / k["total_customers"] * 100, 1)
    k["churned_monthly_rev"]= round(df.loc[df["Churn_Binary"]==1,"MonthlyCharges"].sum(), 2)
    return k


# ═════════════════════════════════════════════════════════════════════════════
# ML PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_lists(df):
    cat_features = [c for c in [
        "gender","Partner","Dependents","PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
    ] if c in df.columns]
    num_features = [c for c in [
        "SeniorCitizen","tenure","MonthlyCharges","TotalCharges","ServiceCount",
        "ChargePerService","IsMonthToMonth","HasProtection","IsElectronicCheck"
    ] if c in df.columns]
    return cat_features, num_features


def build_preprocessor(cat_features, num_features):
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num_pipe = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


@st.cache_resource(show_spinner=False)
def train_all_models(df):
    cat_features, num_features = build_feature_lists(df)
    X = df[cat_features + num_features]
    y = df["Churn_Binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    cw_arr = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
    spw    = cw_arr[1] / cw_arr[0]   # scale_pos_weight for XGBoost

    pre = build_preprocessor(cat_features, num_features)

    model_defs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=0.5, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, clf in model_defs.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        results[name] = {
            "pipeline":    pipe,
            "X_test":      X_test, "y_test": y_test,
            "y_pred":      y_pred, "y_prob": y_prob,
            "accuracy":    round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision":   round(precision_score(y_test, y_pred) * 100, 2),
            "recall":      round(recall_score(y_test, y_pred) * 100, 2),
            "f1":          round(f1_score(y_test, y_pred) * 100, 2),
            "roc_auc":     round(roc_auc_score(y_test, y_prob) * 100, 2),
            "cv_auc_mean": round(cv_auc.mean() * 100, 2),
            "cv_auc_std":  round(cv_auc.std() * 100, 2),
            "cm":          confusion_matrix(y_test, y_pred),
            "clf_report":  classification_report(y_test, y_pred,
                                                  target_names=["Retained","Churned"]),
            "cat_features": cat_features,
            "num_features": num_features,
        }

    # ── Hyperparameter Tuning (XGBoost) ─────────────────────────────────────
    tuning_info = {}
    try:
        param_distributions = {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [3, 4, 5, 6],
            "clf__learning_rate":     [0.05, 0.1, 0.15],
            "clf__subsample":         [0.8, 0.9, 1.0],
            "clf__colsample_bytree":  [0.8, 0.9, 1.0],
            "clf__min_child_weight":  [1, 3, 5],
        }
        xgb_pipe_tuning = Pipeline([
            ("pre", pre),
            ("clf", xgb.XGBClassifier(
                scale_pos_weight=spw, random_state=RANDOM_SEED,
                eval_metric="logloss", verbosity=0
            ))
        ])
        search = RandomizedSearchCV(
            xgb_pipe_tuning, param_distributions, n_iter=30,
            scoring="roc_auc", cv=cv, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)

        y_pred_t = search.best_estimator_.predict(X_test)
        y_prob_t = search.best_estimator_.predict_proba(X_test)[:, 1]
        cv_auc_t = cross_val_score(search.best_estimator_, X_train, y_train,
                                   cv=cv, scoring="roc_auc", n_jobs=-1)

        tuned_metrics = {
            "pipeline":    search.best_estimator_,
            "X_test":      X_test, "y_test": y_test,
            "y_pred":      y_pred_t, "y_prob": y_prob_t,
            "accuracy":    round(accuracy_score(y_test, y_pred_t) * 100, 2),
            "precision":   round(precision_score(y_test, y_pred_t) * 100, 2),
            "recall":      round(recall_score(y_test, y_pred_t) * 100, 2),
            "f1":          round(f1_score(y_test, y_pred_t) * 100, 2),
            "roc_auc":     round(roc_auc_score(y_test, y_prob_t) * 100, 2),
            "cv_auc_mean": round(cv_auc_t.mean() * 100, 2),
            "cv_auc_std":  round(cv_auc_t.std() * 100, 2),
            "cm":          confusion_matrix(y_test, y_pred_t),
            "clf_report":  classification_report(y_test, y_pred_t,
                                                  target_names=["Retained","Churned"]),
            "cat_features": cat_features,
            "num_features": num_features,
        }

        tuning_info = {
            "best_params":   search.best_params_,
            "best_cv_auc":   round(search.best_score_ * 100, 2),
            "base_roc_auc":  results["XGBoost"]["roc_auc"],
            "tuned_roc_auc": tuned_metrics["roc_auc"],
        }

        # Add tuned model if it improves over base XGBoost
        if tuned_metrics["roc_auc"] >= results["XGBoost"]["roc_auc"]:
            results["XGBoost (Tuned)"] = tuned_metrics
            tuning_info["selected"] = "XGBoost (Tuned)"
        else:
            tuning_info["selected"] = "XGBoost"
    except Exception:
        tuning_info = {"error": "Hyperparameter tuning failed; using base models."}

    return results, X_train, X_test, y_train, y_test, tuning_info


# ═════════════════════════════════════════════════════════════════════════════
# SHAP
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def compute_shap_values(_pipeline, X_sample):
    pre  = _pipeline.named_steps["pre"]
    clf  = _pipeline.named_steps["clf"]
    X_t  = pre.transform(X_sample)

    num_names = pre.transformers_[0][2]
    ohe_names = list(pre.transformers_[1][1].named_steps["ohe"].get_feature_names_out(
        pre.transformers_[1][2]
    ))
    feat_names = list(num_names) + ohe_names

    try:
        # Fix known SHAP/XGBoost compatibility issue with base_score format
        if hasattr(clf, "get_booster"):
            booster = clf.get_booster()
            cfg = booster.save_config()
            import json as _json
            cfg_dict = _json.loads(cfg)
            lm = cfg_dict.get("learner", {}).get("learner_model_param", {})
            bs = lm.get("base_score", "0.5")
            if isinstance(bs, str) and ("[" in bs or "]" in bs):
                bs_clean = bs.strip("[]")
                lm["base_score"] = bs_clean
                booster.save_config()  # trigger refresh
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_t)
    except Exception:
        # Fallback: use KernelExplainer with a small background sample
        bg = shap.sample(X_t, min(50, X_t.shape[0]))
        explainer = shap.KernelExplainer(clf.predict_proba, bg)
        sv = explainer.shap_values(X_t, nsamples=100)
        shap_vals = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv

    return shap_vals, feat_names


# ═════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═════════════════════════════════════════════════════════════════════════════

def assign_risk(prob):
    if prob >= HIGH_RISK_THRESHOLD:  return "High Risk"
    if prob >= LOW_RISK_THRESHOLD:   return "Medium Risk"
    return "Low Risk"


@st.cache_data(show_spinner=False)
def score_customers(_pipeline, df, cat_features, num_features):
    X      = df[cat_features + num_features]
    probs  = _pipeline.predict_proba(X)[:, 1]
    preds  = _pipeline.predict(X)

    return pd.DataFrame({
        "customerID":      df[ID_COL].values,
        "tenure":          df["tenure"].values,
        "Contract":        df["Contract"].values,
        "MonthlyCharges":  df["MonthlyCharges"].values,
        "TotalCharges":    df["TotalCharges"].values,
        "InternetService": df["InternetService"].values,
        "PaymentMethod":   df["PaymentMethod"].values,
        "ActualChurn":     df[TARGET_COL].values,
        "ChurnProbability":np.round(probs, 4),
        "PredictedChurn":  np.where(preds == 1, "Yes", "No"),
        "RiskCategory":    [assign_risk(p) for p in probs],
    })


# ═════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def get_recommendation(row):
    prob     = float(row.get("ChurnProbability", 0))
    contract = str(row.get("Contract", ""))
    tenure   = int(row.get("tenure", 0))
    charges  = float(row.get("MonthlyCharges", 0))
    internet = str(row.get("InternetService", ""))
    payment  = str(row.get("PaymentMethod", ""))

    if prob < LOW_RISK_THRESHOLD:
        return "Low churn risk. Continue standard engagement."

    recs = []
    if contract == "Month-to-month":
        recs.append("Contract Conversion: Offer incentivised annual or 2-year contract upgrade.")
    if tenure <= 12:
        recs.append("Onboarding Intervention: Proactive check-in, welcome discount, dedicated support.")
    if charges > 70:
        recs.append("Pricing Review: Personalised bundle offers or loyalty discounts.")
    if internet == "Fiber optic":
        recs.append("Fibre Quality: Investigate service quality; offer proactive technical support.")
    if payment == "Electronic check":
        recs.append("Payment Migration: Encourage auto-pay with a small incentive.")
    if prob >= HIGH_RISK_THRESHOLD and charges > 70:
        recs.append("HIGH VALUE AT RISK: Escalate to retention team immediately.")
    if not recs:
        recs.append("Monitor satisfaction; offer loyalty rewards to reduce churn risk.")
    return " | ".join(recs)


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def churn_donut(kpis):
    fig = go.Figure(go.Pie(
        labels=["Retained","Churned"],
        values=[kpis["retained"], kpis["churned"]],
        hole=0.62,
        marker=dict(colors=[COLORS["churn_no"], COLORS["churn_yes"]],
                    line=dict(color="#0f172a", width=2)),
        textfont=dict(size=13, color="#e2e8f0"),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Churn Distribution", font=dict(size=16, color="#a5b4fc")),
        showlegend=True,
        annotations=[dict(text=f"<b>{kpis['churn_rate']}%</b><br>Churn",
                          x=0.5, y=0.5, font=dict(size=16, color="#e2e8f0"), showarrow=False)]
    )
    return fig


def bar_churn_cat(grp, col, title):
    grp = grp.sort_values("ChurnRate", ascending=True)
    fig = go.Figure(go.Bar(
        x=grp["ChurnRate"], y=grp[col].astype(str), orientation="h",
        marker=dict(
            color=grp["ChurnRate"],
            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
            showscale=True,
            colorbar=dict(title="Churn %", tickfont=dict(color="#94a3b8")),
        ),
        text=grp["ChurnRate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=15, color="#a5b4fc")),
                      xaxis_title="Churn Rate (%)", height=max(280, len(grp)*55))
    return fig


def box_plot(df, col, title):
    fig = go.Figure()
    for label, color in [("No", COLORS["churn_no"]), ("Yes", COLORS["churn_yes"])]:
        fig.add_trace(go.Box(
            y=df.loc[df[TARGET_COL]==label, col],
            name="Retained" if label=="No" else "Churned",
            marker_color=color, line_color=color, boxmean="sd",
        ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=title, font=dict(size=15, color="#a5b4fc")), yaxis_title=col)
    return fig


def roc_curves_fig(results):
    fig = go.Figure()
    palette = ["#6366f1","#06b6d4","#f59e0b","#34d399"]
    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={r['roc_auc']}%)",
            line=dict(color=palette[i % 4], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#64748b", dash="dash", width=1.5),
        name="Random", showlegend=True,
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="ROC Curves", font=dict(size=15, color="#a5b4fc")),
                      xaxis_title="FPR", yaxis_title="TPR", height=420)
    return fig


def conf_matrix_fig(cm, name):
    labels = [["TN","FP"],["FN","TP"]]
    z_text = [[f"{labels[i][j]}<br>{cm[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: Retained","Pred: Churned"],
        y=["Actual: Retained","Actual: Churned"],
        colorscale=[[0,"#0f172a"],[1,"#6366f1"]],
        text=z_text, texttemplate="%{text}",
        textfont=dict(size=16, color="#fff"), showscale=False,
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"Confusion Matrix — {name}", font=dict(size=15, color="#a5b4fc")),
                      height=340)
    return fig


def feat_importance_fig(pipeline, cat_features, num_features, top_n=15):
    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    if not hasattr(clf, "feature_importances_"):
        return go.Figure()
    ohe_names = list(pre.transformers_[1][1].named_steps["ohe"].get_feature_names_out(cat_features))
    feat_names = list(num_features) + ohe_names
    fi = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance"],
                    colorscale=[[0, COLORS["secondary"]],[1, COLORS["primary"]]]),
        text=fi["importance"].apply(lambda x: f"{x:.4f}"),
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"Top {top_n} Feature Importances",
                                 font=dict(size=15, color="#a5b4fc")),
                      xaxis_title="Importance", height=max(350, top_n*28))
    return fig


def shap_bar_fig(shap_values, feat_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_s = pd.DataFrame({"feature": feat_names, "shap": mean_abs})
    df_s = df_s.sort_values("shap", ascending=False).head(top_n).sort_values("shap")
    fig = go.Figure(go.Bar(
        x=df_s["shap"], y=df_s["feature"], orientation="h",
        marker=dict(color=df_s["shap"],
                    colorscale=[[0,"#06b6d4"],[1,"#8b5cf6"]]),
        text=df_s["shap"].apply(lambda x: f"{x:.4f}"),
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"Global SHAP Feature Importance (Top {top_n})",
                                 font=dict(size=15, color="#a5b4fc")),
                      xaxis_title="Mean |SHAP value|", height=max(350, top_n*28))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def kpi_card(value, label, delta=""):
    d = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""<div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>{d}
    </div>"""


def kpi_row(cards, n=4):
    cols = st.columns(n)
    for i, c in enumerate(cards):
        with cols[i % n]:
            st.markdown(c, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight(html):
    st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

def page_overview(df, kpis, eda_stats, results, final_name, risk_df):
    st.markdown('<div class="page-title">📊 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown("**Customer Churn Prediction · Retention Intelligence System · Telco Dataset**")
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    hr_cnt = int((risk_df["RiskCategory"] == "High Risk").sum())
    hr_pct = round(hr_cnt / kpis["total_customers"] * 100, 1)
    rev_ar  = round(risk_df.loc[risk_df["RiskCategory"]=="High Risk","MonthlyCharges"].sum(), 2)

    kpi_row([
        kpi_card(f"{kpis['total_customers']:,}", "Total Customers"),
        kpi_card(f"{kpis['churned']:,}", "Churned Customers", "Actual in dataset"),
        kpi_card(f"{kpis['churn_rate']}%", "Overall Churn Rate", "Industry avg ~15-25%"),
        kpi_card(f"{kpis['retained']:,}", "Retained Customers"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_row([
        kpi_card(f"{hr_cnt:,}", "High-Risk Customers", "Model-predicted"),
        kpi_card(f"{hr_pct}%", "High-Risk %", f"Threshold >= {HIGH_RISK_THRESHOLD*100:.0f}%"),
        kpi_card(f"${rev_ar:,.0f}", "Est. Monthly Revenue at Risk", "High-risk customers"),
        kpi_card(f"{kpis['avg_tenure']} mo", "Average Tenure"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.plotly_chart(churn_donut(kpis), use_container_width=True)
    with c2:
        if "churn_by_Contract" in eda_stats:
            st.plotly_chart(bar_churn_cat(eda_stats["churn_by_Contract"], "Contract",
                                          "Churn Rate by Contract Type"), use_container_width=True)

    section("🔑 Top Churn Drivers (Feature Importance)")
    fr = results[final_name]
    cfi, cins = st.columns([1.5, 1])
    with cfi:
        st.plotly_chart(feat_importance_fig(fr["pipeline"], fr["cat_features"],
                                             fr["num_features"], 10), use_container_width=True)
    with cins:
        section("💡 Key Insights")
        if "churn_by_Contract" in eda_stats:
            cdf = eda_stats["churn_by_Contract"].set_index("Contract")
            m   = cdf.loc["Month-to-month","ChurnRate"] if "Month-to-month" in cdf.index else 0
            o   = cdf.loc["One year","ChurnRate"]       if "One year"       in cdf.index else 0
            t   = cdf.loc["Two year","ChurnRate"]       if "Two year"       in cdf.index else 0
            insight(f"📋 <strong>Contract:</strong> Month-to-month customers churn at "
                    f"<strong>{m:.1f}%</strong> vs {o:.1f}% (one-year) and {t:.1f}% (two-year).")
        if "churn_by_InternetService" in eda_stats:
            idf = eda_stats["churn_by_InternetService"].set_index("InternetService")
            fo  = idf.loc["Fiber optic","ChurnRate"] if "Fiber optic" in idf.index else 0
            ds  = idf.loc["DSL","ChurnRate"]         if "DSL"         in idf.index else 0
            insight(f"🌐 <strong>Internet:</strong> Fiber optic customers churn at "
                    f"<strong>{fo:.1f}%</strong> vs DSL at {ds:.1f}%.")
        if "churn_by_PaymentMethod" in eda_stats:
            pdf = eda_stats["churn_by_PaymentMethod"].set_index("PaymentMethod")
            ec  = pdf.loc["Electronic check","ChurnRate"] if "Electronic check" in pdf.index else 0
            insight(f"💳 <strong>Payment:</strong> Electronic check payers have "
                    f"<strong>{ec:.1f}%</strong> churn — highest across payment methods.")
        insight(f"🤖 <strong>Best Model:</strong> {final_name} — "
                f"ROC-AUC = <strong>{fr['roc_auc']}%</strong>, "
                f"Recall = <strong>{fr['recall']}%</strong>.")

    section("⚠️ High-Risk Customer Sample")
    hr = risk_df[risk_df["RiskCategory"]=="High Risk"].copy()
    hr_disp = hr[["customerID","tenure","Contract","MonthlyCharges","ChurnProbability"]].copy()
    hr_disp["ChurnProbability"] = hr_disp["ChurnProbability"].apply(lambda x: f"{x:.1%}")
    hr_disp = hr_disp.rename(columns={"customerID":"ID","tenure":"Tenure (mo)","ChurnProbability":"Churn Prob."})
    st.dataframe(hr_disp.head(10), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def page_churn_analysis(df, eda_stats):
    st.markdown('<div class="page-title">🔍 Customer & Churn Analysis</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    st.sidebar.markdown("### 🎛️ Analysis Filters")
    g_opts = ["All"] + sorted(df["gender"].unique().tolist())
    c_opts = ["All"] + sorted(df["Contract"].unique().tolist())
    i_opts = ["All"] + sorted(df["InternetService"].unique().tolist())
    s_opts = ["All","Senior Citizen","Non-Senior"]
    sg = st.sidebar.selectbox("Gender",           g_opts)
    sc = st.sidebar.selectbox("Contract Type",    c_opts)
    si = st.sidebar.selectbox("Internet Service", i_opts)
    ss = st.sidebar.selectbox("Senior Citizen",   s_opts)

    mask = pd.Series([True]*len(df), index=df.index)
    if sg != "All": mask &= df["gender"]           == sg
    if sc != "All": mask &= df["Contract"]         == sc
    if si != "All": mask &= df["InternetService"]  == si
    if ss == "Senior Citizen": mask &= df["SeniorCitizen"] == 1
    elif ss == "Non-Senior":   mask &= df["SeniorCitizen"] == 0
    dff = df[mask]

    if len(dff) == 0:
        st.warning("No customers match the selected filters."); return
    st.markdown(f"**Showing {len(dff):,} customers** matching selected filters.")

    c1, c2 = st.columns(2)
    with c1:
        fk = {"total_customers":len(dff),"churned":int(dff["Churn_Binary"].sum()),
              "retained":len(dff)-int(dff["Churn_Binary"].sum()),
              "churn_rate":round(dff["Churn_Binary"].mean()*100,2)}
        st.plotly_chart(churn_donut(fk), use_container_width=True)
    with c2:
        fig_sc = px.scatter(dff, x="tenure", y="MonthlyCharges", color=TARGET_COL,
                             color_discrete_map={"No":COLORS["churn_no"],"Yes":COLORS["churn_yes"]},
                             opacity=0.5)
        fig_sc.update_layout(**CHART_LAYOUT,
                              title=dict(text="Tenure vs Monthly Charges by Churn",
                                         font=dict(size=15,color="#a5b4fc")))
        st.plotly_chart(fig_sc, use_container_width=True)

    section("📊 Churn Rate by Segment")
    t1, t2, t3, t4 = st.tabs(["Demographics","Services","Contract & Billing","Charges"])

    def grp_chart(col_name, title, parent):
        g = dff.groupby(col_name)["Churn_Binary"].agg(["mean","count"]).reset_index()
        g.columns = [col_name,"ChurnRate","Count"]; g["ChurnRate"] *= 100
        with parent:
            st.plotly_chart(bar_churn_cat(g, col_name, title), use_container_width=True)

    with t1:
        a1, a2 = st.columns(2)
        grp_chart("gender","Churn by Gender",a1)
        gsc = dff.groupby("SeniorCitizen")["Churn_Binary"].agg(["mean","count"]).reset_index()
        gsc.columns = ["SeniorCitizen","ChurnRate","Count"]; gsc["ChurnRate"] *= 100
        gsc["SeniorCitizen"] = gsc["SeniorCitizen"].map({0:"Non-Senior",1:"Senior Citizen"})
        with a2:
            st.plotly_chart(bar_churn_cat(gsc,"SeniorCitizen","Churn by Senior Status"),
                            use_container_width=True)
        a3, a4 = st.columns(2)
        grp_chart("Partner","Churn by Partner",a3)
        grp_chart("Dependents","Churn by Dependents",a4)

    with t2:
        b1, b2 = st.columns(2)
        grp_chart("InternetService","Churn by Internet",b1)
        grp_chart("MultipleLines","Churn by Multiple Lines",b2)
        b3, b4 = st.columns(2)
        grp_chart("OnlineSecurity","Churn by Online Security",b3)
        grp_chart("TechSupport","Churn by Tech Support",b4)
        b5, b6 = st.columns(2)
        grp_chart("StreamingTV","Churn by Streaming TV",b5)
        grp_chart("StreamingMovies","Churn by Streaming Movies",b6)

    with t3:
        c1c, c2c = st.columns(2)
        grp_chart("Contract","Churn by Contract",c1c)
        grp_chart("PaymentMethod","Churn by Payment Method",c2c)
        c3c, c4c = st.columns(2)
        grp_chart("PaperlessBilling","Churn by Paperless Billing",c3c)

        # Tenure group chart
        tg = dff.groupby("TenureGroup")["Churn_Binary"].agg(["mean","count"]).reset_index()
        tg.columns = ["TenureGroup","ChurnRate","Count"]; tg["ChurnRate"] *= 100
        order = ["0-12 months","13-24 months","25-48 months","49+ months"]
        tg["TenureGroup"] = pd.Categorical(tg["TenureGroup"],categories=order,ordered=True)
        tg = tg.sort_values("TenureGroup")
        with c4c:
            st.plotly_chart(bar_churn_cat(tg,"TenureGroup","Churn by Tenure Group"),
                            use_container_width=True)
            t0  = tg.loc[tg["TenureGroup"]=="0-12 months","ChurnRate"].values
            t49 = tg.loc[tg["TenureGroup"]=="49+ months","ChurnRate"].values
            if len(t0) and len(t49):
                insight(f"New customers (0-12 months) churn at <strong>{t0[0]:.1f}%</strong> "
                        f"vs <strong>{t49[0]:.1f}%</strong> for 49+ months. "
                        f"Early engagement is critical.")

    with t4:
        d1, d2 = st.columns(2)
        with d1: st.plotly_chart(box_plot(dff,"MonthlyCharges","Monthly Charges by Churn"),
                                  use_container_width=True)
        with d2: st.plotly_chart(box_plot(dff,"TotalCharges","Total Charges by Churn"),
                                  use_container_width=True)
        st.plotly_chart(box_plot(dff,"tenure","Tenure by Churn"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

def page_model_perf(results, final_name, tuning_info):
    st.markdown('<div class="page-title">🤖 ML Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    section("📊 Model Comparison Table")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":         name,
            "Accuracy (%)":  r["accuracy"],
            "Precision (%)": r["precision"],
            "Recall (%)":    r["recall"],
            "F1 (%)":        r["f1"],
            "ROC-AUC (%)":   r["roc_auc"],
            f"CV-AUC ({CV_FOLDS}-fold)": f"{r['cv_auc_mean']}% ± {r['cv_auc_std']}%",
            "Selected":      "✅ Final Model" if name == final_name else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fr = results[final_name]
    insight(f"<strong>Final Model: {final_name}</strong> — Selected for highest ROC-AUC "
            f"({fr['roc_auc']}%) and Recall ({fr['recall']}%). For churn prediction, "
            f"Recall is prioritised to minimise missed churners (false negatives). "
            f"ROC-AUC provides threshold-independent discrimination ability.")

    # ── Hyperparameter Tuning Results ───────────────────────────────────────
    section("🔧 Hyperparameter Tuning (XGBoost)")
    if "error" in tuning_info:
        st.warning(tuning_info["error"])
    elif tuning_info:
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Best Parameters (RandomizedSearchCV, 30 iterations):**")
            params_df = pd.DataFrame([
                {"Parameter": k.replace("clf__", ""), "Value": str(v)}
                for k, v in tuning_info.get("best_params", {}).items()
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        with tc2:
            kpi_row([
                kpi_card(f"{tuning_info.get('best_cv_auc', '-')}%", "Tuned CV AUC"),
                kpi_card(f"{tuning_info.get('base_roc_auc', '-')}%", "Base XGBoost AUC"),
            ], 2)
            st.markdown("<br>", unsafe_allow_html=True)
            kpi_row([
                kpi_card(f"{tuning_info.get('tuned_roc_auc', '-')}%", "Tuned Test AUC"),
                kpi_card(tuning_info.get('selected', '-'), "Selected Variant"),
            ], 2)
        if tuning_info.get("tuned_roc_auc", 0) >= tuning_info.get("base_roc_auc", 0):
            insight("Tuned XGBoost improves upon or matches the base XGBoost and is included "
                    "in the model comparison above.")
        else:
            insight("Tuning did not improve over the base XGBoost. Base model retained.")
    else:
        st.info("No tuning information available.")

    section("📈 ROC Curves")
    st.plotly_chart(roc_curves_fig(results), use_container_width=True)

    section("🔬 Per-Model Details")
    model_tabs = st.tabs(list(results.keys()))
    for tab, (name, r) in zip(model_tabs, results.items()):
        with tab:
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(conf_matrix_fig(r["cm"], name), use_container_width=True)
            with cb:
                kpi_row([kpi_card(f"{r['accuracy']}%","Accuracy"),
                         kpi_card(f"{r['precision']}%","Precision")], 2)
                st.markdown("<br>", unsafe_allow_html=True)
                kpi_row([kpi_card(f"{r['recall']}%","Recall"),
                         kpi_card(f"{r['roc_auc']}%","ROC-AUC")], 2)
                st.markdown(f"**{CV_FOLDS}-fold CV AUC:** {r['cv_auc_mean']}% ± {r['cv_auc_std']}%")
            st.markdown("**Classification Report:**")
            st.code(r["clf_report"], language="text")

    section("🔍 Feature Importance (Final Model)")
    st.plotly_chart(feat_importance_fig(fr["pipeline"], fr["cat_features"],
                                        fr["num_features"]), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP
# ═════════════════════════════════════════════════════════════════════════════

def page_shap(final_pipeline, df, cat_features, num_features, risk_df):
    st.markdown('<div class="page-title">🔬 SHAP Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)
    insight("<strong>Note:</strong> SHAP values describe feature contributions to predictions. "
            "They do <em>not</em> imply causation. Language: 'contributes to prediction'.")

    with st.spinner("Computing SHAP values (sampling 500 customers)..."):
        sample_df = df.sample(min(500, len(df)), random_state=RANDOM_SEED)
        X_sample  = sample_df[cat_features + num_features]
        shap_vals, feat_names = compute_shap_values(final_pipeline, X_sample)

    st.plotly_chart(shap_bar_fig(shap_vals, feat_names, 15), use_container_width=True)

    section("🎯 Individual Customer SHAP Explanation")
    top_ids = risk_df.sort_values("ChurnProbability", ascending=False)["customerID"].head(50).tolist()
    sel_id  = st.selectbox("Select a High-Risk Customer ID", top_ids)

    if sel_id:
        cust = df[df[ID_COL]==sel_id].iloc[0]
        X_c  = cust[cat_features + num_features].to_frame().T
        sv, fn = compute_shap_values(final_pipeline, X_c)

        shap_df = pd.DataFrame({"Feature": fn, "SHAP": sv[0]})
        shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index).head(12)
        shap_df = shap_df.sort_values("SHAP", key=abs)

        colors = ["#f87171" if v > 0 else "#34d399" for v in shap_df["SHAP"]]
        fig_c = go.Figure(go.Bar(
            x=shap_df["SHAP"], y=shap_df["Feature"], orientation="h",
            marker_color=colors,
            text=shap_df["SHAP"].apply(lambda x: f"{x:+.4f}"),
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig_c.update_layout(**CHART_LAYOUT,
                             title=dict(text=f"SHAP Explanation — {sel_id}",
                                        font=dict(size=15,color="#a5b4fc")),
                             xaxis_title="SHAP Value (positive = increases churn risk)",
                             height=420)
        st.plotly_chart(fig_c, use_container_width=True)

        prob_val = risk_df.loc[risk_df["customerID"]==sel_id,"ChurnProbability"].values[0]
        risk_cat = risk_df.loc[risk_df["customerID"]==sel_id,"RiskCategory"].values[0]
        pos_feats = shap_df[shap_df["SHAP"]>0].tail(3)["Feature"].tolist()
        neg_feats = shap_df[shap_df["SHAP"]<0].tail(3)["Feature"].tolist()

        insight(f"<strong>Customer {sel_id}</strong> has churn probability "
                f"<strong>{prob_val:.1%}</strong> ({risk_cat}).<br>"
                f"<strong>Risk-increasing factors:</strong> {', '.join(pos_feats) or 'None'}<br>"
                f"<strong>Risk-reducing factors:</strong> {', '.join(neg_feats) or 'None'}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER PREDICTION FORM
# ═════════════════════════════════════════════════════════════════════════════

def page_prediction(final_pipeline, cat_features, num_features, df):
    st.markdown('<div class="page-title">🎯 Customer Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details to receive a real-time churn risk prediction.")
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            gender     = st.selectbox("Gender",          ["Male","Female"])
            senior     = st.selectbox("Senior Citizen",  ["No","Yes"])
            partner    = st.selectbox("Partner",         ["Yes","No"])
            dependents = st.selectbox("Dependents",      ["No","Yes"])
        with c2:
            st.markdown("**Account**")
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract",          ["Month-to-month","One year","Two year"])
            paperless= st.selectbox("Paperless Billing", ["Yes","No"])
            payment  = st.selectbox("Payment Method",    [
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ])
        with c3:
            st.markdown("**Charges**")
            monthly  = st.number_input("Monthly Charges ($)",  0.0, 200.0, 65.0, step=0.5)
            total    = st.number_input("Total Charges ($)",     0.0, 10000.0, float(monthly*tenure), step=1.0)

        st.markdown("---")
        st.markdown("**Services**")
        s1,s2,s3,s4 = st.columns(4)
        with s1:
            phone  = st.selectbox("Phone Service",   ["Yes","No"])
            multi  = st.selectbox("Multiple Lines",  ["No","Yes","No phone service"])
        with s2:
            inet   = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
            osec   = st.selectbox("Online Security", ["No","Yes","No internet service"])
        with s3:
            obk    = st.selectbox("Online Backup",   ["Yes","No","No internet service"])
            dprot  = st.selectbox("Device Protect",  ["No","Yes","No internet service"])
        with s4:
            tsupp  = st.selectbox("Tech Support",    ["No","Yes","No internet service"])
            stv    = st.selectbox("Streaming TV",    ["No","Yes","No internet service"])
            smv    = st.selectbox("Streaming Movies",["No","Yes","No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        svc_list = [phone, multi, inet, osec, obk, dprot, tsupp, stv, smv]
        svc_cnt  = sum(1 for v in svc_list if v in ["Yes","DSL","Fiber optic"])
        cps      = monthly / svc_cnt if svc_cnt > 0 else monthly

        inp = {
            "gender":gender,"SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner,"Dependents":dependents,"tenure":tenure,
            "PhoneService":phone,"MultipleLines":multi,"InternetService":inet,
            "OnlineSecurity":osec,"OnlineBackup":obk,"DeviceProtection":dprot,
            "TechSupport":tsupp,"StreamingTV":stv,"StreamingMovies":smv,
            "Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,
            "MonthlyCharges":monthly,"TotalCharges":total,
            "ServiceCount":svc_cnt,"ChargePerService":cps,
            "IsMonthToMonth":1 if contract=="Month-to-month" else 0,
            "HasProtection":1 if (tsupp=="Yes" or osec=="Yes") else 0,
            "IsElectronicCheck":1 if payment=="Electronic check" else 0,
        }

        X_inp = pd.DataFrame([inp])[cat_features + num_features]
        prob  = float(final_pipeline.predict_proba(X_inp)[0,1])
        pred  = int(final_pipeline.predict(X_inp)[0])
        risk  = assign_risk(prob)
        col_map = {"High Risk":"#f87171","Medium Risk":"#fbbf24","Low Risk":"#34d399"}

        st.markdown('<hr class="fancy">', unsafe_allow_html=True)
        section("📋 Prediction Result")
        kpi_row([
            kpi_card(f"{prob:.1%}", "Churn Probability"),
            kpi_card("Will Churn" if pred==1 else "Will Retain", "Predicted Outcome"),
            kpi_card(risk, "Risk Category"),
        ], 3)

        section("💡 Recommended Business Action")
        rec = get_recommendation(inp)
        for line in rec.split(" | "):
            if line.strip():
                insight(line.strip())

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            number={"suffix":"%","font":{"size":36,"color":"#e2e8f0"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#94a3b8"},
                "bar":{"color":col_map.get(risk,"#e2e8f0")},
                "steps":[
                    {"range":[0,35],"color":"rgba(52,211,153,.2)"},
                    {"range":[35,65],"color":"rgba(251,191,36,.2)"},
                    {"range":[65,100],"color":"rgba(248,113,113,.2)"},
                ],
            },
            delta={"reference":26.5,"valueformat":".1f","suffix":"% vs avg"}
        ))
        gauge.update_layout(**CHART_LAYOUT, height=320,
                            title=dict(text="Churn Risk Gauge",font=dict(size=15,color="#a5b4fc")))
        st.plotly_chart(gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — HIGH-RISK CUSTOMERS
# ═════════════════════════════════════════════════════════════════════════════

def page_high_risk(risk_df, df):
    st.markdown('<div class="page-title">⚠️ High-Risk Customer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    hr = risk_df[risk_df["RiskCategory"]=="High Risk"]
    mr = risk_df[risk_df["RiskCategory"]=="Medium Risk"]
    lr = risk_df[risk_df["RiskCategory"]=="Low Risk"]

    kpi_row([
        kpi_card(f"{len(hr):,}", "High Risk", f">= {HIGH_RISK_THRESHOLD*100:.0f}%"),
        kpi_card(f"{len(mr):,}", "Medium Risk", f"{LOW_RISK_THRESHOLD*100:.0f}-{HIGH_RISK_THRESHOLD*100:.0f}%"),
        kpi_card(f"{len(lr):,}", "Low Risk", f"< {LOW_RISK_THRESHOLD*100:.0f}%"),
        kpi_card(f"${hr['MonthlyCharges'].sum():,.0f}", "Monthly Revenue at Risk"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    rc = risk_df["RiskCategory"].value_counts().reset_index()
    rc.columns = ["RiskCategory","Count"]
    fig_r = px.bar(rc, x="RiskCategory", y="Count",
                   color="RiskCategory",
                   color_discrete_map={"High Risk":"#f87171","Medium Risk":"#fbbf24","Low Risk":"#34d399"},
                   text="Count")
    fig_r.update_layout(**CHART_LAYOUT, showlegend=False,
                        title=dict(text="Risk Distribution",font=dict(size=15,color="#a5b4fc")))
    fig_r.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
    st.plotly_chart(fig_r, use_container_width=True)

    st.sidebar.markdown("### ⚠️ Risk Table Filters")
    min_p    = st.sidebar.slider("Min Churn Probability", 0.0, 1.0, HIGH_RISK_THRESHOLD, step=0.05)
    sort_col = st.sidebar.selectbox("Sort By", ["ChurnProbability","MonthlyCharges","tenure"])
    sort_asc = st.sidebar.checkbox("Ascending", value=False)

    disp = risk_df[risk_df["ChurnProbability"] >= min_p].sort_values(sort_col, ascending=sort_asc).copy()
    disp["Recommendation"] = disp.apply(lambda r: get_recommendation(r.to_dict()).split(" | ")[0], axis=1)

    section(f"📋 Customer Risk Table ({len(disp):,} customers)")
    dp = disp.copy()
    dp["ChurnProbability"] = dp["ChurnProbability"].apply(lambda x: f"{x:.1%}")
    dp["MonthlyCharges"]   = dp["MonthlyCharges"].apply(lambda x: f"${x:.2f}")
    dp["TotalCharges"]     = dp["TotalCharges"].apply(lambda x: f"${x:.2f}")
    st.dataframe(
        dp[["customerID","RiskCategory","ChurnProbability","tenure","Contract",
            "MonthlyCharges","TotalCharges","InternetService","PaymentMethod",
            "PredictedChurn","Recommendation"]],
        use_container_width=True, hide_index=True, height=450
    )

    section("⬇️ Download Results")
    dl1, dl2 = st.columns(2)
    raw_dl = risk_df[risk_df["ChurnProbability"] >= min_p].sort_values(sort_col, ascending=sort_asc).copy()
    raw_dl["Recommendation"] = raw_dl.apply(lambda r: get_recommendation(r.to_dict()).split(" | ")[0], axis=1)
    with dl1:
        st.download_button("📥 Download Filtered Customers CSV",
                           data=raw_dl.to_csv(index=False),
                           file_name="filtered_customers.csv", mime="text/csv",
                           use_container_width=True)
    with dl2:
        full = risk_df.copy()
        full["Recommendation"] = full.apply(lambda r: get_recommendation(r.to_dict()).split(" | ")[0], axis=1)
        st.download_button("📥 Download All Customer Predictions CSV",
                           data=full.to_csv(index=False),
                           file_name="all_customer_predictions.csv", mime="text/csv",
                           use_container_width=True)

    section("💎 High-Value Customers at Risk")
    hv_thresh = df["MonthlyCharges"].quantile(0.75)
    hv = risk_df[
        (risk_df["RiskCategory"].isin(["High Risk","Medium Risk"])) &
        (risk_df["MonthlyCharges"] >= hv_thresh)
    ].sort_values("ChurnProbability", ascending=False)

    insight(f"High-value threshold: Monthly Charges >= <strong>${hv_thresh:.2f}</strong> (75th percentile). "
            f"<strong>{len(hv):,}</strong> high-value customers at medium/high risk. "
            f"Est. monthly revenue at risk: <strong>${hv['MonthlyCharges'].sum():,.0f}</strong>.")

    hv_dp = hv.copy()
    hv_dp["ChurnProbability"] = hv_dp["ChurnProbability"].apply(lambda x: f"{x:.1%}")
    hv_dp["MonthlyCharges"]   = hv_dp["MonthlyCharges"].apply(lambda x: f"${x:.2f}")
    st.dataframe(hv_dp[["customerID","RiskCategory","ChurnProbability","tenure",
                         "Contract","MonthlyCharges","TotalCharges"]].head(20),
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BUSINESS INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

def page_business_insights(df, eda_stats, kpis, results, final_name, risk_df):
    st.markdown('<div class="page-title">💼 Business Insights & Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    fr = results[final_name]

    def get_cr(stats_key, col_name, val):
        if stats_key in eda_stats:
            tmp = eda_stats[stats_key].set_index(col_name)
            return tmp.loc[val,"ChurnRate"] if val in tmp.index else 0.0
        return 0.0

    mtm_r = get_cr("churn_by_Contract","Contract","Month-to-month")
    oy_r  = get_cr("churn_by_Contract","Contract","One year")
    ty_r  = get_cr("churn_by_Contract","Contract","Two year")
    fo_r  = get_cr("churn_by_InternetService","InternetService","Fiber optic")
    dsl_r = get_cr("churn_by_InternetService","InternetService","DSL")
    ec_r  = get_cr("churn_by_PaymentMethod","PaymentMethod","Electronic check")

    hr_cnt     = int((risk_df["RiskCategory"]=="High Risk").sum())
    rev_ar     = risk_df.loc[risk_df["RiskCategory"]=="High Risk","MonthlyCharges"].sum()
    hv_thresh  = df["MonthlyCharges"].quantile(0.75)
    hv_cnt     = int(((risk_df["RiskCategory"].isin(["High Risk"])) &
                      (risk_df["MonthlyCharges"] >= hv_thresh)).sum())
    avg_mc_ch  = round(df.loc[df["Churn_Binary"]==1,"MonthlyCharges"].mean(),2)
    avg_mc_ret = round(df.loc[df["Churn_Binary"]==0,"MonthlyCharges"].mean(),2)
    avg_ten_ch = round(df.loc[df["Churn_Binary"]==1,"tenure"].mean(),1)
    avg_ten_r  = round(df.loc[df["Churn_Binary"]==0,"tenure"].mean(),1)

    insights_data = [
        ("📋","Contract Type is the Strongest Churn Signal",
         f"Month-to-month customers churn at {mtm_r:.1f}% vs {oy_r:.1f}% (one-year) and {ty_r:.1f}% (two-year).",
         "Customers on flexible contracts face no switching friction and leave more readily.",
         "Large month-to-month cohort represents significant churn exposure.",
         "Contract-to-longer-term conversions can structurally reduce churn.",
         "Offer time-limited upgrade incentives (bill credits, price locks) to month-to-month customers."),
        ("🌐","Fibre Optic Shows Elevated Churn",
         f"Fiber optic customers churn at {fo_r:.1f}% vs {dsl_r:.1f}% for DSL.",
         "Despite higher investment, fibre customers may face unmet service/price expectations.",
         "Fibre customers pay more — their churn creates disproportionate revenue impact.",
         "Resolving fibre service quality issues can improve satisfaction in this high-value segment.",
         "Launch a fibre NPS survey; deploy proactive technical support for flagged accounts."),
        ("💳","Electronic Check Correlates with High Churn",
         f"Electronic check users churn at {ec_r:.1f}% — highest across all payment methods.",
         "Manual payment may correlate with lower engagement and weaker service commitment.",
         "High-churn payment segment reduces efficiency of blanket retention campaigns.",
         "Encouraging auto-pay may correlate with improved retention and lower churn rates.",
         "Offer a monthly discount or bill credit for switching to automatic payment."),
        ("🚀","New Customers Are Highest Churn Risk",
         f"Churned customers average {avg_ten_ch} months tenure vs {avg_ten_r} months for retained.",
         "Early-stage customers are still evaluating; poor onboarding leads to early exit.",
         "Customer acquisition costs are wasted when new customers churn quickly.",
         "Structured onboarding improves satisfaction and reduces early-stage churn.",
         "Deploy a 90-day new-customer journey with check-ins, dedicated support, and a satisfaction guarantee."),
        ("💰","Churned Customers Pay Higher Monthly Charges",
         f"Churned customers pay ${avg_mc_ch}/month vs ${avg_mc_ret}/month for retained.",
         "Higher charges without perceived value drive customers to seek alternatives.",
         f"Est. ${rev_ar:,.0f}/month revenue at risk from high-risk customers alone.",
         "Value perception improvements can reduce price-driven churn without price reductions.",
         "Offer value-added bundles or loyalty rewards for high-charge customers at risk."),
        ("🔐","Lack of Protective Services Increases Churn",
         "Customers without TechSupport or OnlineSecurity have substantially higher churn rates.",
         "Protective services create stickiness and demonstrate ongoing value delivery.",
         "Customers without protective services have fewer reasons to stay.",
         "Free or discounted trials of protective services can improve retention.",
         "Include a 3-month complimentary TechSupport/OnlineSecurity trial in retention packages."),
        ("⭐",f"High-Value Customers at Risk Demand Priority Attention",
         f"{hv_cnt} high-value customers (>= ${hv_thresh:.0f}/mo) are at medium or high churn risk.",
         "Losing high-value customers creates outsized revenue impact relative to their count.",
         f"These {hv_cnt} customers represent concentrated revenue exposure.",
         "Personalised retention outreach is most cost-effective for high-value customers.",
         "Escalate all high-value, high-risk customers to a dedicated retention team with tailored offers."),
        ("🤖","ML Model Provides Actionable Early Warning",
         f"{final_name} achieves {fr['roc_auc']}% ROC-AUC and {fr['recall']}% Recall on the test set.",
         f"The model identifies {fr['recall']:.0f}% of churners before they leave.",
         f"{100-fr['recall']:.1f}% of churners are missed (false negatives). Over-reliance without human review is risky.",
         "Regular scoring cadence creates a continuous early-warning system.",
         "Score all customers monthly; route high-risk customers to retention team within 48 hours."),
    ]

    for icon, title, fact, ins_txt, risk_txt, opp, action in insights_data:
        with st.expander(f"{icon} {title}", expanded=False):
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown(f"**📊 FACT:** {fact}")
                st.markdown(f"**💡 INSIGHT:** {ins_txt}")
            with ci2:
                st.markdown(f"**⚠️ RISK:** {risk_txt}")
                st.markdown(f"**🚀 OPPORTUNITY:** {opp}")
            st.markdown(f'<div class="rec-card"><div class="rec-title">✅ RECOMMENDED ACTION</div>'
                        f'<div class="rec-body">{action}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="fancy">', unsafe_allow_html=True)
    st.markdown(
        '<div class="alert-warning">⚠️ <strong>Disclaimer:</strong> All insights are based on the publicly '
        'available Telco Customer Churn dataset. Predictions are statistical estimates, not guaranteed outcomes. '
        'Correlation does not imply causation. Revenue-at-risk figures are estimates based on MonthlyCharges. '
        'Recommendations should be validated against current organisational data before deployment.</div>',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8 — DATASET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

def page_dataset_explorer(df_raw, df, cleaning_log, feat_log):
    st.markdown('<div class="page-title">🗂️ Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy">', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Overview","Data Sample","Cleaning Log","Feature Engineering"])

    with t1:
        info = inspect_dataset(df_raw)
        kpi_row([kpi_card(f"{info['n_rows']:,}","Total Rows"),
                 kpi_card(f"{info['n_cols']}","Total Columns"),
                 kpi_card(f"{info['duplicates']}","Duplicate Rows"),
                 kpi_card(f"{info['churn_rate']}%","Churn Rate")])
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Column Information**")
            dtype_df = pd.DataFrame({
                "Column":  list(info["dtypes"].keys()),
                "DType":   [str(v) for v in info["dtypes"].values()],
                "Missing": [info["missing"][k] for k in info["dtypes"].keys()],
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Target Distribution**")
            td = info["target_dist"]
            fig_td = go.Figure(go.Bar(
                x=list(td.keys()), y=list(td.values()),
                marker_color=[COLORS["churn_no"],COLORS["churn_yes"]],
                text=list(td.values()), textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_td.update_layout(**CHART_LAYOUT,
                                  title=dict(text="Churn Distribution (Raw)",
                                             font=dict(size=14,color="#a5b4fc")))
            st.plotly_chart(fig_td, use_container_width=True)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with t2:
        search = st.text_input("Search by CustomerID (partial match)")
        n_show = st.slider("Rows to display", 5, 100, 20)
        disp   = df.copy()
        if search:
            disp = disp[disp[ID_COL].str.contains(search, case=False, na=False)]
        st.dataframe(disp.head(n_show), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(n_show,len(disp))} of {len(disp):,} matching rows.")

    with t3:
        st.markdown("**Data Cleaning Steps Applied**")
        for i, step in enumerate(cleaning_log, 1):
            st.markdown(f"`{i}.` {step}")

    with t4:
        st.markdown("**Engineered Features**")
        for i, step in enumerate(feat_log, 1):
            st.markdown(f"`{i}.` {step}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    # Sidebar nav
    st.sidebar.markdown(
        "<div style='text-align:center;padding:1rem 0;'>"
        "<span style='font-size:2rem;'>📊</span><br>"
        "<span style='color:#a5b4fc;font-size:1.05rem;font-weight:700;'>Churn Intelligence<br>System</span><br>"
        "<span style='color:#64748b;font-size:.75rem;'>Telco Customer Churn · v1.0</span></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    pages = [
        "📊 Executive Overview",
        "🔍 Customer & Churn Analysis",
        "🤖 ML Model Performance",
        "🔬 SHAP Explainability",
        "🎯 Customer Risk Prediction",
        "⚠️ High-Risk Customers",
        "💼 Business Insights",
        "🗂️ Dataset Explorer",
    ]
    selected = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color:#64748b;font-size:.75rem;'>"
        "📂 Telco Customer Churn<br>"
        "🔗 <a href='https://www.kaggle.com/datasets/blastchar/telco-customer-churn' "
        "target='_blank' style='color:#6366f1;'>Kaggle Source</a></div>",
        unsafe_allow_html=True
    )

    # Check dataset
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found: **{DATA_PATH}**\n\n"
            "Please place the CSV file in the same directory as this script.\n\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n\n"
            "Expected filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`"
        )
        st.stop()

    # Load & process
    with st.spinner("Loading and preprocessing data..."):
        df_raw             = load_raw_data(DATA_PATH)
        df_clean, cl_log   = clean_data(df_raw)
        df_feat, feat_log  = engineer_features(df_clean)

    with st.spinner("Computing statistics..."):
        eda_stats = compute_eda_stats(df_feat)
        kpis      = compute_kpis(df_feat)

    with st.spinner("Training ML models and tuning hyperparameters (first run may take ~2 minutes)..."):
        results, X_train, X_test, y_train, y_test, tuning_info = train_all_models(df_feat)

    # Select final model: highest ROC-AUC; tie-break by Recall
    final_name = max(results, key=lambda m: (results[m]["roc_auc"], results[m]["recall"]))
    final_pipe = results[final_name]["pipeline"]
    cat_feat   = results[final_name]["cat_features"]
    num_feat   = results[final_name]["num_features"]

    with st.spinner("Scoring all customers..."):
        risk_df = score_customers(final_pipe, df_feat, cat_feat, num_feat)

    kpis["high_risk_count"] = int((risk_df["RiskCategory"]=="High Risk").sum())

    # Route
    if   selected == pages[0]: page_overview(df_feat, kpis, eda_stats, results, final_name, risk_df)
    elif selected == pages[1]: page_churn_analysis(df_feat, eda_stats)
    elif selected == pages[2]: page_model_perf(results, final_name, tuning_info)
    elif selected == pages[3]: page_shap(final_pipe, df_feat, cat_feat, num_feat, risk_df)
    elif selected == pages[4]: page_prediction(final_pipe, cat_feat, num_feat, df_feat)
    elif selected == pages[5]: page_high_risk(risk_df, df_feat)
    elif selected == pages[6]: page_business_insights(df_feat, eda_stats, kpis, results, final_name, risk_df)
    elif selected == pages[7]: page_dataset_explorer(df_raw, df_feat, cl_log, feat_log)


if __name__ == "__main__":
    main()
