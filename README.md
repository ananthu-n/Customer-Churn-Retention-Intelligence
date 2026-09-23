# 📊 Customer Churn Prediction and Retention Intelligence System

> **End-to-End Machine Learning Project** — From raw data to actionable business intelligence.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)](https://shap.readthedocs.io)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Objectives](#-objectives)
4. [Dataset Source](#-dataset-source)
5. [Dataset Description](#-dataset-description)
6. [Features](#-features)
7. [Target Variable](#-target-variable)
8. [Methodology](#-methodology)
9. [Data Preprocessing](#-data-preprocessing)
10. [Exploratory Data Analysis](#-exploratory-data-analysis)
11. [Machine Learning Models](#-machine-learning-models)
12. [Model Evaluation](#-model-evaluation)
13. [Hyperparameter Tuning](#-hyperparameter-tuning)
14. [Explainability](#-explainability)
15. [Dashboard Features](#-dashboard-features)
16. [Business Insights](#-business-insights)
17. [Recommendations](#-recommendations)
18. [Project Structure](#-project-structure)
19. [Installation Instructions](#-installation-instructions)
20. [How to Run the Application](#-how-to-run-the-application)
21. [Example Usage](#-example-usage)
22. [Limitations](#-limitations)
23. [Future Improvements](#-future-improvements)
24. [Dataset Attribution & License](#-dataset-attribution--license)

---

## 🔍 Project Overview

This project builds a **complete Customer Churn Prediction and Retention Intelligence System** for a telecommunications company. It connects the full pipeline:

**DATA → CLEANING → EDA → KPIs → ML MODEL → MODEL EXPLANATION → CUSTOMER RISK → BUSINESS INSIGHTS → ACTIONABLE RECOMMENDATIONS**

The system uses machine learning to predict which customers are likely to leave and provides data-driven recommendations for retention interventions. All findings are presented through a professional interactive Streamlit dashboard.

---

## 💼 Business Problem

Customer churn directly impacts revenue and growth. The business wants to:

1. **Identify** customers who are at risk of leaving
2. **Understand** what factors are associated with churn
3. **Predict** which customers are most likely to churn
4. **Prioritise** high-value customers at risk
5. **Act** with targeted retention strategies based on customer profiles

The system answers:
- How many customers are currently in the dataset? → **7,043 customers**
- What is the overall churn rate? → **26.54%**
- Which customer groups have higher churn? → Month-to-month contract (42.7%), Fiber optic (41.9%), Electronic check (45.3%)
- What factors are associated with churn? → Contract type, internet service, tenure, monthly charges, payment method
- Which customers are predicted to be at high risk? → Generated per-customer risk scores
- What actions could the business take? → Targeted retention campaigns, contract conversion, onboarding support

---

## 🎯 Objectives

1. Clean and preprocess the Telco Customer Churn dataset
2. Perform comprehensive exploratory data analysis
3. Calculate business KPIs
4. Build and compare multiple classification models
5. Perform hyperparameter tuning
6. Apply SHAP explainability to the final model
7. Generate customer risk scores and segmentation
8. Create a rule-based business recommendation engine
9. Build a professional Streamlit dashboard
10. Document all findings in a reproducible format

---

## 📁 Dataset Source

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle |
| **Dataset Name** | Telco Customer Churn |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Filename** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Provider** | IBM Sample Data Sets (via BlastChar on Kaggle) |

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| **Rows** | 7,043 |
| **Columns** | 21 |
| **Duplicate Rows** | 0 |
| **Missing Values** | 11 blank strings in `TotalCharges` (new customers with tenure = 0) |
| **Target Column** | `Churn` (Yes / No) |
| **Target Distribution** | No: 5,174 (73.46%) · Yes: 1,869 (26.54%) |
| **Identifier Column** | `customerID` (excluded from modelling) |

---

## 📑 Features

### Original Features (21 columns)

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | String | Unique customer identifier (not used as feature) |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Whether the customer has a partner (Yes/No) |
| `Dependents` | Categorical | Whether the customer has dependents (Yes/No) |
| `tenure` | Numeric | Number of months the customer has stayed |
| `PhoneService` | Categorical | Whether the customer has phone service (Yes/No) |
| `MultipleLines` | Categorical | Whether the customer has multiple lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Yes / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Monthly charge amount ($) |
| `TotalCharges` | Numeric (stored as string) | Total charges to date ($) |
| `Churn` | Categorical | Yes / No — **Target variable** |

### Engineered Features (6 additional)

| Feature | Description | Leakage-Free |
|---------|-------------|:---:|
| `TenureGroup` | Binned tenure (0-12, 13-24, 25-48, 49+ months) | ✅ |
| `ServiceCount` | Number of active services subscribed | ✅ |
| `ChargePerService` | MonthlyCharges / ServiceCount | ✅ |
| `IsMonthToMonth` | Binary: 1 if month-to-month contract | ✅ |
| `HasProtection` | Binary: 1 if TechSupport or OnlineSecurity = Yes | ✅ |
| `IsElectronicCheck` | Binary: 1 if PaymentMethod = Electronic check | ✅ |

---

## 🎯 Target Variable

| Target | Column | Values | Encoding |
|--------|--------|--------|----------|
| Customer Churn | `Churn` | Yes / No | Yes → 1, No → 0 |

Class distribution: **73.46% Retained (No) · 26.54% Churned (Yes)** — moderate class imbalance handled via `class_weight="balanced"` and `scale_pos_weight`.

---

## 🔬 Methodology

```
Raw CSV Data
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering (6 new features)
    ↓
Exploratory Data Analysis
    ↓
Business KPI Calculation
    ↓
Stratified Train/Test Split (80/20)
    ↓
Model Training (4 algorithms + hyperparameter tuning)
    ↓
Model Evaluation & Selection
    ↓
SHAP Explainability
    ↓
Customer Risk Scoring & Segmentation
    ↓
Business Recommendation Engine
    ↓
Interactive Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

| Step | Action | Details |
|------|--------|---------|
| 1 | Duplicate detection | 0 duplicates found |
| 2 | Whitespace stripping | Applied to all string columns |
| 3 | `TotalCharges` conversion | 11 blank strings (new customers, tenure ≈ 0) replaced with 0.0; converted to float64 |
| 4 | Target encoding | `Churn`: Yes → 1, No → 0 (stored in `Churn_Binary`) |
| 5 | `SeniorCitizen` | Already binary (0/1), no conversion needed |
| 6 | Missing value check | No remaining NaN values after cleaning |
| 7 | Categorical encoding | OneHotEncoder (in sklearn Pipeline) for model training |
| 8 | Numerical scaling | StandardScaler (in sklearn Pipeline) for numerical features |
| 9 | Data leakage prevention | All transformations applied via ColumnTransformer pipeline fitted only on training data |

**Final cleaned shape:** 7,043 rows × 28 columns (including engineered features)

---

## 📈 Exploratory Data Analysis

Key EDA analyses performed:

| Analysis | Key Finding |
|----------|------------|
| Overall churn | 26.54% churn rate (1,869 of 7,043 customers) |
| Contract type | Month-to-month: **42.7%** churn vs One year: **11.3%** vs Two year: **2.8%** |
| Internet service | Fiber optic: **41.9%** churn vs DSL: **19.0%** vs No internet: **7.4%** |
| Payment method | Electronic check: **45.3%** churn — highest of all methods |
| Tenure | Churned customers avg **~18 months** vs retained avg **~38 months** |
| Monthly charges | Churned customers pay **~$74/month** vs retained **~$61/month** |
| Senior citizens | Higher churn rate than non-seniors |
| No tech support | Significantly higher churn without TechSupport or OnlineSecurity |
| Paperless billing | Customers with paperless billing churn at higher rates |

All charts are interactive (Plotly) and available in the Streamlit dashboard.

---

## 🤖 Machine Learning Models

### Models Trained

| Model | Class Imbalance Handling |
|-------|------------------------|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest (200 trees) | `class_weight="balanced"` |
| Gradient Boosting (200 trees) | Default (no explicit weighting) |
| XGBoost (200 trees) | `scale_pos_weight` (computed from training set) |
| XGBoost Tuned | `scale_pos_weight` + RandomizedSearchCV (30 iterations) |

### Pipeline Architecture

```
ColumnTransformer
├── Numerical (9 features) → StandardScaler
└── Categorical (15 features) → OneHotEncoder
         ↓
    Classifier
```

- Stratified 80/20 train/test split
- 5-fold stratified cross-validation
- No data leakage: ColumnTransformer fitted only on training data

---

## 📊 Model Evaluation

### Comparison Table (Test Set — 20%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-AUC (5-fold) |
|-------|----------|-----------|--------|-----|---------|----------------|
| Logistic Regression | 74.24% | 50.96% | **78.34%** | 61.75% | 84.20% | 84.56% ± 1.29% |
| Random Forest | 77.71% | 60.27% | 47.06% | 52.85% | 82.12% | 82.51% ± 1.07% |
| Gradient Boosting | 79.21% | **63.11%** | 52.14% | 57.10% | 83.52% | 83.81% ± 0.93% |
| XGBoost | 75.51% | 52.64% | 77.27% | 62.62% | 83.72% | 83.87% ± 1.13% |
| **XGBoost (Tuned)** | 75.66% | 52.70% | **81.02%** | **63.86%** | **84.71%** | **84.83%** ± N/A |

### Final Model Selection

**XGBoost (Tuned)** was selected as the final model based on:
- **Highest ROC-AUC (84.71%)** — Best threshold-independent discrimination
- **Highest Recall (81.02%)** — Catches the most churners (minimises false negatives)
- **Best F1 (63.86%)** — Best balance of precision and recall

> For churn prediction, Recall is prioritised over Accuracy because missing a churner (false negative) has greater business cost than a false alarm (false positive).

---

## 🔧 Hyperparameter Tuning

**Method:** RandomizedSearchCV (30 iterations, 5-fold stratified CV, ROC-AUC scoring)

**Search Space:**

| Parameter | Values Explored |
|-----------|----------------|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 4, 5, 6 |
| `learning_rate` | 0.05, 0.1, 0.15 |
| `subsample` | 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.8, 0.9, 1.0 |
| `min_child_weight` | 1, 3, 5 |

**Best Parameters Found:**

| Parameter | Best Value |
|-----------|-----------|
| `subsample` | 0.9 |
| `n_estimators` | 100 |
| `min_child_weight` | 1 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `colsample_bytree` | 0.9 |

**Result:** Tuned XGBoost achieved **84.83% CV-AUC** and **84.71% test AUC**, improving over the base XGBoost (83.72% test AUC).

---

## 🔬 Explainability

### SHAP (SHapley Additive exPlanations)

- **Global feature importance:** SHAP summary plot shows which features contribute most to churn predictions across all customers
- **Individual customer explanation:** For any selected customer, SHAP values explain which features pushed the prediction toward or away from churn
- **Top predictive features:** Contract type, tenure, internet service, monthly charges, and payment method are consistently the strongest contributors

> **Important:** SHAP values describe feature contributions to model predictions. They do **not** imply causation. Language used: "contributes to prediction" / "associated with prediction."

---

## 🖥️ Dashboard Features

The Streamlit dashboard contains 8 interactive pages:

| Page | Description |
|------|-------------|
| **📊 Executive Overview** | KPI cards, churn distribution, top drivers, key insights, high-risk sample |
| **🔍 Customer & Churn Analysis** | Interactive filters (gender, contract, internet, senior status), segmented charts, scatter plots |
| **🤖 ML Model Performance** | Model comparison table, ROC curves, confusion matrices, hyperparameter tuning results, feature importance |
| **🔬 SHAP Explainability** | Global SHAP importance, individual customer SHAP explanations |
| **🎯 Customer Risk Prediction** | Interactive prediction form — enter customer details, get real-time churn probability, risk category, and recommended action |
| **⚠️ High-Risk Customers** | Filterable/sortable risk table, high-value customers at risk, CSV download |
| **💼 Business Insights** | FACT → INSIGHT → RISK → OPPORTUNITY → ACTION framework for 8 key findings |
| **🗂️ Dataset Explorer** | Data overview, sample viewer, cleaning log, feature engineering log, customer search |

---

## 💡 Business Insights

| # | Insight | Data Support |
|---|---------|-------------|
| 1 | **Contract type is the strongest churn signal** | MTM: 42.7% vs 1-year: 11.3% vs 2-year: 2.8% |
| 2 | **Fiber optic shows elevated churn** | Fiber: 41.9% vs DSL: 19.0% |
| 3 | **Electronic check correlates with high churn** | E-check: 45.3% — highest across all payment methods |
| 4 | **New customers are highest churn risk** | Churned avg tenure ~18 months vs retained ~38 months |
| 5 | **Churned customers pay higher monthly charges** | Churned ~$74/mo vs retained ~$61/mo |
| 6 | **Lack of protective services increases churn** | Customers without TechSupport/OnlineSecurity churn more |
| 7 | **High-value customers at risk demand priority** | Customers with high charges AND high churn probability |
| 8 | **ML model provides actionable early warning** | 81% of churners identified before they leave |

---

## ✅ Recommendations

| Condition | Recommended Action |
|-----------|-------------------|
| High risk + Month-to-month contract | Offer incentivised annual or 2-year contract upgrade |
| High risk + Short tenure (≤ 12 months) | Proactive onboarding check-in, welcome discount, dedicated support |
| High risk + High monthly charges (> $70) | Personalised bundle offers or loyalty discounts |
| High risk + Fiber optic internet | Investigate service quality; offer proactive technical support |
| High risk + Electronic check payment | Encourage auto-pay with a small incentive |
| High risk + High value customer | Escalate to retention team immediately |

> **Disclaimer:** These are business suggestions based on observed data patterns, not guaranteed outcomes. Correlation does not imply causation.

---

## 📂 Project Structure

```
Customer-Churn-Retention-Intelligence/
│
├── customer_churn_project.py              # Complete application (single file)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (7,043 rows × 21 columns)
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── Customer_Churn_Project_Report.pdf       # Professional project report
```

---

## ⚙️ Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Customer-Churn-Retention-Intelligence.git
cd Customer-Churn-Retention-Intelligence

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure the dataset is present
# The file WA_Fn-UseC_-Telco-Customer-Churn.csv should be in the project root.
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

---

## 🚀 How to Run the Application

```bash
streamlit run customer_churn_project.py
```

The application will open in your default browser at `http://localhost:8501`.

**First launch** may take 1–2 minutes as all 5 models are trained and hyperparameter tuning is performed. Results are cached for subsequent page loads.

---

## 📸 Example Usage

1. **Executive Overview** — View KPIs, churn distribution, and top insights at a glance
2. **Customer Analysis** — Filter by gender, contract type, internet service, etc. to explore churn patterns
3. **Model Performance** — Compare all models side-by-side; review confusion matrices, ROC curves, and tuning results
4. **SHAP Explainability** — Select a high-risk customer to see exactly which features contribute to their prediction
5. **Risk Prediction** — Fill in a customer form to get instant churn probability, risk category, and recommended action
6. **High-Risk Table** — Sort and filter at-risk customers; download results as CSV for CRM integration
7. **Business Insights** — Review data-driven FACT → INSIGHT → RISK → OPPORTUNITY → ACTION summaries

---

## ⚠️ Limitations

1. **Dataset is publicly available** — Results are specific to this Telco dataset and may not generalise to other businesses without retraining
2. **Static dataset** — The model is trained on a snapshot; real-world deployment requires periodic retraining on current data
3. **Correlation ≠ Causation** — Feature importance and SHAP values describe statistical associations, not causal relationships
4. **Predicted churn ≠ Guaranteed churn** — Predictions are probabilistic estimates
5. **Revenue-at-risk is an estimate** — Calculated as the sum of MonthlyCharges for high-risk customers, not actual future revenue loss
6. **Class imbalance** — Handled via balanced class weights, but precision–recall trade-offs exist
7. **No external data** — The model uses only features available in the dataset; enrichment with NPS, support tickets, or usage data could improve predictions
8. **Model should be validated** on current organisational data before real-world deployment

---

## 🔮 Future Improvements

1. **Deep learning models** — Neural networks for pattern discovery in larger datasets
2. **Time-series churn prediction** — Incorporate temporal patterns in customer behaviour
3. **Customer Lifetime Value (CLV)** — Integrate CLV models for prioritisation
4. **A/B testing framework** — Measure the effectiveness of retention interventions
5. **Real-time scoring API** — Deploy the model as a REST API for CRM integration
6. **NPS and support ticket data** — Enrich features with customer sentiment data
7. **Automated retraining pipeline** — MLOps pipeline for periodic model updates
8. **Fairness analysis** — Audit model predictions for demographic bias

---

## 📜 Dataset Attribution & License

| Field | Value |
|-------|-------|
| **Dataset Name** | Telco Customer Churn |
| **Source** | [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Original Provider** | IBM Sample Data Sets |
| **License** | Data files © Original Authors. Please refer to the Kaggle dataset page for specific licensing terms. |
| **Usage** | This project uses the dataset for educational and demonstration purposes only. No ownership of the dataset is claimed. |

---

## 🙏 Acknowledgements

- **IBM** for the original Telco Customer Churn dataset
- **Kaggle** and **BlastChar** for hosting and distributing the dataset
- Open-source communities behind **scikit-learn**, **XGBoost**, **SHAP**, **Streamlit**, and **Plotly**

---

*Built with ❤️ for data-driven customer retention intelligence.*
