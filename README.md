# Insurance - Predictive Analytics Suite: Early Warning & Decision Support System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Data Science](https://img.shields.io/badge/Data_Science-Machine_Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple)

## 📑 Executive Summary
In the insurance sector, profitability is primarily threatened by two silent disruptors: **Customer Churn** (the loss of highly profitable, loyal policyholders) and **Insurance Fraud** (high-severity, malicious claims). Historically, these issues were addressed reactively, leading to significant financial leakage.

This project introduces a **proactive, dual-engine Artificial Intelligence platform** tailored for the enterprise ecosystem:
1. **Customer Retention Radar (Churn Model):** Anticipates policy non-renewal, allowing for targeted, proactive retention campaigns.
2. **Financial Armor (Fraud Model):** A highly specialized, **Cost-Sensitive Learning** engine that dynamically balances the operational cost of investigating honest customers against the catastrophic financial risk of approving fraudulent claims. 

By integrating this system into production, the business transitions from manual oversight to automated **"Fast-Track"** settlements for low-risk claims, drastically improving Customer Net Promoter Score (NPS) while reserving expert human investigation exclusively for high-risk flags.

---

## 🏗️ Architecture & Tech Stack

The pipeline is modularly designed following MLOps best practices, ensuring clear separation of concerns from data ingestion to model explainability.

*   **`eda_processor.py`**: Handles raw data ingestion, missing value imputation, and complex aggregations (transforming claim-level data to policy-level KPIs like `loss_ratio` and `claim_frequency`).
*   **`feature_engineering.py`**: Generates mathematically rigorous features. Strictly prevents Data Leakage by masking post-event variables (like settlement time) for the Fraud model, while heavily utilizing them for the Churn model (e.g., `profitability`, `delay_risk`).
*   **`churn_model.py`**: Deploys a stratified Machine Learning pipeline using `GridSearchCV` to optimize XGBoost, LightGBM, and Random Forest. Thresholds are optimized purely for the **F1-Score**.
*   **`fraud_model.py`**: Implements **Cost-Matrix Optimization**. Realizing that a False Negative ($195 cost) is vastly more damaging than a False Positive ($5 operational friction), the optimal decision threshold is calculated mathematically to strictly minimize the net business cost.

**Tech Stack:** Python, Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, SHAP, Seaborn.

---

## 📊 Key Insights & Explainable AI (XAI)

Based on SHAP (SHapley Additive exPlanations) values and historical feature importance analysis, the models reveal highly actionable business intelligence:

### 1. Customer Churn Dynamics (Retention Radar)
*   **The Critical First 6 Months:** The `Churn Rate by Tenure` analysis shows alarming attrition rates (>50%) within the `(0, 3]` and `(3, 6]` month bins. Loyalty stabilizes significantly only after the 12-18 month mark.
*   **Digital Disconnect:** `cat__sales_channel_Online` is the top categorical predictor for churn. Customers acquired digitally show the least brand loyalty compared to direct or broker channels.
*   **Demographic & Financial Friction:** SHAP values indicate that younger demographics (`num__age` trending low) combined with high monthly financial pressure (`num__monthly_premium`) are highly susceptible to competitor poaching.

### 2. Fraud Typologies (Financial Armor)
*   **The "Claim to Premium" Anomaly:** The most glaring red flag is the `num__claim_to_premium` ratio. When the requested claim drastically outweighs the lifetime value or premium paid, the fraud probability skyrockets.
*   **Severity Triggers:** Absolute financial values (`num__total_claim_amount` and `num__claim_severity`) dominate the tree-splits. High-severity claims inherently carry a much higher likelihood of malice.
*   **Hit-and-Run Fraud:** The `num__tenure_claim` feature highlights that new customers logging high-value claims almost immediately after policy inception represent acute risks.

---

## 💡 Business Implementations

Deploying this predictive suite directly influences operational KPIs:
1.  **Fast-Track Automation:** Inferences scoring below the risk threshold bypass manual underwriting. Immediate payouts for low-risk clients boost NPS and drastically lower operational expenditure (OPEX).
2.  **Special Investigation Unit (SIU) Routing:** High-severity claims flagged by the anomaly detection engine are routed instantly to senior investigators, armed with the exact SHAP reasoning (e.g., "Flagged due to high Claim-to-Premium ratio").
3.  **Proactive Onboarding:** Customers segmented as "High Churn Risk" (e.g., young, online-acquired) trigger automated retention workflows (e.g., personalized welcome calls, loyalty discounts 1 month before renewal).

---

## 🚀 Setup & Installation

**1. Clone the repository and navigate to the root directory:**
```bash
git clone https://github.com/SelimNajaf/Insurance-Churn-Fraud-ML.git
cd insur-risk-ai
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Directory Structure:**
Ensure your data files are placed correctly before running the pipelines:
```text
├── data/
│   ├── policies_crm.csv     # Raw CRM data
│   └── claims_data.csv      # Raw Claims data
├── src/
│   ├── eda_processor.py
│   ├── feature_engineering.py
│   ├── churn_model.py
│   └── fraud_model.py
├── models/                  # Exported .joblib pipelines
└── plots/                   # Auto-generated SHAP and EDA charts
```

---

## 💻 Usage Instructions

To train the models and generate insights, execute the pipeline sequentially:

```bash
# 1. Process Raw Data and Generate Initial KPIs
python src/eda_processor.py

# 2. Engineer Leakage-Free Features for Models
python src/feature_engineering.py

# 3. Train Churn Model & Extract Explanations
python src/churn_model.py

# 4. Train Fraud Model & Optimize Cost-Matrix
python src/fraud_model.py
```

*Upon successful execution, production-ready `.joblib` artifacts containing the data preprocessors, trained models, and optimized business thresholds will be saved to the `/models/` directory.*

---
*Built with precision for enterprise financial risk management.*
