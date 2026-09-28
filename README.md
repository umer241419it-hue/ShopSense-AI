# ShopSense AI

> **AI-Powered Online Shopper Purchase Intention Prediction and Analytics System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-active%20%26%20verified-success.svg)]()
[![Repository](https://img.shields.io/badge/github-ShopSense--AI-181717.svg?logo=github)](https://github.com/umer241419it-hue/ShopSense-AI.git)

---

## 📌 Project Overview
**ShopSense AI** is a complete, production-grade machine learning system designed to predict whether an e-commerce website visitor will make a purchase (`Revenue = True`) during their browsing session. Built on real-world session logs from the official UCI Machine Learning Repository, the project delivers an end-to-end data pipeline, exploratory analysis, leak-free preprocessing, cost-sensitive class balancing, hyperparameter tuning, model comparison, automated testing, and an interactive Streamlit analytics dashboard.

---

## 🎯 Problem Statement
In electronic commerce, only ~2% to 15% of sessions conclude with a financial transaction. Platforms need the ability to distinguish high-intent buyers from casual bouncers and window shoppers early in the session.
- **Severe Class Imbalance:** ~84.5% non-purchasing sessions vs ~15.5% purchasing sessions.
- **High-Dimensional Heterogeneous Data:** Continuous duration metrics, Google Analytics interaction rates (`BounceRates`, `ExitRates`), e-commerce value indicators (`PageValues`), and nominal metadata (operating systems, browsers, regions, traffic channels, seasonality).
- **Business Need:** Timely real-time intent prediction allows platforms to offer targeted conversion incentives (e.g., dynamic checkout assistance, personalized discounts) without unnecessary margin erosion.

---

## ✨ Features Currently Implemented

### 1. Interactive Frontend (Streamlit Dashboard)
- **User Interface (`app.py`):** Clean, responsive web dashboard with dedicated tabs for live prediction, model performance metrics, and exploratory data analysis.
- **Customer Persona Presets:** One-click testing with realistic pre-configured profiles:
  - *High-Intent Shopper* (elevated PageValues, deep product navigation)
  - *Quick Bouncer* (single-page landing, 20% bounce rate)
  - *Window Shopper* (high catalog views, zero page value)
  - *Holiday New Visitor* (seasonal visit, positive intent)
- **Visual Analytics:** Real-time conversion probability breakdown via Plotly, intent level categorization (*Low*, *Moderate*, *High*, *Very High*), session diagnostic summaries, and commercial action recommendations.

### 2. Backend & Inference Engine (`src/predict.py`)
- **Standalone Predictor:** Modular Python inference service (`ShopperPurchasePredictor`) capable of scoring single dictionary inputs or batch DataFrames.
- **Automated Feature Engineering:** Enforces identical transformations to raw inputs at inference time without retraining.
- **Confidence & Intent Scoring:** Calculates dual probabilities ($P(\text{Purchase})$ and $P(\text{No Purchase})$) and classifies customer intent tiers.

### 3. Machine Learning Core (`src/train.py`, `src/preprocessing.py`, `src/evaluate.py`)
- **Model Implementations:**
  - Logistic Regression (Baseline & Cost-Sensitive Balanced)
  - Decision Tree Classifier (Baseline & Cost-Sensitive Balanced)
  - Random Forest Classifier (Baseline & Cost-Sensitive Balanced)
  - Gradient Tree Boosting
  - Tuned Random Forest (4-fold cross-validated grid search)
- **Leak-Free Preprocessing:** Scikit-learn `Pipeline` with `ColumnTransformer` standardizing numerical features via `StandardScaler` and encoding nominal features via `OneHotEncoder(handle_unknown='ignore')`. Fit exclusively on training data.
- **Feature Engineering:** Five domain-informed features: `TotalPageViews`, `TotalDuration`, `ProductRelated_Ratio`, `BounceExit_Product`, and `PageValues_per_Duration`.
- **Imbalance Remediation:** Cost-sensitive class weighting (`class_weight='balanced'`) boosting recall on purchasing sessions from 42.67% to 73.30%.
- **Model Serialization:** Complete end-to-end pipeline persisted at `models/best_model.joblib`.

### 4. Database & Storage Architecture
- **Current Architecture:** File-based data architecture using CSV storage (`data/raw/` and `data/processed/`) and serialized joblib artifacts (`models/`). 
- **Database Status:** For this academic ML experiment, a relational database is intentionally omitted to keep the project lightweight, portable, and reproducible without external database dependencies.

### 5. Academic Reporting & Visualizations (`reports/`)
- 16 publication-quality analytical plots in `reports/figures/` (ROC curves, PR curves, confusion matrices, correlation heatmap, feature importance).
- Complete academic laboratory writeup (`reports/experiment_writeup.md`).
- Experimental evaluation report and CSV comparison (`reports/model_comparison.csv`, `reports/evaluation_report.txt`, `reports/final_results.md`).
- 28-question viva voce preparation guide (`reports/viva_questions.md`).

### 6. Executed Jupyter Notebook (`notebooks/online_shoppers_analysis.ipynb`)
- 24 comprehensive sections executed end-to-end with embedded outputs, visualizations, and commentary.

### 7. Automated Test Suite (`tests/test_pipeline.py`)
- Comprehensive test suite covering data integrity, preprocessing, model loading, prediction validity, and Streamlit import readiness.

---

## 🛠️ Technology Stack
| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Manipulation** | pandas, numpy, scipy |
| **Machine Learning** | scikit-learn, joblib |
| **Data Visualization** | matplotlib, seaborn, plotly |
| **Web Dashboard** | streamlit |
| **Data Acquisition** | ucimlrepo, requests |
| **Testing** | unittest / pytest |
| **Notebook Execution** | nbformat, nbclient, ipykernel |

---

## 📁 Project Structure
```text
ShopSense-AI/
│
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv    # Pristine raw UCI dataset
│   └── processed/
│       ├── train.csv                        # Processed stratified training set
│       └── test.csv                         # Processed stratified test set
│
├── notebooks/
│   └── online_shoppers_analysis.ipynb       # Executed 24-section research notebook
│
├── src/
│   ├── __init__.py                          # Package init
│   ├── data_loader.py                       # Automated dataset acquisition & validation
│   ├── preprocessing.py                     # Cleaning, feature engineering & transformers
│   ├── eda.py                               # Visual EDA suite generating all charts
│   ├── train.py                             # CV, tuning, training & model persistence
│   ├── evaluate.py                          # Metrics, curves, confusion matrix, reports
│   └── predict.py                           # Standalone inference engine
│
├── models/
│   └── best_model.joblib                    # Serialized end-to-end ML pipeline
│
├── reports/
│   ├── figures/                             # 16 analytical visualization plots
│   ├── model_comparison.csv                 # Test set benchmark metrics table
│   ├── evaluation_report.txt                # Full text evaluation summary
│   ├── final_results.md                     # Deep experimental findings
│   ├── experiment_writeup.md                # Structured academic writeup
│   └── viva_questions.md                    # 28 viva questions with detailed answers
│
├── tests/
│   └── test_pipeline.py                     # Automated testing suite
│
├── app.py                                   # Streamlit interactive application
├── build_notebook.py                        # Script to construct notebook cells
├── execute_notebook.py                      # Script to execute notebook via nbclient
├── run_project.py                           # Master end-to-end orchestrator
├── requirements.txt                         # Pinned dependency requirements
├── .env.example                             # Environment configuration template
├── .gitignore                               # Git ignore configuration
└── README.md                                # Project documentation
```

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
Ensure Python 3.10+ is installed on your workstation.

### 2. Clone the Repository
```bash
git clone https://github.com/umer241419it-hue/ShopSense-AI.git
cd ShopSense-AI
```

### 3. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration
Copy the template configuration file:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## 💻 How to Run Major Components

### A. Run the Complete End-to-End ML Pipeline
Downloads the dataset, verifies schemas, executes EDA figure generation, trains all algorithms, performs hyperparameter tuning, evaluates the holdout test set, and serializes the best model:
```bash
python run_project.py
```

### B. Launch the Interactive Streamlit Web Application
Starts the live graphical dashboard:
```bash
streamlit run app.py
```
*Access the interface in your browser at: `http://localhost:8501`*

### C. Run the Backend Inference Script
Test the prediction engine directly via CLI:
```bash
python src/predict.py
```

### D. Run Automated Test Suite
Verify dataset schemas, preprocessing pipelines, model loading, and app imports:
```bash
python tests/test_pipeline.py
```

### E. Open the Executed Jupyter Notebook
```bash
jupyter notebook notebooks/online_shoppers_analysis.ipynb
```

---

## 📊 Experimental Results Benchmark

Evaluated on an independent 20% stratified hold-out test set ($N = 2,441$):

| Rank | Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest (Tuned)** | **89.18%** | **63.35%** | **73.30%** | **0.6796** | **0.9265** | **0.7225** |
| 2 | Gradient Boosting | 90.33% | 71.60% | 63.35% | 0.6722 | 0.9357 | 0.7404 |
| 3 | Random Forest (Balanced) | 87.01% | 55.61% | 84.29% | 0.6701 | 0.9318 | 0.7282 |
| 4 | Random Forest (Default) | 90.74% | 76.53% | 58.90% | 0.6657 | 0.9315 | 0.7448 |
| 5 | Decision Tree | 89.64% | 71.86% | 55.50% | 0.6263 | 0.9104 | 0.6597 |
| 6 | Logistic Regression (Balanced) | 84.23% | 49.76% | 80.63% | 0.6154 | 0.9118 | 0.6729 |
| 7 | Decision Tree (Balanced) | 82.10% | 46.21% | 87.70% | 0.6052 | 0.9119 | 0.6778 |
| 8 | Logistic Regression (Default) | 89.10% | 77.62% | 42.67% | 0.5507 | 0.9018 | 0.6632 |

- **Final Model Selected:** Random Forest Classifier with `class_weight='balanced'`, `n_estimators=150`, `max_depth=None`, and `min_samples_split=2`.
- **Top Predictive Features:** `PageValues` (34.2% importance), `ExitRates` (8.4%), `ProductRelated_Duration` (7.6%), `TotalDuration` (7.1%), and `TotalPageViews` (6.2%).

---

## 🔮 Example Inference Usage

```python
from src.predict import ShopperPurchasePredictor

# Initialize predictor with the serialized pipeline
predictor = ShopperPurchasePredictor("models/best_model.joblib")

# Sample visitor session
session = {
    "Administrative": 3,
    "Administrative_Duration": 85.0,
    "Informational": 1,
    "Informational_Duration": 42.0,
    "ProductRelated": 28,
    "ProductRelated_Duration": 1150.0,
    "BounceRates": 0.005,
    "ExitRates": 0.015,
    "PageValues": 38.5,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}

result = predictor.predict(session)
print(f"Prediction: {result['prediction_label']}")
print(f"Purchase Probability: {result['purchase_probability']:.2%}")
print(f"Intent Level: {result['intent_level']}")
# Output:
# Prediction: Purchase
# Purchase Probability: 80.00%
# Intent Level: Very High
```

---

## ⚠️ Current Implementation Status & Limitations
- **Current Status:** Fully operational end-to-end ML pipeline, evaluated models, tested inference engine, and interactive dashboard.
- **Session-Level Aggregation:** Features represent cumulative session totals rather than millisecond clickstream event sequences.
- **Anonymity:** No cross-session user ID is present in the dataset; customer lifetime loyalty cannot be tracked across visits.
- **Database:** Uses local filesystem persistence (`data/` and `models/`). A centralized SQL/NoSQL database has not been integrated and is not required for this phase.

---

## 📚 Dataset Reference & Citation
> Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Comput & Applic* 31, 6893–6908 (2019). [https://doi.org/10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0)
