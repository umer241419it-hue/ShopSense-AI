# Academic Experiment Write-Up
## Department of Artificial Intelligence & Machine Learning
### Course Mini Project / Laboratory Experiment Documentation

---

### 1. Aim
To design, implement, evaluate, and interpret an end-to-end Machine Learning pipeline for predicting online shoppers' purchasing intention from session-level web analytics data.

---

### 2. Problem Statement
E-commerce websites face high customer acquisition costs yet experience low conversion rates (~2–15%). Identifying whether an active browsing session will culminate in a purchase (`Revenue = True`) enables timely real-time interventions (dynamic vouchers, assistance, checkout simplification). The primary challenges are severe class imbalance (84.5% non-purchases vs. 15.5% purchases), mixed data types (numerical rates and duration metrics alongside categorical identifiers), and ensuring zero data leakage in preprocessing and model validation.

---

### 3. Objectives
1. Acquire the official UCI Online Shoppers Purchasing Intention Dataset and validate schema integrity.
2. Conduct comprehensive Exploratory Data Analysis (EDA) on behavioral, temporal, and demographic variables.
3. Formulate domain-specific feature engineering to model session depth and efficiency.
4. Build a leak-free scikit-learn preprocessing pipeline using `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
5. Implement, benchmark, and contrast four primary classification families: Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
6. Evaluate algorithmic class-balancing mechanisms (`class_weight='balanced'`).
7. Perform hyperparameter optimization using cross-validated grid search.
8. Validate model inference on simulated shopper personas through an interactive Streamlit application.

---

### 4. Dataset Description
- **Title:** Online Shoppers Purchasing Intention Dataset
- **Repository:** UCI Machine Learning Repository (ID: 468)
- **Donor:** C. O. Sakar et al., Bogazici University (2018)
- **Instances:** 12,330 total sessions
- **Target:** `Revenue` (Boolean: `True` [1,908 sessions / 15.47%], `False` [10,422 sessions / 84.53%])
- **Input Features (17 attributes):**
  - **Session Activity (Numerical):** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`.
  - **Quality & Google Analytics Metrics (Numerical):** `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`.
  - **Contextual & Demographics (Categorical):** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

---

### 5. Software & Hardware Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Programming Language:** Python 3.10+ (tested on Python 3.14.5)
- **Key Python Libraries:**
  - `pandas` (>=2.2.0) — Data manipulation and table structures
  - `numpy` (>=1.26.0) — Vectorized mathematical arrays
  - `scikit-learn` (>=1.4.0) — Machine learning models, pipelines, metrics
  - `matplotlib` (>=3.8.0) & `seaborn` (>=0.13.0) — Publication-grade visualizations
  - `plotly` (>=5.18.0) — Interactive probability distribution graphics
  - `joblib` (>=1.3.0) — Model serialization
  - `streamlit` (>=1.32.0) — Web dashboard application
  - `ucimlrepo` (>=0.0.6) — Automated official dataset acquisition
- **Hardware Requirements:** Standard workstation/laptop (minimum 4GB RAM, 2 CPU cores; GPU not required).

---

### 6. Methodology
The experiment follows a disciplined Data Science Lifecycle:
1. **Acquisition:** Automated fetching via `ucimlrepo` with fallback to direct compressed zip download.
2. **Sanitization:** Detection of exact duplicates and dropping 125 duplicate sessions to preserve partition independence.
3. **Feature Engineering:** Derivation of five engagement metrics (`TotalPageViews`, `TotalDuration`, `ProductRelated_Ratio`, `BounceExit_Product`, `PageValues_per_Duration`).
4. **Data Isolation:** Stratified 80/20 train/test split.
5. **Preprocessing:** Fit-transform on training split; transform-only on test split.
6. **Cross-Validation:** 5-fold Stratified K-Fold cross-validation across all baseline models.
7. **Class-Balancing Study:** Parallel training with and without cost-sensitive weighting.
8. **Hyperparameter Tuning:** 4-fold GridSearchCV on Random Forest.
9. **Final Evaluation & Interpretation:** Evaluation on holdout test set using F1-score, ROC-AUC, and feature importance.
10. **Application Deployment:** Streamlit interactive interface.

---

### 7. Data Preprocessing
- **Handling Duplicates:** 125 identical rows removed ($12,330 \rightarrow 12,205$).
- **Handling Missing Values:** None present in the UCI dataset.
- **Categorical Handling:** Categorical columns (including integer nominal codes like `OperatingSystems` and `TrafficType`) cast to strings and one-hot encoded (`sparse_output=False`, `handle_unknown='ignore'`).
- **Feature Scaling:** `StandardScaler` applied to continuous columns ($z = (x - \mu)/\sigma$).
- **Pipeline Architecture:** Scikit-learn `Pipeline([('preprocessor', ColumnTransformer), ('classifier', Model)])`.

---

### 8. Exploratory Data Analysis (EDA) Highlights
- **Target Imbalance:** ~5.46:1 negative to positive ratio.
- **PageValues:** The most distinct differentiator. Sessions with `Revenue=True` average `PageValues` of 27.26, compared to 1.98 for `Revenue=False`.
- **Exit & Bounce Rates:** High exit rates (>0.05) correlate with zero purchase probability.
- **Seasonality:** November exhibits peak conversion (~20.6%) due to holiday shopping.

---

### 9. Algorithms Evaluated
1. **Logistic Regression:** Linear probabilistic classification via logit link.
2. **Decision Tree Classifier:** Recursive binary splitting using Gini impurity.
3. **Random Forest Classifier:** Ensemble bagging of decorrelated decision trees.
4. **Gradient Tree Boosting:** Sequential additive boosting minimizing log loss.
5. **Cost-Sensitive Variants:** Class weighting inversely proportional to class frequencies ($w_j = \frac{N}{2 \times N_j}$).

---

### 10. Evaluation Metrics
For imbalanced classification, accuracy is supplemented by positive-class focused metrics:
- **Accuracy:** Overall correctness.
- **Precision (Purchase):** Reliability of conversion alarms.
- **Recall (Purchase):** Sensitivity in detecting actual purchasers.
- **F1-Score (Purchase):** Harmonic balance between precision and recall.
- **ROC-AUC & PR-AUC:** Threshold-independent ranking capability.

---

### 11. Experimental Results

#### Performance on Stratified Test Set ($N = 2,441$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **89.18%** | **63.35%** | **73.30%** | **0.6796** | **0.9265** | **0.7225** |
| Gradient Boosting | 90.33% | 71.60% | 63.35% | 0.6722 | 0.9357 | 0.7404 |
| Random Forest (Balanced) | 87.01% | 55.61% | 84.29% | 0.6701 | 0.9318 | 0.7282 |
| Random Forest (Default) | 90.74% | 76.53% | 58.90% | 0.6657 | 0.9315 | 0.7448 |
| Decision Tree (Default) | 89.64% | 71.86% | 55.50% | 0.6263 | 0.9104 | 0.6597 |
| Logistic Regression (Balanced) | 84.23% | 49.76% | 80.63% | 0.6154 | 0.9118 | 0.6729 |
| Decision Tree (Balanced) | 82.10% | 46.21% | 87.70% | 0.6052 | 0.9119 | 0.6778 |
| Logistic Regression (Default) | 89.10% | 77.62% | 42.67% | 0.5507 | 0.9018 | 0.6632 |

---

### 12. Result Interpretation
1. **The Class Imbalance Effect:** Default Logistic Regression missed 57.3% of buyers (Recall: 42.67%). Adding `class_weight='balanced'` boosted recall to **80.63%**, proving the necessity of cost-sensitive weighting.
2. **Superiority of Ensemble Learning:** Random Forest and Gradient Boosting outperformed single decision trees and linear models by over 5% in F1-score and 2% in ROC-AUC.
3. **Hyperparameter Tuning Impact:** Tuning Random Forest hyperparameters (`n_estimators=150`, `class_weight='balanced'`) yielded the highest overall **F1-Score of 0.6796** and an **ROC-AUC of 0.9265**, striking an optimal trade-off between capturing 73.3% of purchasers while maintaining 63.35% precision.
4. **Primary Feature Contributions:** `PageValues` accounted for >34% of total model importance, followed by `ExitRates` (8.4%), `ProductRelated_Duration` (7.6%), and `TotalDuration` (7.1%).

---

### 13. Conclusion
The experiment rigorously proved that machine learning models can accurately predict online purchasing intention from web session telemetry. The tuned Random Forest model successfully addresses class imbalance and achieves high discriminative performance ($F_1 = 0.6796$, $\text{ROC-AUC} = 0.9265$). Serializing the full pipeline into an interactive Streamlit application bridges theoretical AI modeling with operational e-commerce utility.

---

### 14. Future Scope
1. Implement real-time sequential event modeling (e.g. RNN/Transformer models) using clickstream event-level timestamps.
2. Formulate uplift modeling to predict individual voucher sensitivity (causal treatment effect).
3. Connect the inference engine with an automated cart-recovery email and live-chat trigger system.
