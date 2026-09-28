# Final Experimental Results & Evaluation Report
## Online Shoppers Purchasing Intention Prediction Using Machine Learning

---

### 1. Project Title
**Online Shoppers Purchasing Intention Prediction Using Machine Learning**  
*An Academic Machine Learning & Decision Intelligence Mini Project*

---

### 2. Problem Statement
In electronic commerce, the conversion rate of browsing visitors into paying customers is notoriously low (typically 2% to 4% industry-wide, and approximately 15.5% in session logs with purchase outcomes). Identifying high-intent shoppers early during a browsing session enables platforms to dynamically offer targeted promotions, streamline checkout, or allocate real-time customer support. However, predicting purchase intent entails several key challenges:
- **Severe Class Imbalance:** Over 84% of sessions do not generate revenue, causing standard classifiers to disproportionately favor the majority class.
- **Multimodal Feature Dynamics:** Sessions encompass continuous duration metrics, engagement rates (bounce and exit rates), e-commerce utility metrics (`PageValues`), and nominal metadata (operating systems, browsers, regions, traffic channels, seasonality).
- **Prevention of Data Leakage:** Preprocessing transformations and scaling must strictly be isolated from test data.

---

### 3. Objective
The core objective of this project is to build, evaluate, and interpret a complete end-to-end Machine Learning pipeline that predicts whether an online session will result in a purchase (`Revenue = True`). Specific sub-goals include:
1. Automated acquisition and integrity validation of the official UCI Online Shoppers dataset.
2. Comprehensive exploratory data analysis to unearth behavioral conversion drivers.
3. Principled feature engineering to capture session depth and velocity.
4. Objective comparison of standard vs. cost-sensitive class balancing techniques.
5. Hyperparameter optimization across candidate models.
6. Deployment of an interactive Streamlit inference system.

---

### 4. Dataset
- **Official Source:** UCI Machine Learning Repository (Dataset ID: 468)  
  *URL:* [UCI Dataset Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Dataset Character:** Real-world session logs collected from an e-commerce website over a 1-year duration.
- **Missing Values:** 0 missing values across all records.
- **Duplicate Records:** 125 duplicate sessions detected and dropped to ensure strict separation between train and test distributions.

---

### 5. Dataset Size
- **Total Raw Instances:** 12,330 sessions
- **Cleaned Unique Instances:** 12,205 sessions
- **Train Set (80% Stratified Split):** 9,764 sessions
- **Test Set (20% Stratified Hold-out):** 2,441 sessions

---

### 6. Features
The dataset contains 17 input attributes (10 numerical, 7 categorical) plus 5 engineered features:

#### Raw Numerical Features:
1. `Administrative`: Number of administrative/account management pages visited.
2. `Administrative_Duration`: Time in seconds spent on administrative pages.
3. `Informational`: Number of informational (e.g. about us, policy, contact) pages visited.
4. `Informational_Duration`: Time in seconds spent on informational pages.
5. `ProductRelated`: Number of product/catalog pages visited.
6. `ProductRelated_Duration`: Total time spent on product pages (seconds).
7. `BounceRates`: Percentage of visitors who enter the site and exit without triggering other requests.
8. `ExitRates`: Percentage of pageviews that were the last in the session.
9. `PageValues`: Average value of pages visited by the user before completing an e-commerce transaction.
10. `SpecialDay`: Closeness of the browsing date to a specific holiday/festival (e.g. Mother's Day).

#### Raw Categorical Features:
11. `Month`: Month of the session (`Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`).
12. `OperatingSystems`: Operating system identifier code (1 to 8).
13. `Browser`: Web browser identifier code (1 to 13).
14. `Region`: Geographic region code (1 to 9).
15. `TrafficType`: Web traffic source channel code (1 to 20).
16. `VisitorType`: Visitor loyalty status (`Returning_Visitor`, `New_Visitor`, `Other`).
17. `Weekend`: Boolean indicator whether browsing occurred on a weekend.

#### Engineered Behavioral Features:
18. `TotalPageViews`: Total pages navigated across all categories (`Administrative + Informational + ProductRelated`).
19. `TotalDuration`: Cumulative session duration across all page types.
20. `ProductRelated_Ratio`: Proportion of total pages that were product pages (`ProductRelated / TotalPageViews`).
21. `BounceExit_Product`: Interaction term between bounce and exit rates (`BounceRates * ExitRates`).
22. `PageValues_per_Duration`: Value-density metric (`PageValues / (TotalDuration + 1.0)`).

---

### 7. Target Variable
- **Attribute Name:** `Revenue`
- **Type:** Binary indicator (`True` = Purchase made, `False` = No purchase)
- **Raw Class Distribution:**
  - `No Purchase (False)`: 10,422 sessions (84.53%)
  - `Purchase (True)`: 1,908 sessions (15.47%)
  - Imbalance Ratio: ~5.46 : 1

---

### 8. Data Preprocessing
- **Duplicate Removal:** Removed 125 duplicate records to avoid data leakage between splits.
- **Stratified Splitting:** 80/20 train/test split utilizing `stratify=y` to preserve the exact positive-to-negative class ratio (84.35% / 15.65%) across both subsets.
- **Leak-Free ColumnTransformer:**
  - Numerical Pipeline: Standardized via `StandardScaler()` (mean=0, variance=1) fit strictly on `X_train`.
  - Categorical Pipeline: Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fit strictly on `X_train`.

---

### 9. Exploratory Data Analysis
Key empirical observations from EDA:
1. **Dominance of PageValues:** Non-purchasing sessions have a median `PageValues` of 0.00. In contrast, sessions resulting in purchases have substantially elevated `PageValues` (mean: ~27.26).
2. **Engagement vs. Bouncing:** Sessions with high `BounceRates` (>0.05) or `ExitRates` (>0.06) almost never convert.
3. **Visitor Type Loyalty:** `New_Visitor` cohorts exhibit higher conversion rates (~24.9%) compared to `Returning_Visitor` cohorts (~14.9%), indicating that new visitors often arrive with specific buying intent (e.g. from targeted search ads).
4. **Seasonal Spikes:** November displays massive session volume and conversion (~20.6%), driven by Black Friday and holiday shopping. May has high volume but lower conversion (~10.8%).

---

### 10. Algorithms Used
1. **Logistic Regression (Baseline vs. Balanced):** Standard linear classifier vs. cost-sensitive inverse class frequency weighting.
2. **Decision Tree (Baseline vs. Balanced):** Non-linear rule-based tree model.
3. **Random Forest (Baseline vs. Balanced):** Bagging ensemble of decision trees with feature subsampling.
4. **Gradient Boosting:** Sequential boosting optimizing binomial deviance.
5. **Random Forest (Tuned):** Hyperparameter-optimized ensemble using 4-fold cross-validated grid search.

---

### 11. Evaluation Metrics
Because accuracy alone is misleading for imbalanced datasets (a naive majority-class model achieves 84.5% accuracy with 0% purchase recall), the primary evaluation metric is **F1-Score (Positive Class)** and **ROC-AUC**:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** $\frac{TP}{TP + FP}$ — Proportion of predicted purchases that were genuine.
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Proportion of actual purchases successfully captured.
- **F1-Score:** Harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC:** Area under Receiver Operating Characteristic curve across all discrimination thresholds.
- **PR-AUC:** Average Precision across recall levels.

---

### 12. Model Comparison (Actual Experimental Results)

All metrics were computed on the independent hold-out test set ($N = 2,441$ samples):

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

#### Impact of Class Imbalance Handling:
- Default Logistic Regression achieved high accuracy (89.10%) but missed nearly 60% of buyers (Recall = 42.67%, F1 = 0.5507).
- Applying `class_weight='balanced'` boosted Logistic Regression recall from **42.67% to 80.63%**, elevating F1 to **0.6154**.
- For Random Forest, tuning with balanced class weights produced an optimal harmonic tradeoff: **Recall = 73.30%**, **Precision = 63.35%**, **F1 = 0.6796**, and **ROC-AUC = 0.9265**.

---

### 13. Final Selected Model
- **Selected Algorithm:** **Random Forest Classifier (Tuned with Class Balancing)**
- **Optimal Hyperparameters:**
  - `n_estimators`: 150
  - `max_depth`: None (unconstrained tree depth with leaf purity criteria)
  - `min_samples_split`: 2
  - `class_weight`: `'balanced'`
- **Confusion Matrix on Test Set ($N = 2,441$):**
  - True Negatives (Correct Non-Purchases): **1,897** (92.1% of actual non-purchases)
  - False Positives (False Alarms): **162**
  - False Negatives (Missed Buyers): **102**
  - True Positives (Captured Purchases): **280** (73.3% of actual purchases)

---

### 14. Important Features (Model Interpretation)
Feature importance extracted from the trained Random Forest classifier indicates:
1. **`PageValues` (~34.2%):** The overwhelmingly strongest predictor of purchasing intent. Pages with high historical revenue association indicate high buyer intent.
2. **`ExitRates` (~8.4%):** Strong negative association; users who frequently reach terminal exit states rarely complete checkout.
3. **`ProductRelated_Duration` (~7.6%):** Time spent inspecting product details correlates positively with purchase likelihood up to an exhaustion threshold.
4. **`TotalDuration` (~7.1%) & `TotalPageViews` (~6.2%):** Overall session depth and sustained engagement.
5. **`BounceExit_Product` (~5.5%):** Captures immediate bounce leakage.
6. **`Month_Nov` (~4.8%):** Strong positive seasonal coefficient due to Black Friday/Cyber Week retail spikes.

---

### 15. Interpretation & E-Commerce Value
- **Early Intent Detection:** Because `PageValues` and cumulative page browsing distinguish potential buyers within the first few interactions, platforms can adaptively personalize the session.
- **Targeted Promotions:** Rather than issuing blanket discounts to all visitors (eroding margins), businesses can target visitors with **Moderate Intent** (probability 30%–60%) who need an incentive to convert, while letting **High Intent** visitors (>70%) purchase at full price.

---

### 16. Limitations
1. **Aggregated Summaries vs. Event Streams:** The dataset captures pre-aggregated session metrics rather than millisecond clickstream sequences.
2. **Anonymity:** No cross-session customer ID exists, precluding long-term customer lifetime value (LTV) modeling.
3. **Price/Catalog Agnostic:** Transaction basket size and product price points are not included in the raw dataset.

---

### 17. Future Scope
1. **Clickstream Sequential Modeling:** Deploying LSTM or Transformer architectures on raw click sequence event logs.
2. **Dynamic Uplift Modeling:** Direct estimation of causal discount elasticity.
3. **Continuous Re-ranking:** Re-ordering search catalog results in real time based on instant intent scoring.

---

### 18. Conclusion
The project has successfully delivered an end-to-end Machine Learning solution for online shopper purchasing intention prediction. With an F1-score of 0.6796, ROC-AUC of 0.9265, recall of 73.30%, and a production-ready interactive Streamlit interface, the solution fulfills all academic and practical requirements.
