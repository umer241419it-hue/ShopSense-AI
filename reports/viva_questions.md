# Viva Voce Preparation Guide: 28 Essential Questions & Answers
## Online Shoppers Purchasing Intention Prediction Using Machine Learning

---

### Q1: Why was this specific problem chosen for an AI/ML project?
**Answer:** In e-commerce, the vast majority of visitors (~85-98%) browse without buying. Being able to predict purchase intent in real time allows e-commerce platforms to intervene intelligently—such as offering dynamic cart-abandonment discounts, initiating live chat, or streamlining checkout—without wasting margin on visitors who would either buy anyway or never buy. It represents a classic, high-value binary classification challenge featuring class imbalance and mixed data types.

---

### Q2: Why use the UCI Online Shoppers Purchasing Intention Dataset?
**Answer:** The UCI dataset (ID: 468, donated by Sakar et al.) is an industry-standard, peer-reviewed benchmark collected from real-world e-commerce logs over a one-year period. With 12,330 sessions, 17 multimodal features, zero missing values, and natural class imbalance (15.5% conversion), it provides an ideal testbed for rigorous machine learning without synthetic bias.

---

### Q3: What is the target variable, and what does it represent?
**Answer:** The target variable is `Revenue`, a boolean attribute:
- `True (1)`: The browsing session concluded with an e-commerce purchase transaction.
- `False (0)`: The browsing session concluded without a transaction.

---

### Q4: What is supervised binary classification?
**Answer:** Supervised binary classification is a machine learning paradigm where the algorithm is provided with labeled input-output pairs $(X, y)$, where each label $y \in \{0, 1\}$. The objective is to learn a mapping function $f: X \rightarrow \{0, 1\}$ or probability estimator $P(y=1|X)$ that accurately generalizes to unseen test instances.

---

### Q5: Why is train/test splitting essential in machine learning?
**Answer:** Train/test splitting evaluates how well a model generalizes to completely unseen data. Evaluating a model on the data it was trained on results in optimistic bias (memorization or overfitting). A hold-out test set provides an unbiased estimate of real-world generalization performance.

---

### Q6: Why did you use stratified splitting instead of simple random splitting?
**Answer:** In imbalanced datasets (e.g. 15.5% positive instances), simple random splitting can produce arbitrary fluctuations in class proportions between train and test sets. Stratified splitting enforces the exact same class distribution (84.35% negative, 15.65% positive) across both training and test partitions, ensuring statistical consistency and preventing sampling bias.

---

### Q7: What is data leakage, and how did your pipeline prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (specifically from the test set) is inadvertently used to train the model, resulting in overly optimistic evaluation. In our project:
1. Exact duplicates (125 rows) were removed prior to splitting.
2. Preprocessing scalers (`StandardScaler`) and categorical encoders (`OneHotEncoder`) were **fit strictly on `X_train`** and merely applied via `transform` to `X_test`.
3. Hyperparameter tuning was performed exclusively on the training set using cross-validation.

---

### Q8: Why was categorical encoding necessary, and which method did you use?
**Answer:** Machine learning algorithms (such as Logistic Regression and SVM) require mathematical numerical vectors. Features like `Month` and `VisitorType` are strings, while features like `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are recorded as arbitrary integers. If treated as numbers, the model would assume false ordinal ranking (e.g. Region 8 > Region 2). We applied `OneHotEncoder(handle_unknown='ignore')` to convert them into binary dummy vectors without assuming false ordinal hierarchy.

---

### Q9: Why was numerical feature scaling performed?
**Answer:** Features in this dataset possess vastly disparate scales; for example, `PageValues` ranges from 0 to 360, while `BounceRates` ranges from 0 to 0.20, and `ProductRelated_Duration` spans several thousand seconds. Algorithms relying on gradient descent (Logistic Regression) or distance metrics would be dominated by large-scale features. `StandardScaler` standardizes features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$). Tree-based models are invariant to monotonic scaling, but standardization ensures consistent pipeline compatibility across all candidate models.

---

### Q10: What is class imbalance, and what is the class ratio in this dataset?
**Answer:** Class imbalance occurs when the categories of the target variable are not represented equally. In this dataset, `Revenue = False` comprises 84.53% (10,422 sessions) and `Revenue = True` comprises 15.47% (1,908 sessions), representing an imbalance ratio of ~5.46 to 1.

---

### Q11: Why is raw accuracy an insufficient evaluation metric for this project?
**Answer:** A naive "zero-rule" dummy classifier that blindly predicts `Revenue = False` for every single visitor would achieve an **84.5% accuracy** while completely failing to identify a single purchasing customer (0% recall, 0% F1-score). In business terms, relying on accuracy would render the model useless for driving sales interventions.

---

### Q12: Explain the difference between Precision and Recall in this context.
**Answer:**
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all sessions the model flagged as "purchasing", what fraction actually bought? High precision avoids wasting expensive discount codes or sales rep time on false alarms.
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Out of all actual buyers, what fraction did the model successfully identify? High recall ensures the company captures potential sales opportunities and minimizes missed conversions.

---

### Q13: What is the F1-Score, and why is it preferred here?
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Because it uses the harmonic mean, it penalizes extreme imbalances between precision and recall. A model with high precision but dismal recall will have a low F1-score. We used F1-score as our primary optimization criterion.

---

### Q14: What does the ROC-AUC score measure?
**Answer:** The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Recall) vs. False Positive Rate across all classification probability thresholds (0.0 to 1.0). The Area Under the Curve (ROC-AUC) quantifies the model's ability to rank a randomly chosen positive session higher than a randomly chosen negative session. An AUC of 0.5 represents random guessing; our final tuned model achieved an ROC-AUC of **0.9265**.

---

### Q15: What is a Confusion Matrix, and what are its four quadrants?
**Answer:** A table summarizing prediction outcomes against ground truth:
1. **True Negative (TN):** Predicted No Purchase, Actually No Purchase (1,897 in our test set).
2. **False Positive (FP):** Predicted Purchase, Actually No Purchase (162; Type I Error).
3. **False Negative (FN):** Predicted No Purchase, Actually Purchase (102; Type II Error).
4. **True Positive (TP):** Predicted Purchase, Actually Purchase (280; Correct Detections).

---

### Q16: How does Logistic Regression work for classification?
**Answer:** Logistic Regression models the log-odds (logit) of the positive class as a linear combination of input features:
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{i=1}^n \beta_i x_i$$
The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps the output to a calibrated probability between 0 and 1. It assumes a linear decision boundary in log-odds space.

---

### Q17: How does a Decision Tree make split decisions?
**Answer:** A Decision Tree recursively partitions feature space by selecting the feature and threshold that maximizes class purity at each node. Common splitting criteria are **Gini Impurity** ($I_G = 1 - \sum p_i^2$) or **Information Gain (Entropy)** ($H = -\sum p_i \log_2 p_i$). While intuitive, standalone trees are prone to high variance and overfitting.

---

### Q18: What is a Random Forest, and why does it outperform a single Decision Tree?
**Answer:** Random Forest is an ensemble learning method based on **Bagging (Bootstrap Aggregation)** and **Random Subspace Projection**. It fits $B$ individual decision trees on bootstrap samples of the training data, and at each split, considers only a random subset of $m \approx \sqrt{p}$ features. By averaging predictions across hundreds of de-correlated trees, Random Forest drastically reduces model variance while maintaining low bias.

---

### Q19: How does Gradient Boosting differ fundamentally from Random Forest?
**Answer:**
- **Random Forest (Bagging):** Trees are grown in parallel independently on bootstrap samples; trees are deep and fully grown (low bias, high variance); averaging reduces variance.
- **Gradient Boosting (Boosting):** Trees are grown sequentially in series; each new shallow tree (stump/learner) is trained to predict the pseudo-residuals (negative gradient of the loss function) of the preceding ensemble; boosting systematically reduces bias.

---

### Q20: What is Overfitting, and what measures were taken to prevent it?
**Answer:** Overfitting occurs when a model learns noise, outliers, and idiosyncratic details of the training set rather than the underlying pattern, performing poorly on unseen test data. We prevented overfitting by:
1. Stratified 5-fold cross-validation.
2. Regularizing tree depth and setting `min_samples_split`.
3. Pruning noisy duplicate records.
4. Using an ensemble model (Random Forest) with feature subsampling.

---

### Q21: What is K-Fold Cross-Validation, and why use Stratified K-Fold?
**Answer:** In K-Fold cross-validation, the training set is split into $K$ equal subsets (folds). The model is trained on $K-1$ folds and validated on the remaining fold, rotating $K$ times. The overall metric is the average across all folds. **Stratified K-Fold** ensures every single fold contains the exact same class proportion as the overall dataset, which is critical for imbalanced classes.

---

### Q22: How was class imbalance handled in this project, and what were the findings?
**Answer:** We evaluated algorithmic cost-sensitive learning via `class_weight='balanced'`. This adjusts the loss function inversely proportional to class frequencies:
$$w_j = \frac{N}{2 \times N_j}$$
- **Result:** In Logistic Regression, balancing increased recall from **42.67% to 80.63%**, lifting F1 from 0.5507 to 0.6154. In Random Forest, balancing achieved a superior F1-score of **0.6796** with 73.30% recall and 0.9265 ROC-AUC.

---

### Q23: What hyperparameters were tuned, and what was the method?
**Answer:** We used `GridSearchCV` on the training set with 4-fold cross-validation optimizing for the `f1` metric. The search space included:
- `n_estimators`: [100, 150]
- `max_depth`: [10, 15, None]
- `min_samples_split`: [2, 5]
- `class_weight`: [None, 'balanced']  
**Optimal Configuration:** `{'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}`.

---

### Q24: Which feature was found to be the most important, and why?
**Answer:** **`PageValues`** was by far the single most influential predictor (~34.2% importance). `PageValues` is a Google Analytics metric reflecting the average transactional value of web pages browsed prior to completing a purchase. Visitors viewing pages with high PageValues are navigating high-intent conversion funnel pages (e.g. cart, shipping calculator, reviews).

---

### Q25: What other features significantly influence the prediction?
**Answer:**
- `ExitRates` & `BounceRates`: Negative correlation; high values indicate disinterest or navigation friction.
- `ProductRelated_Duration` & `TotalPageViews`: Positive correlation; sustained product examination signals active shopping intent.
- `Month_Nov`: Positive correlation; captures seasonal holiday shopping surges (Black Friday).
- `VisitorType_New_Visitor`: Shows higher conversion rate per session than returning browsers.

---

### Q26: Why was Random Forest selected as the final deployed model over Gradient Boosting?
**Answer:** Although Gradient Boosting achieved slightly higher overall accuracy (90.33% vs 89.18%), the Tuned Random Forest achieved a substantially higher **Positive Class Recall (73.30% vs 63.35%)** and the highest overall **F1-Score (0.6796 vs 0.6722)**. In e-commerce, capturing 10% more genuine buyers is worth a modest tradeoff in precision.

---

### Q27: How is the trained model deployed for interactive prediction?
**Answer:** The complete end-to-end pipeline (incorporating feature scaling, one-hot encoding, and the trained ensemble) is serialized using `joblib`. A modern Streamlit web application (`app.py`) loads this pipeline artifact, accepts interactive user inputs or pre-configured customer personas, executes instantaneous inference, and displays prediction labels, probability gauges, and tailored commercial recommendations.

---

### Q28: What are the primary academic limitations and real-world improvements?
**Answer:**
- **Limitation:** The dataset provides session aggregates rather than sequential event logs (e.g. click timestamp streams).
- **Future Improvement:** Using Recurrent Neural Networks (LSTMs) or Transformers on raw sequential clickstream data, dynamic promotion uplift modeling, and combining purchase probability with predicted basket order value.
