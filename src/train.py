"""
Model Training and Hyperparameter Tuning Module.
Trains multiple classification algorithms, evaluates class imbalance handling,
performs hyperparameter tuning, compares models, and persists the best pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

from src.data_loader import load_raw_data
from src.preprocessing import prepare_data
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_model_metrics,
    plot_feature_importance,
    generate_evaluation_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define candidate classifiers including standard and class-balanced variants."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=random_state
        ),
        "Decision Tree (Balanced)": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state
        )
    }
    return models


def train_and_cross_validate(
    models: Dict[str, Any],
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> Dict[str, Pipeline]:
    """Train all candidate models in full pipelines with 5-fold cross-validation."""
    logger.info(f"Training and cross-validating {len(models)} models with {cv_folds}-fold CV...")
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # 5-fold CV F1 score (positive class)
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        logger.info(
            f"[{name}] 5-Fold CV -> F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f}) | "
            f"ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})"
        )
        
    return trained_pipelines


def tune_best_model(
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Perform hyperparameter tuning for Random Forest and Gradient Boosting using GridSearchCV.
    Returns the best tuned pipeline and best parameters.
    """
    logger.info("Performing targeted hyperparameter tuning for Random Forest...")
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [8, 12, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__class_weight": [None, "balanced"]
    }
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Tuning complete. Best CV F1-Score: {grid_search.best_score_:.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    best_pipeline = grid_search.best_estimator_
    return best_pipeline, grid_search.best_params_


def run_training_pipeline() -> Dict[str, Any]:
    """Execute complete training, evaluation, comparison, and persistence pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING FULL MACHINE LEARNING EXPERIMENT PIPELINE")
    logger.info("=" * 80)
    
    # 1. Load data
    raw_df = load_raw_data()
    
    # 2. Preprocess & Stratified Split
    prep_data = prepare_data(
        df=raw_df,
        test_size=0.20,
        random_state=42,
        drop_duplicates=True,
        use_feature_engineering=True
    )
    
    X_train = prep_data["X_train"]
    X_test = prep_data["X_test"]
    y_train = prep_data["y_train"]
    y_test = prep_data["y_test"]
    preprocessor = prep_data["preprocessor"]
    
    # 3. Candidate models & baseline training
    candidate_models = get_candidate_models(random_state=42)
    trained_pipelines = train_and_cross_validate(
        models=candidate_models,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        cv_folds=5
    )
    
    # 4. Hyperparameter tuning on top candidate
    tuned_pipeline, best_params = tune_best_model(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        random_state=42
    )
    trained_pipelines["Random Forest (Tuned)"] = tuned_pipeline
    
    # 5. Evaluate all models on test set
    comparison_records = []
    
    for name, model in trained_pipelines.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        comparison_records.append(metrics)
        
    comparison_df = pd.DataFrame(comparison_records)
    # Order columns
    cols_order = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    comparison_df = comparison_df[cols_order].sort_values("F1-Score", ascending=False).reset_index(drop=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SET PERFORMANCE COMPARISON:")
    logger.info("=" * 80)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison table to: {csv_path}")
    
    # 6. Select Best Model (highest F1-score balancing precision and recall on conversion)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    logger.info(f"\nSELECTED BEST MODEL: {best_model_name}")
    
    # Detailed classification report for best model
    best_y_pred = best_pipeline.predict(X_test)
    class_report_str = classification_report(
        y_test, best_y_pred,
        target_names=["No Purchase (0)", "Purchase (1)"],
        digits=4
    )
    logger.info("\n" + class_report_str)
    
    # Save best model pipeline
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"Saved complete end-to-end best pipeline to: {best_model_path}")
    
    # 7. Generate evaluation plots
    logger.info("Generating evaluation visualization artifacts...")
    plot_confusion_matrices(trained_pipelines, X_test, y_test)
    plot_roc_curves(trained_pipelines, X_test, y_test)
    plot_precision_recall_curves(trained_pipelines, X_test, y_test)
    plot_model_metrics(comparison_df)
    
    # Feature importance
    feature_names = prep_data["num_features"] + prep_data["cat_features"]
    plot_feature_importance(best_pipeline, feature_names)
    
    # 8. Generate text evaluation report
    report_txt_path = REPORTS_DIR / "evaluation_report.txt"
    generate_evaluation_report(comparison_df, best_model_name, class_report_str, report_txt_path)
    
    return {
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "best_params": best_params,
        "class_report_str": class_report_str,
        "trained_pipelines": trained_pipelines
    }


if __name__ == "__main__":
    results = run_training_pipeline()
