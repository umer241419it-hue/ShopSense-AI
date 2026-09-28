"""
Model Evaluation and Performance Reporting Module.
Computes metrics, plots ROC/PR curves, confusion matrices, feature importance,
and generates structured academic comparison tables and reports.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

logger = logging.getLogger(__name__)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Calculate key classification evaluation metrics for positive class (Purchase=1)."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob))
    }


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "confusion_matrices_comparison.png"
) -> None:
    """Plot confusion matrices for evaluated models in a grid layout."""
    n_models = len(models)
    cols = 2
    rows = (n_models + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display counts and percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([[f"{cm[i, j]:,}\n({cm_pct[i, j]*100:.1f}%)" for j in range(2)] for i in range(2)])
        
        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No Purchase (0)", "Purchase (1)"],
            yticklabels=["No Purchase (0)", "Purchase (1)"],
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}", fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to: {save_path.name}")


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "roc_curves_comparison.png"
) -> None:
    """Plot overlaid ROC curves for all models."""
    plt.figure(figsize=(9, 7))
    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = palette[idx % len(palette)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2.2, color=color)
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves to: {save_path.name}")


def plot_precision_recall_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "precision_recall_curves_comparison.png"
) -> None:
    """Plot Precision-Recall curves (critical for imbalanced datasets)."""
    plt.figure(figsize=(9, 7))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    base_rate = y_test.mean()
    plt.axhline(base_rate, color="black", linestyle="--", label=f"Baseline Purchase Rate ({base_rate*100:.1f}%)")
    
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        color = palette[idx % len(palette)]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", linewidth=2.2, color=color)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (True Positive Rate)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Precision-Recall (PR) Curves Comparison", fontweight="bold", pad=15)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved PR curves to: {save_path.name}")


def plot_model_metrics(
    comparison_df: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "model_metrics_comparison.png"
) -> None:
    """Plot multi-metric bar chart comparing all models."""
    df_melt = comparison_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.ylim(0, 1.08)
    plt.title("Model Performance Metrics Comparison on Test Set", fontweight="bold", pad=15)
    plt.ylabel("Score")
    plt.xlabel("Evaluation Metric", fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved metric comparison bar chart to: {save_path.name}")


def plot_feature_importance(
    best_pipeline: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Path = FIGURES_DIR / "feature_importance.png"
) -> pd.DataFrame:
    """
    Extract and plot feature importance for the best model.
    Handles tree-based feature_importances_ and linear coef_.
    """
    classifier = best_pipeline.named_steps["classifier"]
    
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        metric_name = "Feature Importance (Gini / Information Gain)"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        metric_name = "Absolute Coefficient Magnitude"
    else:
        logger.warning("Classifier does not have feature_importances_ or coef_ attribute.")
        return pd.DataFrame()
        
    # Get transformed feature names from preprocessor
    preprocessor = best_pipeline.named_steps["preprocessor"]
    try:
        all_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        all_feature_names = [f"feat_{i}" for i in range(len(importances))]
        
    # Clean feature names (remove num__ and cat__)
    clean_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in all_feature_names
    ]
    
    fi_df = pd.DataFrame({
        "Feature": clean_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    top_df = fi_df.head(top_n).sort_values("Importance", ascending=True)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_df["Feature"], top_df["Importance"], color="#2b5c8f", edgecolor="black")
    plt.title(f"Top {top_n} Most Important Features for Purchase Prediction", fontweight="bold", pad=15)
    plt.xlabel(metric_name, fontweight="bold")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")
        
    plt.xlim(0, max(top_df["Importance"]) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to: {save_path.name}")
    
    return fi_df


def generate_evaluation_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    test_report_str: str,
    output_txt_path: Path
) -> None:
    """Generate comprehensive academic text report."""
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ONLINE SHOPPERS PURCHASING INTENTION PREDICTION\n")
        f.write("ACADEMIC MACHINE LEARNING EXPERIMENT EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. MODEL COMPARISON ON TEST SET (STRATIFIED 20% HOLD-OUT):\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. SELECTED BEST MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Performing Model: {best_model_name}\n")
        f.write("Selection Criteria: Maximizing F1-score & ROC-AUC for positive purchase class (Revenue=True).\n\n")
        
        f.write("3. DETAILED CLASSIFICATION REPORT FOR BEST MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(test_report_str)
        f.write("\n\n")
        
        f.write("4. BUSINESS & E-COMMERCE INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        f.write("- High Precision: Minimizes false alarms when directing high-value conversion incentives.\n")
        f.write("- High Recall: Captures genuine buying intent so cart abandonment interventions can trigger.\n")
        f.write("- PageValues is consistently the single strongest predictive feature of purchasing session.\n")
        f.write("=" * 80 + "\n")
        
    logger.info(f"Generated detailed evaluation report: {output_txt_path}")
