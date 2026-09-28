"""
Exploratory Data Analysis (EDA) and Visualization Module.
Generates comprehensive academic figures for the Online Shoppers Purchasing Intention project.
All figures are saved to reports/figures/.
"""

import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot target distribution (Revenue: True vs False) with counts and percentages."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Revenue"].value_counts()
    percentages = (df["Revenue"].value_counts(normalize=True) * 100).round(2)
    
    labels = ["No Purchase (False)", "Purchase (True)"]
    colors = ["#4575b4", "#d73027"]
    bars = ax.bar(labels, [counts[False], counts[True]], color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    
    for bar, pct in zip(bars, [percentages[False], percentages[True]]):
        height = bar.get_height()
        ax.annotate(f"{height:,}\n({pct}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=12)
        
    ax.set_title("Target Distribution: Online Shopper Purchase Intention", pad=15, fontweight="bold")
    ax.set_ylabel("Number of Sessions")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    save_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path.name}")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for all continuous numerical features."""
    numerical_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    corr = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: Numerical Features", pad=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path.name}")


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Plot distribution histograms and boxplots for key numerical metrics."""
    features = ["PageValues", "ExitRates", "BounceRates", "ProductRelated_Duration"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        ax = axes[i]
        sns.histplot(
            data=df,
            x=col,
            hue="Revenue",
            kde=True,
            bins=30,
            palette={False: "#4575b4", True: "#d73027"},
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"Distribution of {col} by Purchase Status", fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylabel("Log Frequency")
        
    plt.tight_layout()
    save_path = FIGURES_DIR / "numerical_distributions.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path.name}")


def plot_purchase_rate_by_category(
    df: pd.DataFrame,
    col: str,
    title: str,
    filename: str,
    order: list = None
) -> None:
    """Helper to calculate and plot purchase conversion rate across categories."""
    summary = df.groupby(col)["Revenue"].agg(["count", "mean"]).reset_index()
    summary["conversion_rate"] = summary["mean"] * 100
    
    if order:
        summary[col] = pd.Categorical(summary[col], categories=order, ordered=True)
        summary = summary.sort_values(col)
    else:
        summary = summary.sort_values("conversion_rate", ascending=False)
        
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    x = range(len(summary))
    # Bar plot for volume
    bars = ax1.bar(x, summary["count"], color="#bdd7e7", width=0.5, label="Total Sessions", edgecolor="grey")
    # Line plot for conversion rate
    lines = ax2.plot(x, summary["conversion_rate"], color="#d73027", marker="o", linewidth=2.5, label="Purchase Rate (%)")
    
    ax1.set_xlabel(col, fontweight="bold")
    ax1.set_ylabel("Total Sessions (Volume)", color="#2b5c8f")
    ax2.set_ylabel("Purchase Conversion Rate (%)", color="#d73027")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary[col].astype(str), rotation=30, ha="right")
    ax2.grid(False)
    
    # Annotate conversion percentages
    for i, val in enumerate(summary["conversion_rate"]):
        ax2.annotate(f"{val:.1f}%", (i, val), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontweight="bold", fontsize=9, color="#990000")
        
    ax1.set_title(title, pad=15, fontweight="bold")
    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path.name}")


def plot_behavioral_relationships(df: pd.DataFrame) -> None:
    """Plot relationship between key behavioral variables and Revenue."""
    plot_df = df.copy()
    plot_df["Purchase_Status"] = plot_df["Revenue"].map({False: "No Purchase", True: "Purchase"}).astype(str)
    palette_map = {"No Purchase": "#4575b4", "Purchase": "#d73027"}
    
    # 1. PageValues vs Revenue
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x="Purchase_Status", y="PageValues", hue="Purchase_Status",
        data=plot_df, palette=palette_map, legend=False, ax=ax
    )
    ax.set_title("PageValues Distribution by Purchase Status", fontweight="bold")
    ax.set_xlabel("Purchase Status")
    ax.set_ylabel("PageValues (Average Page Value Metric)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pagevalues_vs_revenue.png", bbox_inches="tight")
    plt.close()
    
    # 2. ExitRates vs Revenue
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x="Purchase_Status", y="ExitRates", hue="Purchase_Status",
        data=plot_df, palette=palette_map, legend=False, ax=ax
    )
    ax.set_title("ExitRates Distribution by Purchase Status", fontweight="bold")
    ax.set_xlabel("Purchase Status")
    ax.set_ylabel("ExitRates")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exitrates_vs_revenue.png", bbox_inches="tight")
    plt.close()
    
    # 3. BounceRates vs Revenue
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x="Purchase_Status", y="BounceRates", hue="Purchase_Status",
        data=plot_df, palette=palette_map, legend=False, ax=ax
    )
    ax.set_title("BounceRates Distribution by Purchase Status", fontweight="bold")
    ax.set_xlabel("Purchase Status")
    ax.set_ylabel("BounceRates")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bouncerates_vs_revenue.png", bbox_inches="tight")
    plt.close()
    
    # 4. ProductRelated activity analysis (Pages vs Duration scatter)
    fig, ax = plt.subplots(figsize=(8, 6))
    sub_df = plot_df[plot_df["ProductRelated"] < 250]
    sns.scatterplot(
        data=sub_df,
        x="ProductRelated",
        y="ProductRelated_Duration",
        hue="Purchase_Status",
        alpha=0.6,
        palette=palette_map,
        ax=ax
    )
    ax.set_title("Product-Related Pages Visited vs Time Spent (Duration)", fontweight="bold")
    ax.set_xlabel("Number of Product-Related Pages Visited")
    ax.set_ylabel("Total Duration on Product Pages (Seconds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "productrelated_activity.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved behavioral relationship figures.")


def run_full_eda() -> None:
    """Execute complete EDA pipeline and save all figures."""
    logger.info("Running full Exploratory Data Analysis suite...")
    df = load_raw_data()
    
    plot_target_distribution(df)
    plot_correlation_matrix(df)
    plot_numerical_distributions(df)
    
    # VisitorType
    plot_purchase_rate_by_category(
        df, "VisitorType",
        "Purchase Conversion Rate by Visitor Type",
        "purchase_rate_by_visitor_type.png"
    )
    
    # Month order
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plot_purchase_rate_by_category(
        df, "Month",
        "Purchase Conversion Rate and Session Volume by Month",
        "purchase_rate_by_month.png",
        order=month_order
    )
    
    # Weekend
    plot_purchase_rate_by_category(
        df, "Weekend",
        "Purchase Conversion Rate: Weekday vs Weekend",
        "purchase_rate_by_weekend.png"
    )
    
    # TrafficType (top 15)
    top_traffic = df[df["TrafficType"].isin(df["TrafficType"].value_counts().head(12).index)]
    plot_purchase_rate_by_category(
        top_traffic, "TrafficType",
        "Purchase Conversion Rate by Traffic Type (Top 12 Channels)",
        "purchase_rate_by_traffic_type.png"
    )
    
    plot_behavioral_relationships(df)
    logger.info("All EDA plots generated and saved successfully!")


if __name__ == "__main__":
    run_full_eda()
