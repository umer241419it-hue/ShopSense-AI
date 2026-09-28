"""
Data Preprocessing and Feature Engineering Pipeline for Online Shoppers Purchasing Intention.
Handles cleaning, feature creation, encoding, scaling, and stratified splitting without data leakage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)

# Feature definitions
RAW_NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

RAW_CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "TotalPageViews",
    "TotalDuration",
    "ProductRelated_Ratio",
    "BounceExit_Product",
    "PageValues_per_Duration"
]

TARGET_COLUMN = "Revenue"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate meaningful domain-specific e-commerce behavioral features.
    
    Engineered Features:
    1. TotalPageViews: Sum of pages visited across Admin, Info, and Product categories.
    2. TotalDuration: Total time spent in the session across all page types.
    3. ProductRelated_Ratio: Proportion of page views that were product-focused.
    4. BounceExit_Product: Interaction between bounce rate and exit rate (indicates low engagement/leakage).
    5. PageValues_per_Duration: Page value efficiency normalized by session duration.
    """
    data = df.copy()
    
    # 1. Total page views
    data["TotalPageViews"] = (
        data["Administrative"] + data["Informational"] + data["ProductRelated"]
    )
    
    # 2. Total duration in seconds
    data["TotalDuration"] = (
        data["Administrative_Duration"] + 
        data["Informational_Duration"] + 
        data["ProductRelated_Duration"]
    )
    # Ensure non-negative duration
    data["TotalDuration"] = data["TotalDuration"].clip(lower=0.0)
    
    # 3. Product related page ratio
    data["ProductRelated_Ratio"] = data["ProductRelated"] / (data["TotalPageViews"] + 1e-5)
    
    # 4. Interaction between bounce rates and exit rates
    data["BounceExit_Product"] = data["BounceRates"] * data["ExitRates"]
    
    # 5. Page values per unit duration
    data["PageValues_per_Duration"] = data["PageValues"] / (data["TotalDuration"] + 1.0)
    
    return data


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean dataset: check nulls, remove duplicates if requested."""
    cleaned = df.copy()
    
    initial_rows = len(cleaned)
    dup_count = cleaned.duplicated().sum()
    if drop_duplicates and dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {dup_count} duplicate rows ({initial_rows} -> {len(cleaned)} rows)")
    else:
        logger.info(f"Preserved all rows. Duplicates present: {dup_count}")
        
    return cleaned


def get_feature_lists(use_feature_engineering: bool = True) -> Tuple[List[str], List[str]]:
    """Return lists of numerical and categorical feature names."""
    num_features = list(RAW_NUMERICAL_FEATURES)
    if use_feature_engineering:
        num_features.extend(ENGINEERED_NUMERICAL_FEATURES)
    cat_features = list(RAW_CATEGORICAL_FEATURES)
    return num_features, cat_features


def create_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer.
    - Numerical features: Standardized using StandardScaler (mean=0, variance=1)
    - Categorical features: One-hot encoded (ignoring unknown categories at test/inference time)
    """
    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    drop_duplicates: bool = True,
    use_feature_engineering: bool = True,
    save_processed: bool = True,
    processed_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Full data preparation pipeline:
    1. Cleaning (duplicate handling)
    2. Optional Feature Engineering
    3. Target encoding (True -> 1, False -> 0)
    4. Stratified Train-Test Split (strict prevention of data leakage)
    5. Saving processed train and test sets
    """
    logger.info("Executing prepare_data pipeline...")
    cleaned_df = clean_data(df, drop_duplicates=drop_duplicates)
    
    if use_feature_engineering:
        engineered_df = engineer_features(cleaned_df)
    else:
        engineered_df = cleaned_df
        
    # Cast categorical columns that might be integers to string so OneHotEncoder handles them cleanly
    for col in RAW_CATEGORICAL_FEATURES:
        engineered_df[col] = engineered_df[col].astype(str)
        
    # Split features and target
    X = engineered_df.drop(columns=[TARGET_COLUMN])
    y = engineered_df[TARGET_COLUMN].astype(int)
    
    num_features, cat_features = get_feature_lists(use_feature_engineering)
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    logger.info(f"Train class balance: {dict(y_train.value_counts(normalize=True).round(4))}")
    logger.info(f"Test class balance: {dict(y_test.value_counts(normalize=True).round(4))}")
    
    preprocessor = create_preprocessor(num_features, cat_features)
    
    if save_processed:
        if processed_dir is None:
            processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved processed train data to: {train_path}")
        logger.info(f"Saved processed test data to: {test_path}")
        
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "num_features": num_features,
        "cat_features": cat_features,
        "engineered_df": engineered_df
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw_df = load_raw_data()
    prepared = prepare_data(raw_df)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {prepared['X_train'].shape}")
    print(f"X_test shape: {prepared['X_test'].shape}")
