"""
Data Acquisition and Validation Module for Online Shoppers Purchasing Intention.
Dataset Source: UCI Machine Learning Repository (Dataset ID: 468)
URL: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Expected dataset parameters
EXPECTED_ROWS = 12330
EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue"
]

DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_shoppers_intention.csv"


def download_from_ucimlrepo() -> pd.DataFrame:
    """Fetch the Online Shoppers Purchasing Intention dataset from ucimlrepo."""
    logger.info("Attempting dataset download via ucimlrepo (ID: 468)...")
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=468)
    
    features = dataset.data.features
    targets = dataset.data.targets
    
    # Concatenate features and targets
    df = pd.concat([features, targets], axis=1)
    logger.info(f"Successfully fetched {len(df)} rows and {df.shape[1]} columns via ucimlrepo.")
    return df


def download_fallback_url() -> pd.DataFrame:
    """Fallback download directly from UCI repository archive if ucimlrepo fails."""
    logger.info("Attempting fallback direct download from UCI repository...")
    url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    import io
    import zipfile
    import requests
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find csv inside zip
        csv_filename = [name for name in z.namelist() if name.endswith(".csv")][0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)
    logger.info(f"Successfully downloaded {len(df)} rows from fallback zip URL.")
    return df


def acquire_data(output_path: Path = DEFAULT_RAW_PATH, force_download: bool = False) -> pd.DataFrame:
    """
    Acquire raw dataset: check local cache first, or fetch from official source.
    Saves the pristine raw data to output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not force_download:
        logger.info(f"Loading existing raw dataset from: {output_path}")
        df = pd.read_csv(output_path)
    else:
        logger.info(f"Downloading raw dataset to: {output_path}")
        try:
            df = download_from_ucimlrepo()
        except Exception as e:
            logger.warning(f"ucimlrepo download failed ({e}), falling back to direct URL...")
            df = download_fallback_url()
            
        # Save pristine raw data locally
        df.to_csv(output_path, index=False)
        logger.info(f"Saved pristine raw dataset ({df.shape[0]} rows, {df.shape[1]} cols) to {output_path}")

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform thorough validation of the downloaded dataset.
    Returns validation summary dictionary.
    """
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION REPORT")
    logger.info("=" * 60)
    
    shape = df.shape
    logger.info(f"Dataset Shape: {shape[0]} rows, {shape[1]} columns")
    
    # Check row count
    row_count_valid = shape[0] == EXPECTED_ROWS
    logger.info(f"Row count validation (expected={EXPECTED_ROWS}, actual={shape[0]}): {'PASS' if row_count_valid else 'FAIL'}")
    
    # Check expected columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    cols_valid = len(missing_cols) == 0
    if cols_valid:
        logger.info("All expected columns present: PASS")
    else:
        logger.error(f"Missing expected columns: {missing_cols}")
        
    # Check missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    logger.info(f"Total Missing Values across dataset: {total_nulls} ({'PASS' if total_nulls == 0 else 'WARNING'})")
    
    # Check duplicates
    duplicate_rows = df.duplicated().sum()
    logger.info(f"Duplicate rows detected in raw data: {duplicate_rows}")
    
    # Target distribution
    target_counts = df["Revenue"].value_counts().to_dict()
    target_pct = (df["Revenue"].value_counts(normalize=True) * 100).round(2).to_dict()
    logger.info(f"Target Distribution (Revenue): {target_counts}")
    logger.info(f"Target Percentages: {target_pct}")
    
    # Data types
    dtypes_summary = df.dtypes.value_counts().to_dict()
    logger.info(f"Data types breakdown: {dtypes_summary}")
    
    logger.info("=" * 60)
    
    validation_results = {
        "shape": shape,
        "row_count_valid": row_count_valid,
        "columns_valid": cols_valid,
        "missing_columns": missing_cols,
        "total_nulls": total_nulls,
        "duplicate_rows": duplicate_rows,
        "target_counts": target_counts,
        "target_percentages": target_pct,
    }
    
    return validation_results


def load_raw_data(raw_path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Helper to load and validate raw dataset."""
    df = acquire_data(raw_path)
    val = validate_dataset(df)
    if not val["columns_valid"] or not val["row_count_valid"]:
        logger.warning("Dataset validation completed with warnings.")
    else:
        logger.info("Dataset validation passed successfully.")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("\nFirst 5 rows of raw dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
