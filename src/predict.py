"""
Inference and Prediction Module for Online Shopper Purchase Intention.
Loads the trained pipeline artifact and generates predictions for single or batch inputs.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from src.preprocessing import (
    engineer_features,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERICAL_FEATURES
)

logger = logging.getLogger(__name__)

PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "production_model.joblib"
ACADEMIC_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_MODEL_PATH = PRODUCTION_MODEL_PATH

PRODUCTION_NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "SpecialDay",
]

PRODUCTION_CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "Weekend",
]


class ShopperPurchasePredictor:
    """
    Wrapper class for loading a trained model pipeline and executing inference.
    Supports both:
    1. Production model (10 real-time observable session features)
    2. Academic benchmark model (17 historical UCI features with retrospective attribution)
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.is_production = self._detect_production_model()
        model_type = "Production (Observable Features)" if self.is_production else "Academic Benchmark (Historical UCI)"
        logger.info(f"Loaded {model_type} pipeline from: {self.model_path}")

    def _detect_production_model(self) -> bool:
        """Detect whether the pipeline artifact is configured for production 10-feature schema."""
        if "production" in self.model_path.name.lower():
            return True
        try:
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            if preprocessor and hasattr(preprocessor, "transformers"):
                num_cols = preprocessor.transformers[0][2]
                if len(num_cols) == len(PRODUCTION_NUMERICAL_FEATURES):
                    return True
        except Exception:
            pass
        return False

    def prepare_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Format and validate input data for model consumption."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame.")

        if self.is_production:
            # Production pipeline: exactly 10 real-time observable features
            for col in PRODUCTION_NUMERICAL_FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            for col in PRODUCTION_CATEGORICAL_FEATURES:
                if col not in df.columns:
                    df[col] = "Missing"
                df[col] = df[col].astype(str)

            # Weekend must be stringified for OneHotEncoder compatibility
            df["Weekend"] = df["Weekend"].astype(str)
            return df[PRODUCTION_NUMERICAL_FEATURES + PRODUCTION_CATEGORICAL_FEATURES]

        # Academic benchmark pipeline: 17 raw UCI features + engineered features
        for col in RAW_NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in RAW_CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "Missing"
            df[col] = df[col].astype(str)

        return engineer_features(df)

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate prediction and probabilities for input data.
        Returns standardized prediction dictionary.
        """
        processed_input = self.prepare_input(data)

        preds = self.pipeline.predict(processed_input)
        probs = self.pipeline.predict_proba(processed_input)

        results = []
        for pred, prob in zip(preds, probs):
            prob_no_purchase = float(prob[0])
            prob_purchase = float(prob[1])
            is_purchase = bool(pred == 1)

            # Determine intent tier
            if prob_purchase >= 0.70:
                intent_tier = "Very High"
            elif prob_purchase >= 0.50:
                intent_tier = "High"
            elif prob_purchase >= 0.25:
                intent_tier = "Moderate"
            else:
                intent_tier = "Low"

            results.append({
                "prediction": int(pred),
                "is_purchase": is_purchase,
                "prediction_label": "Purchase" if is_purchase else "No Purchase",
                "purchase_probability": prob_purchase,
                "no_purchase_probability": prob_no_purchase,
                "confidence": max(prob_purchase, prob_no_purchase),
                "intent_level": intent_tier,
            })

        if isinstance(data, dict) or (isinstance(data, pd.DataFrame) and len(data) == 1):
            return results[0]
        return {"predictions": results}


def predict_sample(sample_dict: Dict[str, Any], model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Helper function to predict single sample."""
    predictor = ShopperPurchasePredictor(model_path)
    return predictor.predict(sample_dict)


if __name__ == "__main__":
    # Test production predictor with 10 real-time features
    prod_high_intent = {
        "Administrative": 3,
        "Administrative_Duration": 85.0,
        "Informational": 1,
        "Informational_Duration": 42.0,
        "ProductRelated": 28,
        "ProductRelated_Duration": 1150.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }

    prod_low_intent = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "SpecialDay": 0.0,
        "Month": "Feb",
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }

    print("\n=== Testing Production Model (10 Real-Time Features) ===")
    prod_predictor = ShopperPurchasePredictor(PRODUCTION_MODEL_PATH)
    res_high = prod_predictor.predict(prod_high_intent)
    print(f"High Intent: {res_high['prediction_label']} ({res_high['purchase_probability']:.2%}) - Tier: {res_high['intent_level']}")
    res_low = prod_predictor.predict(prod_low_intent)
    print(f"Low Intent:  {res_low['prediction_label']} ({res_low['purchase_probability']:.2%}) - Tier: {res_low['intent_level']}")

    print("\n=== Testing Academic Benchmark Model (17 UCI Features) ===")
    academic_predictor = ShopperPurchasePredictor(ACADEMIC_MODEL_PATH)
    academic_high = dict(prod_high_intent, BounceRates=0.005, ExitRates=0.015, PageValues=38.5, OperatingSystems=2, Browser=2, Region=1, TrafficType=2)
    res_acad = academic_predictor.predict(academic_high)
    print(f"Academic Benchmark: {res_acad['prediction_label']} ({res_acad['purchase_probability']:.2%}) - Tier: {res_acad['intent_level']}")
