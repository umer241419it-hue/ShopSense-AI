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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


class ShopperPurchasePredictor:
    """Wrapper class for loading the trained model pipeline and executing inference."""
    
    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please run train.py first.")
        self.pipeline = joblib.load(self.model_path)
        logger.info(f"Loaded model pipeline from: {self.model_path}")

    def prepare_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Format and engineer raw inputs for model consumption."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame.")
            
        # Ensure all required raw numerical and categorical columns exist
        for col in RAW_NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
        for col in RAW_CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "Missing"
            df[col] = df[col].astype(str)
            
        # Apply feature engineering
        engineered_df = engineer_features(df)
        return engineered_df

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate prediction and probabilities for input data.
        Returns detailed prediction dictionary.
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
                "intent_level": intent_tier
            })
            
        if isinstance(data, dict) or (isinstance(data, pd.DataFrame) and len(data) == 1):
            return results[0]
        return {"predictions": results}


def predict_sample(sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to predict single sample."""
    predictor = ShopperPurchasePredictor()
    return predictor.predict(sample_dict)


if __name__ == "__main__":
    # Test with a realistic high-intent session
    high_intent_session = {
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
    
    # Test with a low-intent bounce session
    low_intent_session = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "BounceRates": 0.20,
        "ExitRates": 0.20,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "Feb",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False
    }
    
    predictor = ShopperPurchasePredictor()
    
    print("\n--- Testing High Intent Session ---")
    res_high = predictor.predict(high_intent_session)
    print(f"Prediction: {res_high['prediction_label']}")
    print(f"Purchase Probability: {res_high['purchase_probability']:.2%}")
    print(f"Intent Level: {res_high['intent_level']}")
    
    print("\n--- Testing Low Intent Session ---")
    res_low = predictor.predict(low_intent_session)
    print(f"Prediction: {res_low['prediction_label']}")
    print(f"Purchase Probability: {res_low['purchase_probability']:.2%}")
    print(f"Intent Level: {res_low['intent_level']}")
