"""
Automated Test Suite for Online Shopper Purchasing Intention Pipeline.
Validates data integrity, preprocessing, dual-model loading (production & academic),
inference, and app compatibility.
Uses Python's built-in unittest so it can run anywhere without extra dependencies.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, validate_dataset, EXPECTED_COLUMNS, EXPECTED_ROWS
from src.preprocessing import prepare_data, engineer_features
from src.predict import (
    ShopperPurchasePredictor,
    PRODUCTION_MODEL_PATH,
    ACADEMIC_MODEL_PATH,
    PRODUCTION_NUMERICAL_FEATURES,
    PRODUCTION_CATEGORICAL_FEATURES,
)


class TestShopperPipeline(unittest.TestCase):
    """Test suite covering the complete dual-model ML lifecycle."""

    def test_01_data_loader(self):
        """Verify raw dataset exists, has expected shape and columns."""
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame, "Raw data must be a pandas DataFrame")
        self.assertEqual(df.shape[0], EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {df.shape[0]}")
        self.assertEqual(df.shape[1], len(EXPECTED_COLUMNS), f"Expected {len(EXPECTED_COLUMNS)} columns, got {df.shape[1]}")
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(df.isnull().sum().sum(), 0, "Raw dataset should contain no missing values")

    def test_02_preprocessing(self):
        """Verify preprocessing produces clean stratified train/test sets and engineered features."""
        df = load_raw_data()
        prep = prepare_data(df, test_size=0.2, random_state=42, drop_duplicates=True, use_feature_engineering=True)

        X_train = prep["X_train"]
        X_test = prep["X_test"]
        y_train = prep["y_train"]
        y_test = prep["y_test"]

        self.assertEqual(len(X_train) + len(X_test), len(df.drop_duplicates()))
        self.assertIn("TotalPageViews", X_train.columns)
        self.assertIn("TotalDuration", X_train.columns)
        self.assertIn("ProductRelated_Ratio", X_train.columns)
        self.assertIn("BounceExit_Product", X_train.columns)
        self.assertIn("PageValues_per_Duration", X_train.columns)
        self.assertTrue(set(np.unique(y_train)).issubset({0, 1}))
        self.assertTrue(set(np.unique(y_test)).issubset({0, 1}))

    def test_03_academic_benchmark_model(self):
        """Verify historical academic benchmark model loads offline and predicts on 17 UCI features."""
        self.assertTrue(ACADEMIC_MODEL_PATH.exists(), f"Academic model must exist at {ACADEMIC_MODEL_PATH}")

        academic_predictor = ShopperPurchasePredictor(ACADEMIC_MODEL_PATH)
        self.assertFalse(academic_predictor.is_production, "best_model.joblib should be identified as academic benchmark")

        high_intent = {
            "Administrative": 3, "Administrative_Duration": 85.0,
            "Informational": 1, "Informational_Duration": 42.0,
            "ProductRelated": 28, "ProductRelated_Duration": 1150.0,
            "BounceRates": 0.005, "ExitRates": 0.015, "PageValues": 38.5,
            "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
            "Browser": 2, "Region": 1, "TrafficType": 2,
            "VisitorType": "Returning_Visitor", "Weekend": False
        }
        res = academic_predictor.predict(high_intent)
        self.assertIn("prediction", res)
        self.assertIn("purchase_probability", res)
        self.assertTrue(0.0 <= res["purchase_probability"] <= 1.0)
        self.assertAlmostEqual(res["purchase_probability"] + res["no_purchase_probability"], 1.0, places=4)
        self.assertEqual(res["prediction"], 1, "Academic benchmark should predict Purchase for high-intent session")

    def test_04_production_model_loading_and_inference(self):
        """
        Verify production model loads offline, strictly accepts 10 real-time observable features,
        and requires NO PageValues, BounceRates, ExitRates, or anonymized client IDs.
        """
        self.assertTrue(PRODUCTION_MODEL_PATH.exists(), f"Production model must exist at {PRODUCTION_MODEL_PATH}")

        prod_predictor = ShopperPurchasePredictor(PRODUCTION_MODEL_PATH)
        self.assertTrue(prod_predictor.is_production, "production_model.joblib must be identified as production model")

        # Verify production schema contains exactly the 10 intended features
        expected_production_features = {
            "Administrative", "Administrative_Duration",
            "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration",
            "SpecialDay", "Month", "VisitorType", "Weekend"
        }
        self.assertEqual(
            set(PRODUCTION_NUMERICAL_FEATURES + PRODUCTION_CATEGORICAL_FEATURES),
            expected_production_features,
            "Production schema must contain exactly the 10 real-time observable features."
        )

        # Strict production session with ZERO retrospective features or anonymized codes
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

        # Assert no retrospective or anonymized attributes are in the session dict
        for forbidden in ["PageValues", "BounceRates", "ExitRates", "OperatingSystems", "Browser", "Region", "TrafficType"]:
            self.assertNotIn(forbidden, prod_high_intent)

        res_high = prod_predictor.predict(prod_high_intent)
        self.assertIn("prediction", res_high)
        self.assertIn("purchase_probability", res_high)
        self.assertIn("intent_level", res_high)
        self.assertTrue(0.0 <= res_high["purchase_probability"] <= 1.0)
        self.assertAlmostEqual(res_high["purchase_probability"] + res_high["no_purchase_probability"], 1.0, places=4)
        self.assertEqual(res_high["prediction"], 1, "High activity session should predict Purchase")

        # Low intent session
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
        res_low = prod_predictor.predict(prod_low_intent)
        self.assertEqual(res_low["prediction"], 0, "Low intent session should predict No Purchase")
        self.assertLess(res_low["purchase_probability"], 0.25)

    def test_05_streamlit_app_import(self):
        """Verify Streamlit app module can be imported cleanly without errors."""
        import app
        self.assertTrue(hasattr(app, "load_predictor"))
        self.assertTrue(hasattr(app, "PERSONAS"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
