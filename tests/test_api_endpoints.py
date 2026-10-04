"""
Automated Integration and API Verification Tests for ShopSense AI.
Verifies FastAPI ML Service and Node/Express Backend with MongoDB.
"""

import sys
import unittest
import time
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FASTAPI_URL = "http://127.0.0.1:8000"
BACKEND_URL = "http://127.0.0.1:5000/api"


class TestShopSenseAPIs(unittest.TestCase):
    """Integration test suite for ML Service and Backend APIs."""

    def test_01_ml_service_health(self):
        """Verify FastAPI ML service health endpoint."""
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "shopsense-ml")
        self.assertTrue(data.get("model_loaded"))

    def test_02_ml_service_predict(self):
        """Verify FastAPI ML service prediction and validation."""
        valid_session = {
            "Administrative": 3, "Administrative_Duration": 85.0,
            "Informational": 1, "Informational_Duration": 42.0,
            "ProductRelated": 28, "ProductRelated_Duration": 1150.0,
            "BounceRates": 0.005, "ExitRates": 0.015, "PageValues": 38.5,
            "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
            "Browser": 2, "Region": 1, "TrafficType": 2,
            "VisitorType": "Returning_Visitor", "Weekend": False
        }
        resp = requests.post(f"{FASTAPI_URL}/predict", json=valid_session, timeout=5)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("prediction", result)
        self.assertIn("purchase_probability", result)
        self.assertIn("intent_level", result)
        self.assertEqual(result["prediction"], 1)

        # Test validation error with negative Administrative count
        invalid_session = {"Administrative": -10}
        resp_inv = requests.post(f"{FASTAPI_URL}/predict", json=invalid_session, timeout=5)
        self.assertEqual(resp_inv.status_code, 422)

    def test_03_backend_health(self):
        """Verify Node/Express backend health and MongoDB connectivity."""
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("database"), "connected")

    def test_04_backend_auth_and_persistence(self):
        """Verify full user registration, login, JWT token, prediction, history, and analytics."""
        unique_email = f"api_test_{int(time.time()*1000)}@shopsense.local"
        password = "Password123!"

        # Register
        r_reg = requests.post(
            f"{BACKEND_URL}/auth/register",
            json={"name": "API Tester", "email": unique_email, "password": password},
            timeout=5
        )
        self.assertEqual(r_reg.status_code, 201)
        self.assertIn("token", r_reg.json())

        # Login
        r_login = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"email": unique_email, "password": password},
            timeout=5
        )
        self.assertEqual(r_login.status_code, 200)
        token = r_login.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Predict via backend
        session_data = {
            "Administrative": 2, "Administrative_Duration": 40.0,
            "Informational": 0, "Informational_Duration": 0.0,
            "ProductRelated": 20, "ProductRelated_Duration": 600.0,
            "BounceRates": 0.01, "ExitRates": 0.02, "PageValues": 15.0,
            "SpecialDay": 0.0, "Month": "May", "OperatingSystems": 2,
            "Browser": 2, "Region": 1, "TrafficType": 2,
            "VisitorType": "Returning_Visitor", "Weekend": True
        }
        r_pred = requests.post(f"{BACKEND_URL}/predictions", json=session_data, headers=headers, timeout=10)
        self.assertEqual(r_pred.status_code, 201)
        pred_data = r_pred.json()
        self.assertIn("prediction", pred_data)
        self.assertIn("id", pred_data)

        # History
        r_hist = requests.get(f"{BACKEND_URL}/predictions", headers=headers, timeout=5)
        self.assertEqual(r_hist.status_code, 200)
        history = r_hist.json()
        self.assertGreaterEqual(len(history), 1)

        # Analytics summary
        r_ana = requests.get(f"{BACKEND_URL}/analytics/summary", headers=headers, timeout=5)
        self.assertEqual(r_ana.status_code, 200)
        analytics = r_ana.json()
        self.assertGreaterEqual(analytics.get("totalPredictions", 0), 1)

        # Model metrics
        r_model = requests.get(f"{BACKEND_URL}/model/metrics", headers=headers, timeout=5)
        self.assertEqual(r_model.status_code, 200)
        model_info = r_model.json()
        self.assertTrue(
            "Random Forest" in str(model_info.get("bestModel", ""))
            or model_info.get("academicBestModel") == "Random Forest (Tuned)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
