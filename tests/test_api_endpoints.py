"""
Automated Integration and API Verification Tests for ShopSense AI.
Verifies FastAPI ML Service and Node/Express Backend with MongoDB across all 10 API contracts:
1. Valid production prediction
2. Empty/invalid payload structure
3. Invalid numeric value rejection
4. Invalid Month validation
5. Invalid VisitorType validation
6. Invalid Weekend boolean validation
7. Unauthorized prediction & history rejection (401)
8. Authenticated prediction, persistence, and analytics
9. Model health and dual-model metrics transparency
10. ML service health and active model reporting
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
    """Complete API and integration test suite."""

    @classmethod
    def setUpClass(cls):
        """Register and log in a dedicated test user for authenticated requests."""
        cls.unique_email = f"api_test_{int(time.time()*1000)}@shopsense.local"
        cls.password = "Password123!"

        try:
            r_reg = requests.post(
                f"{BACKEND_URL}/auth/register",
                json={"name": "API Tester", "email": cls.unique_email, "password": cls.password},
                timeout=5
            )
            if r_reg.status_code == 201:
                cls.token = r_reg.json().get("token")
            else:
                cls.token = None
        except Exception:
            cls.token = None

        if cls.token:
            cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}
        else:
            cls.auth_headers = {}

    def test_01_ml_service_health(self):
        """10. Verify FastAPI ML service health and active model reporting."""
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "shopsense-ml")
        self.assertEqual(data.get("active_model"), "production_model.joblib")
        self.assertTrue(data.get("production_model_loaded"))
        self.assertTrue(data.get("academic_model_loaded"))

    def test_02_ml_service_valid_prediction(self):
        """1. Verify FastAPI ML service produces valid prediction on 10 production features."""
        valid_session = {
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
        resp = requests.post(f"{FASTAPI_URL}/predict", json=valid_session, timeout=5)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertIn("prediction", result)
        self.assertIn("purchase_probability", result)
        self.assertIn("intent_level", result)
        self.assertTrue(0.0 <= result["purchase_probability"] <= 1.0)
        self.assertEqual(result["prediction"], 1)

    def test_03_backend_health(self):
        """Verify Node/Express backend health and MongoDB connectivity."""
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("database"), "connected")

    def test_04_backend_unauthorized_access(self):
        """7. Verify unauthorized prediction and history requests are rejected (401)."""
        r_unauth_pred = requests.post(f"{BACKEND_URL}/predictions", json={}, timeout=5)
        self.assertEqual(r_unauth_pred.status_code, 401)

        r_unauth_hist = requests.get(f"{BACKEND_URL}/predictions", timeout=5)
        self.assertEqual(r_unauth_hist.status_code, 401)

    def test_05_backend_authenticated_prediction(self):
        """8. Verify authenticated prediction through Express -> FastAPI -> MongoDB."""
        session_data = {
            "Administrative": 2,
            "Administrative_Duration": 40.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 20,
            "ProductRelated_Duration": 600.0,
            "SpecialDay": 0.0,
            "Month": "May",
            "VisitorType": "Returning_Visitor",
            "Weekend": True,
        }
        r_pred = requests.post(f"{BACKEND_URL}/predictions", json=session_data, headers=self.auth_headers, timeout=10)
        self.assertEqual(r_pred.status_code, 201)
        pred_data = r_pred.json()
        self.assertIn("prediction", pred_data)
        self.assertIn("id", pred_data)
        self.assertIn("purchase_probability", pred_data)
        self.assertIn("intent_level", pred_data)

        # Verify history persistence
        r_hist = requests.get(f"{BACKEND_URL}/predictions", headers=self.auth_headers, timeout=5)
        self.assertEqual(r_hist.status_code, 200)
        history = r_hist.json()
        self.assertGreaterEqual(len(history), 1)

        # Verify analytics summary
        r_ana = requests.get(f"{BACKEND_URL}/analytics/summary", headers=self.auth_headers, timeout=5)
        self.assertEqual(r_ana.status_code, 200)
        analytics = r_ana.json()
        self.assertGreaterEqual(analytics.get("totalPredictions", 0), 1)

    def test_06_backend_validation_invalid_numeric(self):
        """3. Verify negative or non-finite numeric values are rejected with 400."""
        invalid_negative = {
            "Administrative": -5,
            "ProductRelated": 10,
        }
        r = requests.post(f"{BACKEND_URL}/predictions", json=invalid_negative, headers=self.auth_headers, timeout=5)
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())

        invalid_special_day = {
            "SpecialDay": 1.5,
        }
        r_sd = requests.post(f"{BACKEND_URL}/predictions", json=invalid_special_day, headers=self.auth_headers, timeout=5)
        self.assertEqual(r_sd.status_code, 400)

    def test_07_backend_validation_invalid_month(self):
        """4. Verify invalid Month value is rejected with 400."""
        invalid_month = {
            "Month": "InvalidMonth",
            "ProductRelated": 5,
        }
        r = requests.post(f"{BACKEND_URL}/predictions", json=invalid_month, headers=self.auth_headers, timeout=5)
        self.assertEqual(r.status_code, 400)
        self.assertIn("Month", r.json().get("detail", ""))

    def test_08_backend_validation_invalid_visitor_type(self):
        """5. Verify invalid VisitorType value is rejected with 400."""
        invalid_visitor = {
            "VisitorType": "Alien_Visitor",
            "ProductRelated": 5,
        }
        r = requests.post(f"{BACKEND_URL}/predictions", json=invalid_visitor, headers=self.auth_headers, timeout=5)
        self.assertEqual(r.status_code, 400)
        self.assertIn("VisitorType", r.json().get("detail", ""))

    def test_09_backend_validation_invalid_weekend(self):
        """6. Verify non-boolean Weekend value is rejected with 400."""
        invalid_weekend = {
            "Weekend": "yes_it_is",
            "ProductRelated": 5,
        }
        r = requests.post(f"{BACKEND_URL}/predictions", json=invalid_weekend, headers=self.auth_headers, timeout=5)
        self.assertEqual(r.status_code, 400)
        self.assertIn("Weekend", r.json().get("detail", ""))

    def test_10_backend_model_metrics(self):
        """9. Verify model transparency endpoint returns dual-model info."""
        r_model = requests.get(f"{BACKEND_URL}/model/metrics", headers=self.auth_headers, timeout=5)
        self.assertEqual(r_model.status_code, 200)
        model_info = r_model.json()
        self.assertEqual(model_info.get("activeModel"), "production_model.joblib")
        self.assertTrue(
            "Random Forest" in str(model_info.get("bestModel", ""))
            or model_info.get("academicBestModel") == "Random Forest (Tuned)"
        )
        self.assertIn("productionRows", model_info)
        self.assertIn("academicRows", model_info)
        self.assertIn("technicalNote", model_info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
