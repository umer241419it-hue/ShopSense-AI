from pathlib import Path
import csv
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import ShopperPurchasePredictor

PROD_MODEL_PATH = ROOT / "models" / "production_model.joblib"
ACADEMIC_MODEL_PATH = ROOT / "models" / "best_model.joblib"
PROD_METRICS_PATH = ROOT / "reports" / "production_model_comparison.csv"
ACADEMIC_METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

app = FastAPI(title="ShopSense AI ML Service", version="2.0.0")
predictor = ShopperPurchasePredictor(PROD_MODEL_PATH)


from typing import Literal

VALID_MONTHS = Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VALID_VISITOR_TYPES = Literal["Returning_Visitor", "New_Visitor", "Other"]


class ShopperSession(BaseModel):
    """
    Production real-time observable session schema (10 features).
    Excludes retrospective attribution features (PageValues, BounceRates, ExitRates)
    and anonymized categorical dataset IDs (OperatingSystems, Browser, Region, TrafficType).
    """
    Administrative: float = Field(0, ge=0)
    Administrative_Duration: float = Field(0, ge=0)
    Informational: float = Field(0, ge=0)
    Informational_Duration: float = Field(0, ge=0)
    ProductRelated: float = Field(0, ge=0)
    ProductRelated_Duration: float = Field(0, ge=0)
    SpecialDay: float = Field(0, ge=0, le=1)
    Month: VALID_MONTHS = "May"
    VisitorType: VALID_VISITOR_TYPES = "Returning_Visitor"
    Weekend: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "shopsense-ml",
        "active_model": "production_model.joblib",
        "production_model_loaded": PROD_MODEL_PATH.exists(),
        "academic_model_loaded": ACADEMIC_MODEL_PATH.exists(),
        "model_loaded": PROD_MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(payload: ShopperSession):
    try:
        return predictor.predict(payload.model_dump())
    except Exception as error:
        raise HTTPException(status_code=500, detail="Model inference failed.") from error


@app.get("/model-info")
def model_info():
    def read_metrics_csv(path: Path):
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        return rows

    prod_rows = read_metrics_csv(PROD_METRICS_PATH)
    academic_rows = read_metrics_csv(ACADEMIC_METRICS_PATH)

    best_prod = max(prod_rows, key=lambda row: float(row.get("F1-Score", 0)), default=None) if prod_rows else None
    best_academic = max(academic_rows, key=lambda row: float(row.get("F1-Score", 0)), default=None) if academic_rows else None

    return {
        "active_model": "production_model.joblib",
        "model": best_prod.get("Model") if best_prod else "Production Random Forest (Tuned)",
        "model_type": "Production (Real-Time Observable Session Features)",
        "artifact": str(PROD_MODEL_PATH.relative_to(ROOT)),
        "metrics": prod_rows,
        "features": [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "SpecialDay",
            "Month",
            "VisitorType",
            "Weekend",
        ],
        "excluded_features": [
            "PageValues",
            "BounceRates",
            "ExitRates",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
        ],
        "academic_benchmark": {
            "model": best_academic.get("Model") if best_academic else "Random Forest (Tuned)",
            "artifact": str(ACADEMIC_MODEL_PATH.relative_to(ROOT)),
            "metrics": academic_rows,
            "notes": "Historical UCI 17-feature benchmark with retrospective PageValues and anonymized categories.",
        },
        "dataset": {
            "name": "UCI Online Shoppers Purchasing Intention",
            "rows": 12330,
            "target": "Revenue",
        },
        "technical_note": (
            "The project maintains two model contexts. The academic benchmark model uses the complete historical "
            "UCI feature set for reproducible dataset benchmarking. The production inference model uses only features "
            "that are genuinely observable during an active browsing session. This separation prevents retrospective "
            "attribution variables and anonymized dataset identifiers from being fabricated as user inputs."
        ),
    }

