from pathlib import Path
import csv
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import ShopperPurchasePredictor

MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

app = FastAPI(title="ShopSense AI ML Service", version="1.1.0")
predictor = ShopperPurchasePredictor(MODEL_PATH)


class ShopperSession(BaseModel):
    Administrative: float = Field(0, ge=0)
    Administrative_Duration: float = Field(0, ge=0)
    Informational: float = Field(0, ge=0)
    Informational_Duration: float = Field(0, ge=0)
    ProductRelated: float = Field(1, ge=0)
    ProductRelated_Duration: float = Field(0, ge=0)
    BounceRates: float = Field(0, ge=0, le=1)
    ExitRates: float = Field(0, ge=0, le=1)
    PageValues: float = Field(0, ge=0)
    SpecialDay: float = Field(0, ge=0, le=1)
    Month: str = "May"
    OperatingSystems: int = Field(2, ge=1, le=8)
    Browser: int = Field(2, ge=1, le=13)
    Region: int = Field(1, ge=1, le=9)
    TrafficType: int = Field(2, ge=1, le=20)
    VisitorType: str = "Returning_Visitor"
    Weekend: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "shopsense-ml",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(payload: ShopperSession):
    try:
        return predictor.predict(payload.model_dump())
    except Exception as error:
        raise HTTPException(status_code=500, detail="Model inference failed.") from error


@app.get("/model-info")
def model_info():
    with METRICS_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass

    best = max(rows, key=lambda row: float(row.get("F1-Score", 0)), default=None)

    return {
        "model": best.get("Model") if best else None,
        "artifact": str(MODEL_PATH.relative_to(ROOT)),
        "metrics": rows,
        "dataset": {
            "name": "UCI Online Shoppers Purchasing Intention",
            "rows": 12330,
            "target": "Revenue",
        },
    }
