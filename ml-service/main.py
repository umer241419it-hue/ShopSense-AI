from pathlib import Path
import sys,csv
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from src.predict import ShopperPurchasePredictor
app=FastAPI(title="ShopSense AI ML Service",version="1.0.0")
predictor=ShopperPurchasePredictor(ROOT/"models"/"best_model.joblib")
class ShopperSession(BaseModel):
 Administrative:float=Field(0,ge=0);Administrative_Duration:float=Field(0,ge=0);Informational:float=Field(0,ge=0);Informational_Duration:float=Field(0,ge=0);ProductRelated:float=Field(1,ge=0);ProductRelated_Duration:float=Field(0,ge=0);BounceRates:float=Field(0,ge=0,le=1);ExitRates:float=Field(0,ge=0,le=1);PageValues:float=Field(0,ge=0);SpecialDay:float=Field(0,ge=0,le=1);Month:str="May";OperatingSystems:int=Field(2,ge=1);Browser:int=Field(2,ge=1);Region:int=Field(1,ge=1);TrafficType:int=Field(2,ge=1);VisitorType:str="Returning_Visitor";Weekend:bool=False
@app.get("/health")
def health():return {"status":"ok","service":"shopsense-ml"}
@app.post("/predict")
def predict(payload:ShopperSession):
 try:return predictor.predict(payload.model_dump())
 except Exception as e:raise HTTPException(status_code=500,detail=str(e))
@app.get("/model-info")
def model_info():
 with (ROOT/"reports"/"model_comparison.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
 return {"model":"Random Forest (Tuned)","artifact":"models/best_model.joblib","metrics":rows,"dataset":{"name":"UCI Online Shoppers Purchasing Intention","rows":12330}}