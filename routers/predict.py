import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

# mlflow.set_tracking_uri("sqlite:///mlflowP2.db")

# Load model once at startup
model = mlflow.xgboost.load_model("mlruns/2/models/m-71068e38fa0847cc9db3d83bdfe697dd/artifacts")

router = APIRouter()

class PassengerData(BaseModel):
    pclass: float
    age: Optional[float] = None   # missing for some passengers
    sibsp: float
    parch: float
    fare: float
    body: Optional[float] = None  # almost always None in real usage
    sex: int                       # 1 = male, 0 = female

@router.post("/predict")
def get_prediction(data: PassengerData):
    df = pd.DataFrame([data.dict()]).astype(float)  # None → NaN (float), safe for XGBoost # JSON → DataFrame
    dmatrix = xgb.DMatrix(df)                 # DataFrame → DMatrix
    pred = model.predict(dmatrix)             # raw probability
    survived = int(pred[0] > 0.5)            # convert to 0 or 1
    return {"survived": survived, "probability": round(float(pred[0]), 4)}