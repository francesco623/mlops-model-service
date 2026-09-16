import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache

router = APIRouter()

class PassengerData(BaseModel):
    pclass: float
    age: Optional[float] = None   # missing for some passengers
    sibsp: float
    parch: float
    fare: float
    body: Optional[float] = None  # almost always None in real usage
    sex: int                       # 1 = male, 0 = female

@lru_cache    # Load model once at first request, then cached
def load_model() -> xgb.core.Booster:

    model = mlflow.xgboost.load_model("model")
    return model

@router.post("/predict")
def get_prediction(data: PassengerData, model = Depends(load_model)):
    df = pd.DataFrame([data.model_dump()]).astype(float)  # None → NaN (float), safe for XGBoost # JSON → DataFrame
    dmatrix = xgb.DMatrix(df)                 # DataFrame → DMatrix
    pred = model.predict(dmatrix)             # raw probability
    survived = int(pred[0] > 0.5)            # convert to 0 or 1
    return {"survived": survived, "probability": round(float(pred[0]), 4)}
