from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model sekali saja saat start
model = joblib.load("model/model.joblib")

app = FastAPI(title="Profit Prediction API")

# Request schema
class PredictionRequest(BaseModel):
    UnitPrice: float
    Discount: float
    Quantity: float

@app.post("/predict")
def predict(data: PredictionRequest):
    features = [[
        data.UnitPrice,
        data.Discount,
        data.Quantity
    ]]

    prediction = model.predict(features)

    return {
        "predicted_profit": prediction[0]
    }
