from fastapi import FastAPI 
from pydantic import BaseModel 
from src.prediction.predictor import Predictor
import joblib 
import pandas as pd 
app = FastAPI(title="Car Price Prediction API") 
predictor = Predictor(model_path="artifacts/model.json")
class CarInput(BaseModel): 
    year: int 
    mileage: float 
    condition: str 
    fuel_type: str
    transmission: str
    manufacturer: str
  
 
@app.post("/predict") 
async def predict_price(car: CarInput): 
        prediction = predictor.predict(car.dict()) 
        return {"predicted_price": float(prediction)} 
