from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.prediction.predictor import Predictor

app = FastAPI(title="Car Price Prediction API")

# Initialize Predictor
predictor = Predictor()


class CarFeatures(BaseModel):
    year: int
    odometer: float
    condition: str
    fuel: str
    transmission: str
    manufacturer: str
    model: str
   
@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API is running smoothly."}


@app.post("/predict")
async def predict_price(car: CarFeatures):
    """Predict car price from input features."""
    try:
        input_data = car.dict()
        for key, value in input_data.items():
            if isinstance(value, str):
                input_data[key] = value.strip().lower()
        logger.info(f"Received user input for prediction: {input_data}")
        prediction = predictor.predict(input_data)
        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
