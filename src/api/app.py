import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from src.prediction.predictor import Predictor

app = FastAPI(title="Car Price Prediction API")

predictor = None  


class CarFeatures(BaseModel):
    year: int
    odometer: float
    condition: str
    fuel: str
    transmission: str
    manufacturer: str
    model: str


@app.on_event("startup")
def load_model():
    """
    Load XGBoost model once per worker at startup.
    Prevents heavy import-time memory usage.
    """
    global predictor
    if predictor is None:
        logger.info("Loading Predictor and XGBoost model...")
        predictor = Predictor()
        logger.info("Model loaded successfully.")


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

        logger.info(f"Received prediction request: {input_data}")

        prediction = predictor.predict(input_data)
        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Only used when running locally without Docker
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
