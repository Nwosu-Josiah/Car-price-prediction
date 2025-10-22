import pandas as pd
from src.models.xgboost_model import XGBoostModel
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class CarPricePredictor:
    def __init__(self):
        self.model = XGBoostModel(model_dir="artifacts")
        self.model.load()

    def predict(self, user_input: dict) -> float:
        logger.info(f"Received user input for prediction: {user_input}")

        input_df = pd.DataFrame([user_input])
        prediction = self.model.predict(input_df)
        logger.info(f"Prediction result: {prediction[0]}")
        return round(prediction[0], 2)
