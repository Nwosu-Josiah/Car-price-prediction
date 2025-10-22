import os
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    def __init__(self, model_dir="artifacts"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.json")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        self.model = None
        self.preprocessor = None
        os.makedirs(model_dir, exist_ok=True)

        self.numeric_features = ["year", "mileage"]
        self.categorical_features = [
            "condition", "fuel_type", "transmission", "manufacturer"
        ]

    def build_preprocessor(self):
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        self.preprocessor = preprocessor
        return preprocessor

    def train(self, df: pd.DataFrame, target_column: str = "price"):
        logger.info("Starting XGBoost model training...")

        X = df[self.numeric_features + self.categorical_features]
        y = df[target_column]

        preprocessor = self.build_preprocessor()
        X_processed = preprocessor.fit_transform(X)

        dtrain = xgb.DMatrix(X_processed, label=y)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.05,
            "max_depth": 9,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            verbose_eval=100
        )

        # Save artifacts
        joblib.dump(preprocessor, self.preprocessor_path)
        self.model.save_model(self.model_path)

        logger.info(f"Model and preprocessor saved to {self.model_dir}")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError("Model or preprocessor not found. Train the model first.")

        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        logger.info("Model and preprocessor loaded successfully.")

    def predict(self, input_data: pd.DataFrame):
        """Run inference using pre-trained model."""
        if self.model is None or self.preprocessor is None:
            self.load()

        X_processed = self.preprocessor.transform(input_data)
        dtest = xgb.DMatrix(X_processed)
        preds = self.model.predict(dtest)
        return preds.tolist()
