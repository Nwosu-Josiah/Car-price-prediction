import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from google.cloud import storage  
from pathlib import Path
from scipy import sparse


class Predictor:
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "artifacts/model.json")
        self.preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")

        # Download GCS artifacts if needed
        self.model_local = self._get_local_copy(self.model_path)
        self.preprocessor_local = self._get_local_copy(self.preprocessor_path)

        # Load Booster (faster than XGBRegressor)
        self.model = xgb.Booster()
        self.model.load_model(self.model_local)

        # Load preprocessor
        self.preprocessor = joblib.load(self.preprocessor_local)

        logger.info("Model + preprocessor loaded successfully.")

    def _get_local_copy(self, path: str) -> str:
        """Download GCS file to /tmp (Cloud Run safe)"""
        if path.startswith("gs://"):
            logger.info(f"Downloading artifact from GCS: {path}")
            bucket_name, blob_path = path.replace("gs://", "").split("/", 1)

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            local_dir = Path("/tmp/gcs_artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / os.path.basename(blob_path)

            blob.download_to_filename(local_file)
            logger.success(f"Downloaded {blob_path} to {local_file}")
            return str(local_file)

        return path

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        expected_cols = [
            "year", "odometer", "condition", "fuel",
            "transmission", "manufacturer", "model"
        ]

        for col in expected_cols:
            if col not in input_data:
                raise ValueError(f"Missing input feature: {col}")

        df = pd.DataFrame([input_data])
        logger.debug(f"Input DataFrame before processing: {df.to_dict(orient='records')}")
        return df

    def predict(self, input_data: dict) -> float:
        try:
            input_df = self.preprocess_input(input_data)

            # Preprocess
            X_processed = self.preprocessor.transform(input_df)
            if not sparse.issparse(X_processed):
                X_processed = sparse.csr_matrix(X_processed)

            # Ensure single-thread inference
            dmatrix = xgb.DMatrix(X_processed, nthread=1)

            # Predict
            log_pred = self.model.predict(dmatrix)
            prediction = float(np.expm1(log_pred[0]))

            logger.info(f"Prediction OK: {prediction:,.2f}")
            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
