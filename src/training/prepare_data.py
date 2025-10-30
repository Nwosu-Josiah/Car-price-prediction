import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.data_preprocessing.preprocessor import DataPreprocessor


RAW_DATA_PATH = Path("datasets/raw_data_sampled.csv")
PROCESSED_DATA_DIR = Path("datasets/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_PATH = Path("artifacts/preprocessor.pkl")


def prepare_data():
    logger.info(f"Loading raw dataset from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Initial dataset shape: {df.shape}")

    preprocessor = DataPreprocessor()

    # Step 1: Clean data
    df = preprocessor.clean_data(df)
    logger.info(f"After cleaning: {df.shape}")

    # Step 2: Encode categorical features
    df_encoded = preprocessor.encode_features(df, fit=True)
    logger.info(f"After encoding: {df_encoded.shape}")

    # Step 3: Split into train/test
    train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Step 4: Save processed data
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    logger.success(f"Processed train/test data saved at {PROCESSED_DATA_DIR}")
    logger.success(f"Preprocessor object saved at {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    prepare_data()
