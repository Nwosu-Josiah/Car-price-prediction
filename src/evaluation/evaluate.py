import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# Paths
# ============================================================
MODEL_PATH = Path("artifacts/model.json")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.pkl")
TEST_DATA_PATH = Path("datasets/processed/test.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = REPORT_DIR / "metrics.json"
EVAL_REPORT_PATH = REPORT_DIR / "evaluation_report.json"

def align_columns(df: pd.DataFrame, preprocessor):
    expected_features = []
    for name, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            expected_features.extend(cols)
        elif isinstance(cols, str):
            expected_features.append(cols)
    missing = [col for col in expected_features if col not in df.columns]
    for col in missing:
        df[col] = np.nan  # Add missing columns with NaN so SimpleImputer can handle them
    return df[expected_features]

# ============================================================
# Evaluation Function
# ============================================================
def evaluate():
    logger.info("🚀 Starting model evaluation...")

    # Load preprocessor and model
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    # Load test data
    df_test = pd.read_csv(TEST_DATA_PATH)
    logger.info(f"Loaded test data: {df_test.shape}")

    # Separate features and target
    if "price" not in df_test.columns:
        raise ValueError("❌ 'price' column not found in test data.")

    X_test = df_test.drop(columns=["price"])
    y_test = df_test["price"]

    # Transform data using the preprocessor
    logger.info("Applying preprocessing transformations...")
    X_test= align_columns(X_test, preprocessor)
    X_processed = preprocessor.transform(X_test)

    # Convert to DMatrix for XGBoost
    dtest = xgb.DMatrix(X_processed, label=y_test)

    # Predict log prices and revert log1p transform
    logger.info("Running model predictions...")
    y_pred_log = model.predict(dtest)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # ✅ Clip extreme predictions to prevent overflow
    y_pred = np.clip(y_pred, 0, np.percentile(y_pred, 99.9))
    # Compute evaluation metrics
    logger.info("Computing evaluation metrics...")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mean_price = float(np.mean(y_true))
    rel_rmse = rmse / mean_price

    # Build metrics dictionary
    metrics = {
        "RMSE": float(rmse),
        "R2": float(r2),
        "Relative_RMSE": float(rel_rmse),
        "Mean_Price": mean_price,
        "Num_Samples": int(len(y_true)),
    }

    # Log and save
    logger.success(f"✅ Evaluation complete — RMSE: {rmse:.2f}, R²: {r2:.4f}")
    logger.info(f"📊 Saving metrics to {METRICS_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump({
            "model_path": str(MODEL_PATH),
            "preprocessor_path": str(PREPROCESSOR_PATH),
            "metrics": metrics
        }, f, indent=4)

    return metrics


if __name__ == "__main__":
    evaluate()
