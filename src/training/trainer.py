import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing.preprocessor import DataPreprocessor
from src.models.model_factory import get_model
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def load_dataset(csv_path: str, required_cols: Tuple[str, ...]) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return df


def train_model(
    data_path: str = "datasets/raw_data_sampled.csv",
    model_dir: str = "artifacts",
    target_col: str = "price",
    model_type: str = "xgboost",
    test_size: float = 0.1,
    random_state: int = 42,
    num_boost_round: int = 100,
    early_stopping_rounds: Optional[int] = 10,
    eval_metric: str = "rmse",
    train_params: Optional[Dict[str, Any]] = None,
) -> None:
    
    os.makedirs(model_dir, exist_ok=True)

    
    numeric_feats = ["year", "mileage"]
    categorical_feats = ["fuel_type", "transmission", "manufacturer", "condition"]
    required_cols = tuple(numeric_feats + categorical_feats + [target_col])

    logger.info("Loading dataset...")
    df = load_dataset(data_path, required_cols)

    X = df[list(numeric_feats + categorical_feats)]
    y = df[target_col]

    logger.info("Building preprocessor and transforming data...")
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    # save preprocessor
    preproc_path = os.path.join(model_dir, "preprocessor.pkl")
    preprocessor.save(preproc_path)
    logger.info(f"Preprocessor saved to {preproc_path}")

    # train/validation split for xgboost's watchlist
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )

    # create dmatrix objects
    import xgboost as xgb  
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # model factory
    logger.info("Instantiating model via factory...")
    model_obj = get_model(model_type=model_type, model_dir=model_dir, **(train_params or {}))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": eval_metric,
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
    }
    if train_params:
        params.update(train_params)

    watchlist = [(dtrain, "train"), (dval, "eval")]

    logger.info("Starting xgboost training (low-level)...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # Save model using model_obj's save model path
    model.save_model(os.path.join(model_dir, "xgb_model.json"))
    logger.info(f"XGBoost model saved to {os.path.join(model_dir, 'xgb_model.json')}")

    # Optionally allow model_obj to know about the trained booster for further wrappers
    try:
        model_obj.model = model
        # If model_obj has save() semantics, call it to maintain API consistency
        if hasattr(model_obj, "save"):
            model_obj.save(os.path.join(model_dir, "model.pkl"))
    except Exception:
        # ignore optional step
        pass

    logger.info("Training completed successfully.")
