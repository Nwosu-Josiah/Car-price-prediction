import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from loguru import logger


class CarPriceTrainer:
    def __init__(self, data_path: str , model_dir: str = "artifacts"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.json")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        os.makedirs(model_dir, exist_ok=True)
        self.processor = None

    def load_data(self):
        """Load and prepare dataset."""
        logger.info(f"Loading dataset from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Keep only relevant columns used by the API
        relevant_features = [
            "year",
            "odometer",
            "condition",
            "fuel",
            "transmission",
            "manufacturer",
            "price"
        ]
        available_cols = [col for col in relevant_features if col in df.columns]
        df = df[available_cols].dropna(subset=["price"])
        logger.info(f"Dataset shape after filtering: {df.shape}")
        return df

    def build_preprocessor(self):
        numerical_features = ["year", "odometer"]
        categorical_features = ["condition", "fuel", "transmission", "manufacturer"]

        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features)
        ])

        logger.info("Preprocessor built successfully.")
        return preprocessor

    def tune_hyperparameters(self, X_train, X_test, y_train, y_test):
        
        logger.info("Starting hyperparameter tuning with Optuna...")

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        dtrain = xgb.DMatrix(X_train_processed, label=y_train)
        dtest = xgb.DMatrix(X_test_processed, label=y_test)

        def objective(trial):
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "tree_method": "hist",
                "seed": 42,
            }

            evals = [(dtrain, "train"), (dtest, "eval")]
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=200,
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            preds = model.predict(dtest)
            rmse = np.sqrt(((preds - y_test) ** 2).mean())
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, show_progress_bar=True)

        logger.success(f"Best trial: {study.best_trial.number}, RMSE: {study.best_value:.4f}")
        logger.success(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    def train(self):
        
        df = self.load_data()
        df["price_bin"] = pd.qcut(df["price"], q=10, duplicates="drop", labels=False)
        X = df.drop(columns=["price"])
        y = np.log1p(df["price"])
        price_bins = df["price_bin"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=price_bins, random_state=42)

        # Build preprocessing pipeline
        self.preprocessor = self.build_preprocessor()
        best_params = self.tune_hyperparameters(X_train, X_test, y_train, y_test)
        best_params.update({
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": 42
        })
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train_processed, label=y_train)
        dtest = xgb.DMatrix(X_test_processed, label=y_test)


        logger.info("Starting XGBoost training with best hyperparameters...")
        evals = [(dtrain, "train"), (dtest, "eval")]

        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=50
        )

        logger.success("Training completed successfully.")

        # Save model and preprocessor
        model.save_model(self.model_path)
        joblib.dump(self.preprocessor, self.preprocessor_path)

        logger.success(f"Model saved at: {self.model_path}")
        logger.success(f"Preprocessor saved at: {self.preprocessor_path}")

        # Evaluate on test data
        preds = model.predict(dtest)
        preds_original = np.expm1(preds)
        y_test_original = np.expm1(y_test)
        rmse = np.sqrt(((preds_original - y_test_original) ** 2).mean())
        r2 = 1 - (((preds_original - y_test_original) ** 2).sum() / ((y_test_original - y_test_original.mean()) ** 2).sum())

        logger.info(f"Evaluation — RMSE: {rmse:.2f}, R²: {r2:.2f}")
        return model, self.preprocessor, rmse, r2


if __name__ == "__main__":
    trainer = CarPriceTrainer(data_path="datasets/raw_data_sampled.csv")
    trainer.train()
