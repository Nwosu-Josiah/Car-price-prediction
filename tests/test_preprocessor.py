import os
import pandas as pd
import pytest
from src.models.xgboost_model import XGBoostModel

@pytest.fixture
def small_df(tmp_path):
    df = pd.DataFrame({
        "year": [2010, 2015, 2018],
        "mileage": [120000, 80000, 30000],
        "condition": ["excellent", "good", "fair"],
        "fuel_type": ["petrol", "diesel", "petrol"],
        "transmission": ["manual", "automatic", "automatic"],
        "manufacturer": ["Toyota", "Ford", "Honda"],
        "price": [3000, 7000, 15000]
    })
    return df

def test_xgboost_train_save_load_predict(tmp_path, small_df):
    model_dir = str(tmp_path / "artifacts")
    os.makedirs(model_dir, exist_ok=True)

    model = XGBoostModel(model_dir=model_dir)

    model.train(small_df, target_column="price")

    assert os.path.exists(model.model_path)
    assert os.path.exists(model.preprocessor_path)

    model_loaded = XGBoostModel(model_dir=model_dir)
    model_loaded.load()
    input_df = small_df.drop(columns=["price"])
    preds = model_loaded.predict(input_df)
    assert isinstance(preds, list)
    assert len(preds) == len(input_df)
    assert all(isinstance(p, float) for p in preds)
