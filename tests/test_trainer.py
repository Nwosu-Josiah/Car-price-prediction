import os
import pandas as pd
from src.training.trainer import train_model
from pathlib import Path

def test_trainer_creates_artifacts(tmp_path):
    dataset_path = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "price": [8000, 12000, 15000],
        "year": [2010, 2016, 2019],
        "mileage": [90000, 50000, 20000],
        "condition": ["excellent", "good", "fair"],
        "fuel_type": ["petrol", "diesel", "petrol"],
        "transmission": ["manual", "automatic", "automatic"],
        "manufacturer": ["Toyota", "Ford", "Honda"]
    })
    df.to_csv(dataset_path, index=False)

    artifacts_dir = tmp_path / "artifacts"
    train_model(
        data_path=str(dataset_path),
        model_dir=str(artifacts_dir),
        target_col="price",
        num_boost_round=10, 
        early_stopping_rounds=5,
        test_size=0.5  
    )

    assert (artifacts_dir / "model.json").exists()
    assert (artifacts_dir / "preprocessor.pkl").exists()
