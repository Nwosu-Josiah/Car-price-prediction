from typing import Dict, Any

from src.models.xgboost_model import XGBoostModel


def get_model(model_type: str = "xgboost", model_dir: str = "artifacts", **kwargs) -> Any:
    model_type = model_type.lower()
    if model_type == "xgboost":
        return XGBoostModel(model_dir=model_dir, **kwargs)

    raise ValueError(f"Unsupported model_type='{model_type}'. Supported: ['xgboost']")
