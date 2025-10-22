from src.models.model_factory import get_model
import os

def test_get_xgboost_model_default(tmp_path):
    model_dir = str(tmp_path / "artifacts")
    os.makedirs(model_dir, exist_ok=True)
    model = get_model("xgboost", model_dir=model_dir)
    assert model is not None
    # the created model should have model_dir attribute
    assert hasattr(model, "model_dir")
    assert model.model_dir == model_dir

def test_get_model_invalid_type_raises():
    import pytest
    with pytest.raises(ValueError):
        get_model("unsupported_model")
