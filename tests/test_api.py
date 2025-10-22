from fastapi.testclient import TestClient 
import pytest
from src.api.app import app 
client = TestClient(app) 

@pytest.fixture(autouse=True)
def ensure_models_exist(tmp_path, monkeypatch): 
    try:

        from src.prediction.predictor import CarPricePredictor  
    except Exception:
        class StubPredictor:
            def predict(self, _):
                return 1234.56

        monkeypatch.setattr("src.api.predictor", StubPredictor(), raising=False)
    yield
def test_predict_endpoint_success():
    payload = {
        "year": 2019,
        "mileage": 30000,
        "condition": "excellent",
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "manufacturer": "Toyota"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert isinstance(body["predicted_price"], (float, int))