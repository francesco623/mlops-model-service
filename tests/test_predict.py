from main import app
from routers.predict import load_model
from fastapi.testclient import TestClient
import pytest

class FakeModel():
    def __init__(self, probability):
        self.probability = probability

    def predict(self, dmatrix):
        return [self.probability]

def test_app():
    assert app is not None

@pytest.fixture
def client():
    
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_predict_survived(client):

    app.dependency_overrides[load_model] = lambda: FakeModel(0.7)

    payload = {"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "sex": 1}
    response = client.post("/predict", json = payload)

    assert response.status_code == 200
    assert response.json()["survived"] == 1
    assert response.json()["probability"] == 0.7

def test_predict_not_survived(client):

    app.dependency_overrides[load_model] = lambda: FakeModel(0.3)

    payload = {"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "sex": 1}
    response = client.post("/predict", json = payload)

    assert response.status_code == 200
    assert response.json()["survived"] == 0
    assert response.json()["probability"] == 0.3

def test_predict_missing_fare(client):

    app.dependency_overrides[load_model] = lambda: FakeModel(0.7)

    payload = {"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "sex": 1}
    response = client.post("/predict", json = payload)

    assert response.status_code == 422
