import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app

data = load_breast_cancer()
FEATURES = [name.replace(" ", "_") for name in data.feature_names]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup/lifespan events
        yield c


@pytest.fixture
def valid_payload():
    return {
        name: float(value)
        for name, value in zip(FEATURES, data.data[0])
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "1.0.0"
    assert result["number_of_features"] == 30


def test_valid_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["class_name"] in ("malignant", "benign")
    assert 0.0 <= result["probability"] <= 1.0
    # label and class name must agree (0 = malignant, 1 = benign)
    expected = {0: "malignant", 1: "benign"}
    assert result["class_name"] == expected[result["prediction"]]


@pytest.mark.parametrize("feature", FEATURES)
def test_missing_feature(client, valid_payload, feature):
    valid_payload.pop(feature)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_wrong_data_type(client, valid_payload):
    valid_payload["mean_radius"] = "invalid_string"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_extra_feature(client, valid_payload):
    valid_payload["extra_feature"] = 123.45
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422  # requires extra="forbid" on the schema