
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "User Intent Classification API is running!"
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint(monkeypatch):
    def mock_predict_intent(text):
        return {
            "intent": "weather_query",
            "confidence": 0.95,
        }

    monkeypatch.setattr(main, "predict_intent", mock_predict_intent)

    response = client.post(
        "/predict",
        json={"text": "Millainen sää tänään on?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "weather_query",
        "confidence": 0.95,
    }


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_rejects_whitespace_only_text():
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422


def test_predict_rejects_text_over_2000_characters():
    response = client.post("/predict", json={"text": "a" * 2001})

    assert response.status_code == 422
