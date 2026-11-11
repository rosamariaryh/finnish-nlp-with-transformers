
from app.model import model, predict_intent


def test_model_returns_valid_prediction():
    result = predict_intent("Millainen sää tänään on?")

    assert "intent" in result
    assert "confidence" in result


def test_prediction_intent_is_valid():
    result = predict_intent("Millainen sää tänään on?")

    valid_labels = model.config.id2label.values()

    assert result["intent"] in valid_labels


def test_prediction_confidence_is_between_zero_and_one():
    result = predict_intent("Millainen sää tänään on?")

    assert 0.0 <= result["confidence"] <= 1.0


def test_prediction_confidence_is_rounded_to_four_decimals():
    result = predict_intent("Millainen sää tänään on?")

    assert result["confidence"] == round(result["confidence"], 4)
