
from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_ID = "rosaryhanen/finnish-intent-classifier"


@lru_cache(maxsize=1)
def load_model():
    """Download and load the model once, then reuse it."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    model.eval()

    return tokenizer, model


def predict_intent(text: str):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()

    label = model.config.id2label[prediction]

    return {
        "intent": label,
        "confidence": round(confidence, 4),
    }
