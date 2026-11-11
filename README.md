# Finnish NLP

Finnish NLP is a natural language processing project exploring Transformer-based models for Finnish text. It includes notebooks covering tokenization, fine-tuning, evaluation and optimization, alongside a FastAPI service that exposes a fine-tuned intent classification model through a REST API.

## Technology

This project relies on the following stack:
- **Python** as the primary programming language
- **Hugging Face Transformers** for pretrained models and tokenization
- **Hugging Face** for training data
- **PyTorch** for model inference
- **FastAPI** for REST API
- **Pydantic** for input validation
- **Docker** for containerization
- **pytest** for testing
- **uv** for dependency management

## Notebooks

You can find four experimental notebooks focusing on the Transformer models in the repository:
1. Tokenization
2. Fine-tuning
3. Evaluation
4. Optimization

## User Intent Classification
User intent classification uses `bert-base-finnish-cased-v1` from TurkuNLP. It has been fine-tuned with the [Amazon user intent data](https://huggingface.co/datasets/mteb/amazon_massive_intent)

### Iteration 1
The first iteration focuses exclusively on Finnish-language data and
the 10 most frequent intent classes in the training split.

The labels in iteration one include:
```
"calendar_query": 0,
    "calendar_set": 1,
    "datetime_query": 2,
    "email_query": 3,
    "email_sendemail": 4,
    "general_quirky": 5,
    "news_query": 6,
    "play_music": 7,
    "qa_factoid": 8,
    "weather_query": 9,
```
The model was trained with the following hyperparameters:

```
PER_DEVICE_TRAIN_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-5
EVALUATION_STRATEGY = "epoch"
```

The training was performed in a Google Colab environment to access `NVIDIA T4 GPU` and speed up the training.

Newer models like `roberta-large-finnish-v2` from FinnishNLP and `finnish-modernbert-base` from TurkuNLP were also tested with the same configuration. 
Among the evaluated models, `bert-base-finnish-cased-v1`
achieved the best macro F1 score of **0.9107**. Its main errors were related to unclear class boundaries, and short and/or ambiguous inputs.

Evaluation was performed on the same test split for all models.

## Limitations

- Predictions may be less reliable for ambiguous or very short inputs.
- The model is limited to the intent categories included in its training data.
- Confidence scores represent model confidence, not guaranteed prediction accuracy.

## API Documentation

Interactive API documentation is available at:
http://localhost:8000/docs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict user intent |

Example request:
```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Millainen sää tänään on?"}'
```

Example response:
```
{
  "intent": "weather_query",
  "confidence": 0.95
}
```

## Installation and Usage

This project uses `uv` for dependency management:
- `uv add <library>` to add a dependency
- `uv sync` to update dependencies

Run tests:
- `uv run pytest -v`

Docker:
- Build image: `docker build -t finnish-nlp-service .`
- Start the container: `docker run -d --name finnish-nlp-container -p 8000:8000 finnish-nlp-service`
- Verify containers: `docker images`
- Check container is running: `docker ps`
- Stop container: `docker stop finnish-nlp-container`
- Get logs: `docker logs finnish-nlp-container`
- Remove container: `docker rm finnish-nlp-container`

## Acknowledgements

- [TurkuNLP](https://huggingface.co/TurkuNLP)
- [Hugging Face](https://huggingface.co/)
- [Amazon MASSIVE dataset](https://huggingface.co/datasets/mteb/amazon_massive_intent)