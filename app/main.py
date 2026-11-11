from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from app.model import predict_intent


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)
    print("Finnish NLP Services is running!")
    print("API documentation: http://localhost:8000/docs")
    print("Health check:      http://localhost:8000/health")
    print("=" * 50 + "\n")
    yield

app = FastAPI(
    title="Finnish NLP Services",
    description="An API providing Finnish natural language processing capabilities, including user intent classification.",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="Text to classify",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty or contain only spaces")
        return value

class PredictionResponse(BaseModel):
    intent: str
    confidence: float


@app.get("/")
def root():
    return {"message": "User Intent Classification API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, summary="Classify user intent",
    description=(
    "Predicts the intent of a Finnish text using "
    "a fine-tuned Transformer model (TurkuNLP)."
    "Possible intents include:"
    "calendar_query, calendar_set, datetime_query, email_query, email_sendemail, general_quirky, news_query, play_music, qa_factoid, weather_query "
    ))
def predict(request: PredictionRequest):
    result = predict_intent(request.text)
    return result