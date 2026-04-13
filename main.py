from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from keyboard.keyboard import ChagataiKeyboard

app = FastAPI(title="Qalam Next Token Prediction API")
kb = ChagataiKeyboard()


class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5


class SuggestionItem(BaseModel):
    word: str
    score: float
    source: str


class PredictResponse(BaseModel):
    input_text: str
    suggestions: list[SuggestionItem]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    suggestions = kb.suggest_full(request.text, request.top_k)
    return PredictResponse(
        input_text=request.text,
        suggestions=[
            SuggestionItem(word=s.word, score=s.score, source=s.source)
            for s in suggestions
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
