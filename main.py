from fastapi import FastAPI
from pydantic import BaseModel, Field
from keyboard.keyboard import ChagataiKeyboard

app = FastAPI(title="Qalam Next Token Prediction API")
kb = ChagataiKeyboard()


class PredictRequest(BaseModel):
    text: str
    top_k: int = Field(default=5, ge=1, le=50)


class FuzzyRequest(PredictRequest):
    max_dist: int = Field(default=2, ge=0, le=5)


class PhoneticRequest(PredictRequest):
    max_edits: int = Field(default=1, ge=0, le=5)


class SuggestionItem(BaseModel):
    word: str
    score: float
    source: str


class PredictResponse(BaseModel):
    input_text: str
    suggestions: list[SuggestionItem]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    suggestions = kb.suggest_full(request.text, top_k=request.top_k)
    return PredictResponse(
        input_text=request.text,
        suggestions=[
            SuggestionItem(word=s.word, score=s.score, source=s.source)
            for s in suggestions
        ],
    )

@app.post("/fuzzy", response_model=PredictResponse)
def fuzzy(request: FuzzyRequest):
    suggestions = kb.suggest_fuzzy(
        request.text,
        max_dist=request.max_dist,
        top_k=request.top_k,
    )
    return PredictResponse(
        input_text=request.text,
        suggestions=[
            SuggestionItem(word=s.word, score=s.score, source=s.source)
            for s in suggestions
        ],
    )

@app.post("/phonetic", response_model=PredictResponse)
def phonetic(request: PhoneticRequest):
    suggestions = kb.suggest_phonetic(
        request.text,
        max_edits=request.max_edits,
        top_k=request.top_k,
    )
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
