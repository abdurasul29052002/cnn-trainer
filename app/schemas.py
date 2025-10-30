from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of input texts to analyze")
    top_k: int = Field(1, ge=1, le=6, description="Number of top emotions to return per text")


class EmotionScore(BaseModel):
    label: str
    score: float


class PredictItemResult(BaseModel):
    top: List[EmotionScore]


class PredictResponse(BaseModel):
    results: List[PredictItemResult]
    labels: List[str]


class HealthResponse(BaseModel):
    status: str
    model_path: Optional[str]
    num_labels: int
    labels: List[str]
