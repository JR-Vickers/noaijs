from fastapi import FastAPI
from pydantic import BaseModel
from packages.core.src.detectors.perplexity_gap import PerplexityGapDetector

detector = PerplexityGapDetector(
    human_model_dir="./packages/training/src/fine_tune/distilbert-human-finetuned",
    detection_model_dir="./packages/training/src/fine_tune/distilbert-detection-finetuned"
)

app = FastAPI()

# Define the request body schema
class TextRequest(BaseModel):
    text: str

# Define the response schema (optional, but good practice)
class DetectResponse(BaseModel):
    human_ppl: float
    detection_ppl: float
    gap: float
    is_ai: bool

@app.post("/detect", response_model=DetectResponse)
async def detect(request: TextRequest):
    result = detector.score(request.text)
    # Simple threshold: gap > 5 means likely AI (tune as needed)
    is_ai = result['gap'] > 5
    return DetectResponse(
        human_ppl=float(result['human_ppl']),
        detection_ppl=float(result['detection_ppl']),
        gap=float(result['gap']),
        is_ai=is_ai
    )