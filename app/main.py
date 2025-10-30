from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse, PredictItemResult, EmotionScore, HealthResponse
from .service import EmotionModel, get_model_dir

app = FastAPI(title="Emotion Detection API", version="1.0.0")

# CORS (optional; adjust for your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global model
    model_dir = get_model_dir()
    try:
        model = EmotionModel(model_dir)
    except FileNotFoundError as e:
        # Delay failure until first request except /health
        model = None
        print(str(e))


@app.get("/health", response_model=HealthResponse)
def health():
    if 'model' not in globals() or model is None:
        return HealthResponse(status="model_not_loaded", model_path=get_model_dir(), num_labels=0, labels=[])
    return HealthResponse(status="ok", model_path=model.model_dir, num_labels=len(model.labels), labels=model.labels)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if 'model' not in globals() or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model and set EMOTION_MODEL_DIR or project.output_dir in config.yaml")
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    results = model.predict(req.texts, top_k=req.top_k)
    output = []
    for item in results:
        output.append(PredictItemResult(top=[EmotionScore(label=it['label'], score=it['score']) for it in item]))

    return PredictResponse(results=output, labels=model.labels)
