### Emotion Detection (6-class) — Training and API Serving

This project fine-tunes a lightweight Transformer (DistilBERT) to detect 6 basic emotions in text:
- angry, disgust, fear, happy, sad, surprise

It includes:
- A configurable training pipeline (`train.py`, `config.yaml`)
- A FastAPI service (`app/main.py`) exposing `/predict` and `/health`
- Evaluation artifacts (classification report + confusion matrix)


#### 1) Setup

1. Create and activate a virtual environment
- Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```
pip install -r requirements.txt
```


#### 2) Prepare your dataset

- Default expects a single CSV at `data\dataset.csv` with columns:
  - `text` — the input sentence/document
  - `label` — one of: `angry, disgust, fear, happy, sad, surprise`
- If your dataset uses different names (e.g., `content` and `emotion`) or labels (e.g., `joy` for `happy`), edit `config.yaml`:
  - `dataset.text_column` and `dataset.label_column`
  - `dataset.label_mapping` to normalize labels (e.g., `joy: happy`)
  - `dataset.labels` to set the final 6 classes (order defines ids)
- Alternatively, provide separate train/val CSV files using `dataset.train_path` and `dataset.val_path`.

Example of minimal CSV format:
```
text,label
I am so happy today!,happy
This is terrible and makes me angry.,angry
```


#### 3) Configure training (config.yaml)

Key fields:
- `project.output_dir`: where the fine-tuned model is saved (default `models\emotion-distilbert`)
- `dataset.*`: paths, columns, labels, and optional mapping
- `model.pretrained_checkpoint`: base model name (default `distilbert-base-uncased`)
- `training.*`: epochs, batch sizes, learning rate, etc.


#### 4) Train
```
python train.py
```
Outputs:
- Model + tokenizer at `models\emotion-distilbert\`
- Reports at `outputs\reports\`:
  - `classification_report.txt`
  - `confusion_matrix.csv`


#### 5) Serve the API

1) Ensure the environment variable (optional) or config points to the trained model
- Option A (env var):
```
$env:EMOTION_MODEL_DIR = "models\emotion-distilbert"
```
- Option B: Make sure `project.output_dir` in `config.yaml` is set to the same path

2) Start the API
```
uvicorn app.main:app --reload --port 8000
```

3) Check health
```
curl http://127.0.0.1:8000/health
```

4) Predict (single or batch)
```
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"texts\":[\"I am proud and happy today!\"],\"top_k\":3}"
```
Response example:
```
{
  "results": [
    {"top": [
      {"label": "happy", "score": 0.83},
      {"label": "surprise", "score": 0.10},
      {"label": "sad", "score": 0.03}
    ]}
  ],
  "labels": ["angry", "disgust", "fear", "happy", "sad", "surprise"]
}
```


#### 6) Tips
- GPU: If you have an NVIDIA GPU, set `training.fp16: true` in `config.yaml` for faster training.
- Max length: Increase `model.max_length` if your texts are long (will use more memory).
- Class imbalance: You can explore weighted loss or data augmentation if needed.


#### 7) Troubleshooting
- "Model directory not found": Train first (`python train.py`), or set `EMOTION_MODEL_DIR` to a directory with a HF model.
- Import errors: Make sure the virtual environment is active and `pip install -r requirements.txt` completed successfully.
- CSV encoding: If reading fails, try saving your CSV as UTF-8 or pass `encoding='utf-8'` in your own loader if you customize `train.py`.
