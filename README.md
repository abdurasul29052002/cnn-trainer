### Emotion Detection from Images (CNN)

This project trains a CNN image classifier to detect 6 basic facial emotions:
- angry, disgust, fear, happy, sad, surprise

It includes a simple, configurable training pipeline for images only:
- `train_image_cnn.py` — end-to-end training script (ResNet18 by default, or a small custom CNN)
- `config.yaml` — configure dataset paths, model, and training hyperparameters
- Evaluation artifacts (classification report + confusion matrix)

#### 1) Setup

1. Create and activate a virtual environment (Windows PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```
pip install -r requirements.txt
```

#### 2) Prepare your dataset (ImageFolder)

Option A — single root folder (auto split by `val_size`):
```
data\images\angry\*.jpg
data\images\disgust\*.jpg
data\images\fear\*.jpg
data\images\happy\*.jpg
data\images\sad\*.jpg
data\images\surprise\*.jpg
```

Option B — separate folders for train/val:
```
data\images_train\<class>\*.jpg
data\images_val\<class>\*.jpg
```
Set these in `config.yaml` under `dataset_image`.

#### 3) Configure training (config.yaml)

Key fields:
- `project.output_dir`: where the trained model is saved (default `models\emotion-cnn`)
- `dataset_image.root_dir` or (`train_dir`, `val_dir`)
- `dataset_image.labels`: enforced label order (optional; defaults to folder alphabetical order)
- `model_image.backbone`: `resnet18` | `resnet34` | `small`
- `model_image.image_size`: default 96 (images are resized)
- `model_image.grayscale`: set `true` if your images are grayscale
- `training_image.*`: epochs, batch sizes, learning rate, workers

#### 4) Train
```
python train_image_cnn.py
```
Outputs:
- Best model checkpoint: `models\emotion-cnn\cnn_image_best.pt`
- Reports in `outputs\reports\`:
  - `classification_report_image.txt`
  - `confusion_matrix_image.csv`

#### 5) Tips
- GPU: Script auto-selects CUDA if available.
- Grayscale images: set `model_image.grayscale: true` (pipeline expands to 3 channels for pretrained CNNs).
- Class imbalance: You can try class-balanced sampling or loss weighting.

#### 6) Troubleshooting
- "Provide either train_dir/val_dir or root_dir": Check `dataset_image` paths in `config.yaml`.
- Import errors: ensure virtual environment is active and `pip install -r requirements.txt` completed successfully.
