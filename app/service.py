import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml


def load_labels_from_config(model_dir: str) -> List[str]:
    # Try to read id2label from model config.json
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.isfile(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        id2label = cfg.get('id2label')
        if isinstance(id2label, dict):
            # Ensure ordered by id key
            labels = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
            return labels
    return []


class EmotionModel:
    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}. Train the model first or set EMOTION_MODEL_DIR.")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.labels = load_labels_from_config(model_dir)
        if not self.labels:
            # Fallback: try model.config.id2label
            cfg = getattr(self.model, 'config', None)
            if cfg is not None and hasattr(cfg, 'id2label') and isinstance(cfg.id2label, dict) and len(cfg.id2label) > 0:
                self.labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
            else:
                # As a last resort, just index labels
                self.labels = [str(i) for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def predict(self, texts: List[str], top_k: int = 1) -> List[List[Dict[str, float]]]:
        if top_k < 1:
            top_k = 1
        top_k = min(top_k, len(self.labels))
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        results: List[List[Dict[str, float]]] = []
        for row in probs:
            scores, idxs = torch.topk(row, k=top_k)
            item = []
            for score, idx in zip(scores.tolist(), idxs.tolist()):
                item.append({
                    'label': self.labels[idx],
                    'score': float(score)
                })
            results.append(item)
        return results


def get_model_dir() -> str:
    # Priority: ENV var -> config.yaml project.output_dir -> default path
    env_path = os.getenv('EMOTION_MODEL_DIR')
    if env_path:
        return env_path
    cfg_path = 'config.yaml'
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        project = cfg.get('project', {})
        out_dir = project.get('output_dir')
        if out_dir:
            return out_dir
    return 'models\\emotion-distilbert'
