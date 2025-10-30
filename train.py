import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorWithPadding)


@dataclass
class Config:
    seed: int
    output_dir: str
    logs_dir: str
    reports_dir: str
    dataset_path: Optional[str]
    train_path: Optional[str]
    val_path: Optional[str]
    text_column: str
    label_column: str
    label_mapping: Dict[str, str]
    labels: List[str]
    val_size: float
    shuffle_before_split: bool
    pretrained_checkpoint: str
    max_length: int
    training: Dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project = cfg.get("project", {})
    dataset = cfg.get("dataset", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    return Config(
        seed=project.get("seed", 42),
        output_dir=project.get("output_dir", "models\\emotion-distilbert"),
        logs_dir=project.get("logs_dir", "outputs\\logs"),
        reports_dir=project.get("reports_dir", "outputs\\reports"),
        dataset_path=dataset.get("path"),
        train_path=dataset.get("train_path"),
        val_path=dataset.get("val_path"),
        text_column=dataset.get("text_column", "text"),
        label_column=dataset.get("label_column", "label"),
        label_mapping=dataset.get("label_mapping", {}),
        labels=dataset.get("labels", ["angry", "disgust", "fear", "happy", "sad", "surprise"]),
        val_size=float(dataset.get("val_size", 0.1)),
        shuffle_before_split=bool(dataset.get("shuffle_before_split", True)),
        pretrained_checkpoint=model.get("pretrained_checkpoint", "distilbert-base-uncased"),
        max_length=model.get("max_length", 160),
        training=training,
    )


def ensure_dirs(cfg: Config):
    for d in [cfg.output_dir, cfg.logs_dir, cfg.reports_dir, "outputs", "models", "data"]:
        os.makedirs(d, exist_ok=True)


def load_dataset(cfg: Config) -> DatasetDict:
    if cfg.train_path and cfg.val_path:
        train_df = pd.read_csv(cfg.train_path)
        val_df = pd.read_csv(cfg.val_path)
    elif cfg.dataset_path:
        df = pd.read_csv(cfg.dataset_path)
        df = df[[cfg.text_column, cfg.label_column]].dropna()
        if cfg.training.get("shuffle_before_split", True):
            df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        val_size = cfg.training.get("val_size", 0.1)
        n_val = max(1, int(len(df) * float(val_size)))
        val_df = df.iloc[:n_val]
        train_df = df.iloc[n_val:]
    else:
        raise ValueError("Please provide dataset.path or (train_path and val_path) in config.yaml")

    # Normalize labels if mapping provided
    if cfg.label_mapping:
        def map_label(x: str) -> str:
            return cfg.label_mapping.get(str(x).strip().lower(), str(x).strip().lower())
        train_df[cfg.label_column] = train_df[cfg.label_column].apply(map_label)
        val_df[cfg.label_column] = val_df[cfg.label_column].apply(map_label)

    # Filter to allowed labels
    allowed = set([l.lower() for l in cfg.labels])
    train_df = train_df[train_df[cfg.label_column].str.lower().isin(allowed)]
    val_df = val_df[val_df[cfg.label_column].str.lower().isin(allowed)]

    # Create label2id
    labels_sorted = [l for l in cfg.labels]
    label2id = {l: i for i, l in enumerate(labels_sorted)}

    # Map to ids
    train_df["label_id"] = train_df[cfg.label_column].str.lower().map(label2id)
    val_df["label_id"] = val_df[cfg.label_column].str.lower().map(label2id)

    train_ds = Dataset.from_pandas(train_df[[cfg.text_column, "label_id"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[[cfg.text_column, "label_id"]], preserve_index=False)

    return DatasetDict({"train": train_ds, "validation": val_ds}), label2id


def tokenize_function(examples, tokenizer: AutoTokenizer, text_column: str, max_length: int):
    return tokenizer(examples[text_column], truncation=True, max_length=max_length)


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1_macro}
    return compute_metrics


def save_reports(cfg: Config, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    os.makedirs(cfg.reports_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    report_path = os.path.join(cfg.reports_dir, "classification_report.txt")
    cm_path = os.path.join(cfg.reports_dir, "confusion_matrix.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, encoding="utf-8")
    print(f"Saved report to {report_path}")
    print(f"Saved confusion matrix to {cm_path}")


def main():
    cfg = load_config()
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    raw_datasets, label2id = load_dataset(cfg)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint)
    tokenized = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer, cfg.text_column, cfg.max_length),
        batched=True,
    )

    num_labels = len(label2id)
    model_config = AutoConfig.from_pretrained(
        cfg.pretrained_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_checkpoint, config=model_config
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.training.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg.training.get("per_device_eval_batch_size", 32),
        learning_rate=float(cfg.training.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.training.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.training.get("warmup_ratio", 0.06)),
        lr_scheduler_type=cfg.training.get("lr_scheduler_type", "linear"),
        gradient_accumulation_steps=int(cfg.training.get("gradient_accumulation_steps", 1)),
        evaluation_strategy=cfg.training.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.training.get("save_strategy", "epoch"),
        logging_steps=int(cfg.training.get("logging_steps", 50)),
        load_best_model_at_end=bool(cfg.training.get("load_best_model_at_end", True)),
        metric_for_best_model=cfg.training.get("metric_for_best_model", "f1_macro"),
        greater_is_better=bool(cfg.training.get("greater_is_better", True)),
        fp16=bool(cfg.training.get("fp16", False)),
        logging_dir=cfg.logs_dir,
        report_to=["none"],
        save_total_limit=2,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
    )

    trainer.train()

    # Evaluate and save reports
    preds_output = trainer.predict(tokenized["validation"])
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids
    save_reports(cfg, y_true, y_pred, [id2label[i] for i in range(len(id2label))])

    # Save model
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
