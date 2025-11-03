import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import yaml


@dataclass
class ImageConfig:
    seed: int
    output_dir: str
    logs_dir: str
    reports_dir: str
    # dataset
    root_dir: Optional[str]
    train_dir: Optional[str]
    val_dir: Optional[str]
    labels: Optional[List[str]]
    val_size: float
    shuffle_before_split: bool
    # model
    backbone: str
    pretrained: bool
    image_size: int
    grayscale: bool
    # training
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "config.yaml") -> ImageConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project = cfg.get("project", {})
    dataset_img = cfg.get("dataset_image", {})
    model_img = cfg.get("model_image", {})
    training_img = cfg.get("training_image", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return ImageConfig(
        seed=project.get("seed", 42),
        output_dir=project.get("output_dir", "models\\emotion-cnn"),
        logs_dir=project.get("logs_dir", "outputs\\logs"),
        reports_dir=project.get("reports_dir", "outputs\\reports"),
        root_dir=dataset_img.get("root_dir"),
        train_dir=dataset_img.get("train_dir"),
        val_dir=dataset_img.get("val_dir"),
        labels=dataset_img.get("labels"),
        val_size=float(dataset_img.get("val_size", 0.1)),
        shuffle_before_split=bool(dataset_img.get("shuffle_before_split", True)),
        backbone=model_img.get("backbone", "resnet18"),
        pretrained=bool(model_img.get("pretrained", True)),
        image_size=int(model_img.get("image_size", 96)),
        grayscale=bool(model_img.get("grayscale", False)),
        epochs=int(training_img.get("epochs", 10)),
        train_batch_size=int(training_img.get("train_batch_size", 64)),
        eval_batch_size=int(training_img.get("eval_batch_size", 128)),
        learning_rate=float(training_img.get("learning_rate", 1e-3)),
        weight_decay=float(training_img.get("weight_decay", 1e-4)),
        num_workers=int(training_img.get("num_workers", 4)),
        device=device,
    )


def ensure_dirs(cfg: ImageConfig):
    for d in [cfg.output_dir, cfg.logs_dir, cfg.reports_dir, "outputs", "models", "data"]:
        os.makedirs(d, exist_ok=True)


def build_transforms(cfg: ImageConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    im_size = cfg.image_size
    common = []
    if cfg.grayscale:
        # Convert grayscale to 3-channel for pretrained CNNs
        common.append(transforms.Grayscale(num_output_channels=3))
    train_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1) if not cfg.grayscale else transforms.Lambda(lambda x: x),
        *common,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        *common,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def load_datasets(cfg: ImageConfig):
    train_tfms, val_tfms = build_transforms(cfg)

    if cfg.train_dir and cfg.val_dir:
        train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(cfg.val_dir, transform=val_tfms)
        # Ensure class order consistent
        if train_ds.classes != val_ds.classes:
            raise ValueError("Train and Val class folders mismatch. Ensure same class names.")
        classes = train_ds.classes
    elif cfg.root_dir:
        full_ds = datasets.ImageFolder(cfg.root_dir, transform=train_tfms)
        classes = full_ds.classes
        # Optional shuffle before split is handled by torch generator seed to ensure reproducibility
        n_total = len(full_ds)
        n_val = int(n_total * cfg.val_size)
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=g)
        # Assign different transforms
        train_subset.dataset.transform = train_tfms
        val_subset.dataset.transform = val_tfms
        train_ds, val_ds = train_subset, val_subset
    else:
        raise ValueError("Provide either dataset_image.train_dir & val_dir or dataset_image.root_dir in config.yaml")

    # If labels list is given, enforce order according to config; otherwise use discovered classes
    if cfg.labels is not None:
        # Validate that config labels match folder classes
        lower_classes = [c.lower() for c in classes]
        for lab in cfg.labels:
            if lab.lower() not in lower_classes:
                raise ValueError(f"Label '{lab}' from config not found in dataset classes {classes}")
        classes = cfg.labels

    return train_ds, val_ds, classes


def create_model(backbone: str, num_classes: int, pretrained: bool, grayscale: bool):
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if grayscale:
            # Adjust first conv to accept 1 channel; alternatively we keep grayscale->3ch transform
            # We will rely on transforms to output 3ch; so no change needed.
            pass
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # Simple custom small CNN for 96x96
        class SmallCNN(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 48x48
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 24x24
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 12x12
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, num_classes),
                )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        in_ch = 1 if grayscale else 3
        # If grayscale, we still transform to 3 channels earlier; keep 3
        model = SmallCNN(num_classes)
    return model


def evaluate(model, loader, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1, y_true, y_pred


def save_reports(cfg: ImageConfig, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    os.makedirs(cfg.reports_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    report_path = os.path.join(cfg.reports_dir, "classification_report_image.txt")
    cm_path = os.path.join(cfg.reports_dir, "confusion_matrix_image.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, encoding="utf-8")
    print(f"Saved report to {report_path}")
    print(f"Saved confusion matrix to {cm_path}")


def main():
    cfg = load_config()
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    train_ds, val_ds, classes = load_datasets(cfg)
    num_classes = len(classes)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = create_model(cfg.backbone, num_classes, cfg.pretrained, cfg.grayscale)
    model = model.to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, cfg.device)

        print(f"Epoch {epoch}/{cfg.epochs} - TrainLoss: {train_loss:.4f} - ValAcc: {val_acc:.4f} - ValF1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                'model_state': model.state_dict(),
                'classes': classes,
                'backbone': cfg.backbone,
                'image_size': cfg.image_size,
                'grayscale': cfg.grayscale,
            }

    # Save best model
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, "cnn_image_best.pt")
    torch.save(best_state, model_path)
    print(f"Saved best CNN model to {model_path}")

    # Final evaluation report using the best model
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
    _, _, y_true, y_pred = evaluate(model, val_loader, cfg.device)
    save_reports(cfg, y_true, y_pred, classes)


if __name__ == "__main__":
    main()
