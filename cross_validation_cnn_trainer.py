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
from sklearn.model_selection import StratifiedKFold

# Optional nice progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback if tqdm not present
    tqdm = None


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


def load_full_dataset(cfg: ImageConfig):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(cfg.root_dir, transform=transform)
    return full_ds, full_ds.classes


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

    # ✅ GPU / CPU tanlash
    if torch.cuda.is_available():
        cfg.device = "cuda"
        print("✅ GPU ishlatilmoqda!")
        torch.backends.cudnn.benchmark = True
    else:
        cfg.device = "cpu"
        print("⚠️ GPU topilmadi, CPU ishlatilmoqda.")

    full_ds, classes = load_full_dataset(cfg)
    num_classes = len(classes)

    labels = np.array([y for _, y in full_ds.imgs])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), 1):
        print(f"\n==== Fold {fold} ====")

        train_subset = torch.utils.data.Subset(full_ds, train_idx)
        val_subset = torch.utils.data.Subset(full_ds, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.eval_batch_size, shuffle=False)

        model = create_model(cfg.backbone, num_classes, cfg.pretrained, cfg.grayscale).to(cfg.device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None  # fold uchun eng yaxshi modelni saqlaymiz

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0

            for images, labels_ in tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch + 1}/{cfg.epochs}"):
                images, labels_ = images.to(cfg.device), labels_.to(cfg.device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels_)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            val_acc, _, _, _ = evaluate(model, val_loader, cfg.device)
            print(f"Val Acc: {val_acc:.4f} | Train Loss: {epoch_loss / len(train_loader):.4f}")

            # Fold uchun eng yaxshi modelni saqlash
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()  # save state dict only, not file yet

        # Fold tugagach, eng yaxshi modelni saqlaymiz
        best_model_path = f"outputs/cross_validation/model_fold_{fold}_acc_{best_acc:.4f}.pt"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(best_state, best_model_path)
        fold_accuracies.append(best_acc)
        print(f"✅ Fold {fold} done. Best Accuracy: {best_acc:.4f}")
        print(f"💾 Saved model: {best_model_path}")

    print("\n===================================")
    print("Fold Accuracies:", fold_accuracies)
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
