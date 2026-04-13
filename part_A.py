import argparse
import os
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    """A small CNN suitable for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

# ---------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
def save_confusion_matrix(labels, preds, classes, output_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
    # -----------------------------
    # Validate arguments
    # -----------------------------
    if args.epochs <= 0:
        raise ValueError("Epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be > 0")
    if args.lr <= 0:
        raise ValueError("Learning rate must be > 0")

    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Transforms
    # -----------------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # -----------------------------
    # Dataset
    # -----------------------------
    train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    sample_batch = next(iter(train_loader))[0]
    print("First batch shape:", sample_batch.shape)

    # -----------------------------
    # Model
    # -----------------------------
    model = SimpleCNN().to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0
    log_data = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        log_data.append(log_entry)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc
            }, os.path.join(args.output_dir, "best_model.pth"))

    # -----------------------------
    # Save logs
    # -----------------------------
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    classes = train_dataset.classes
    save_confusion_matrix(labels, preds, classes, args.output_dir)

    # -----------------------------
    # Inference Demo
    # -----------------------------
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = test_dataset[0][0].unsqueeze(0).to(device)
    pred = model(sample).argmax(dim=1).item()
    print("Inference demo prediction:", pred)

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved PyTorch CIFAR-10 Training Script")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--tracker", type=str, choices=["wandb", "mlflow", "none"], default="none")

    args = parser.parse_args()
    main(args) 