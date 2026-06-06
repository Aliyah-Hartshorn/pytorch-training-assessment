import argparse
import os
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------
def setup_ddp():
    """Initialise the process group when launched via torchrun."""
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    """True only for rank-0 (or when not running distributed at all)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_local_rank():
    """
    torchrun sets LOCAL_RANK in the environment.
    Falls back to 0 when running outside torchrun (plain `python` call).
    """
    return int(os.environ.get("LOCAL_RANK", 0))

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

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
    """A small CNN suitable for MNIST (1-channel, 28x28 grayscale images)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, 3, padding=1),   # -> 32 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 32 x 14 x 14

            nn.Conv2d(32, 64, 3, padding=1),   # -> 64 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 64 x 7 x 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),        # 7x7 because MNIST is 28x28 (not 32x32)
            nn.ReLU(),
            nn.Linear(128, 10)                 # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, sampler, epoch):
    model.train()

    # Tell DistributedSampler which epoch we're on so shuffling differs each epoch.
    # When sampler is None (CPU / single-GPU path) this is a no-op.
    if sampler is not None:
        sampler.set_epoch(epoch)

    running_loss = 0.0
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

    # Aggregate loss/correct/total across all ranks so reported metrics are global.
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([running_loss, float(correct), float(total)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total = stats[0].item(), stats[1].item(), stats[2].item()
        # running_loss was a sum of per-batch averages per rank; normalise by world size
        running_loss /= get_world_size()

    return running_loss / len(loader), correct / total

# ---------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
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

    # NOTE: In DDP mode each rank evaluates a disjoint shard of the test set
    # (because test_loader also uses a DistributedSampler).  We all-reduce the
    # scalar counts so that the reported accuracy is the true global accuracy.
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([running_loss, float(correct), float(total)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running_loss, correct, total = stats[0].item(), stats[1].item(), stats[2].item()
        running_loss /= get_world_size()

    return running_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)

# ---------------------------------------------------------
# Confusion Matrix  (rank-0 only)
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

    # -----------------------------
    # Distributed setup
    # LOCAL_RANK is injected by torchrun; absent when running plain python.
    # -----------------------------
    local_rank = get_local_rank()
    use_ddp = "LOCAL_RANK" in os.environ and torch.cuda.is_available() and torch.cuda.device_count() > 0

    if use_ddp:
        setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Rank-0 is responsible for all I/O (stdout, file writes).
    main_proc = is_main_process()

    if main_proc:
        print(f"Using device: {device}  |  World size: {get_world_size()}")

    set_seed(42 + (dist.get_rank() if use_ddp else 0))  # unique seed per rank

    if main_proc:
        os.makedirs(args.output_dir, exist_ok=True)

    # Barrier so all ranks wait until the output dir is created by rank-0.
    if use_ddp:
        dist.barrier()

    # -----------------------------
    # Transforms
    # MNIST is grayscale so mean/std are single values (not 3-channel tuples).
    # Light augmentation (random rotation) helps the model generalise.
    # -----------------------------
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -----------------------------
    # Dataset
    # Only rank-0 downloads; all others wait at the barrier.
    # -----------------------------
    if main_proc:
        datasets.MNIST(root=args.data, train=True,  download=True)
        datasets.MNIST(root=args.data, train=False, download=True)

    if use_ddp:
        dist.barrier()  # wait for rank-0 to finish downloading

    train_dataset = datasets.MNIST(root=args.data, train=True,  download=False, transform=transform_train)
    test_dataset  = datasets.MNIST(root=args.data, train=False, download=False, transform=transform_test)

    # DistributedSampler partitions data across ranks; shuffle is handled by set_epoch().
    # In single-process mode, samplers are None and the loader falls back to shuffle=True.
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)
        train_shuffle = False   # DistributedSampler does the shuffling
    else:
        train_sampler = None
        test_sampler  = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )

    if main_proc:
        sample_batch = next(iter(train_loader))[0]
        print("First batch shape:", sample_batch.shape)  # Expected: [batch_size, 1, 28, 28]

    # -----------------------------
    # Model — wrap in DDP after moving to the correct device
    # -----------------------------
    model = SimpleCNN().to(device)

    if use_ddp:
        # find_unused_parameters=False is safe here (all params used every forward).
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    if main_proc:
        # Access underlying module for param count whether wrapped or not.
        base = model.module if use_ddp else model
        print(f"Model has {sum(p.numel() for p in base.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0
    log_data = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_sampler, epoch
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        if main_proc:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            log_data.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            })

            if val_acc > best_acc:
                best_acc = val_acc
                # Save the underlying module's state_dict (not DDP wrapper's).
                base = model.module if use_ddp else model
                torch.save(
                    {
                        "model_state_dict":     base.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc":             best_acc,
                    },
                    os.path.join(args.output_dir, "best_model.pth"),
                )

    # Wait for rank-0 to finish writing the checkpoint before others proceed.
    if use_ddp:
        dist.barrier()

    # -----------------------------
    # Save logs  (rank-0 only)
    # -----------------------------
    if main_proc:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(log_data, f, indent=4)

        # -----------------------------
        # Confusion Matrix  (rank-0 only; uses preds/labels from its own shard,
        # which is sufficient for a visual summary)
        # -----------------------------
        classes = [str(i) for i in range(10)]
        save_confusion_matrix(labels, preds, classes, args.output_dir)

        # -----------------------------
        # Inference Demo  (rank-0 only)
        # -----------------------------
        # Load best weights into a clean model (no DDP wrapper needed for inference).
        inference_model = SimpleCNN().to(device)
        checkpoint = torch.load(
            os.path.join(args.output_dir, "best_model.pth"),
            map_location=device,
        )
        inference_model.load_state_dict(checkpoint["model_state_dict"])
        inference_model.eval()

        sample = test_dataset[0][0].unsqueeze(0).to(device)
        pred   = inference_model(sample).argmax(dim=1).item()
        actual = test_dataset[0][1]
        print(f"Inference demo — Actual: {actual} | Predicted: {pred}")

    # -----------------------------
    # Clean up process group
    # -----------------------------
    if use_ddp:
        cleanup_ddp()

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Training Script (DDP-ready)")

    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data",       type=str,   default="./data")
    parser.add_argument("--output-dir", type=str,   default="./outputs")
    parser.add_argument("--tracker",    type=str,   choices=["wandb", "mlflow", "none"], default="none")

    args = parser.parse_args()
    main(args)