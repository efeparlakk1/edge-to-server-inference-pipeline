"""
E�itim döngüsü — kütüphane tabanlı sürüm.

Çalıştır:
    uv run python train.py
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.resnet_arcface import build_model
from src.losses.arcface import build_arcface_loss
from src.data.dataset import get_dataloaders, CIFAR10_CLASSES


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dim",  type=int,   default=512)
    p.add_argument("--pretrained",     action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arcface-margin", type=float, default=28.6)
    p.add_argument("--arcface-scale",  type=float, default=64.0)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=0.1)
    p.add_argument("--weight-decay",   type=float, default=5e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--data-dir",       type=str,   default="./data")
    p.add_argument("--checkpoint-dir", type=str,   default="./outputs/checkpoints")
    p.add_argument("--log-dir",        type=str,   default="./outputs/logs")
    return p.parse_args()


def validate(model, criterion, loader, device):
    """
    pytorch-metric-learning'de validation için:
    loss.get_logits(embeddings, labels) → sınıf logit'leri döner.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            loss   = criterion(embeddings, labels)
            logits = criterion.get_logits(embeddings)
            preds  = logits.argmax(dim=1)

            total_loss    += loss.item() * labels.size(0)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(cfg: argparse.Namespace) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Cihaz   : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Epochs  : {cfg.epochs}  |  Batch: {cfg.batch_size}")
    print(f"{'='*55}\n")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model     = build_model(embedding_dim=cfg.embedding_dim, pretrained=cfg.pretrained).to(device)
    criterion = build_arcface_loss(
        embedding_dim=cfg.embedding_dim,
        num_classes=len(CIFAR10_CLASSES),
        margin=cfg.arcface_margin,
        scale=cfg.arcface_scale,
    ).to(device)

    # Model + ArcFace ağırlık matrisini birlikte optimize et
    optimizer = optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)
    writer    = SummaryWriter(log_dir=cfg.log_dir)
    best_acc  = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        criterion.train()
        epoch_loss, n = 0.0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), \
                             labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss       = criterion(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n          += 1

        scheduler.step()
        val_loss, val_acc = validate(model, criterion, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train loss: {epoch_loss/n:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc*100:.2f}% | "
            f"{time.time()-t0:.1f}s"
        )

        writer.add_scalar("Loss/train",   epoch_loss / n, epoch)
        writer.add_scalar("Loss/val",     val_loss,       epoch)
        writer.add_scalar("Accuracy/val", val_acc,        epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "arcface_state":   criterion.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy":  np.random.get_state(),
                        "torch":  torch.get_rng_state(),
                        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    "val_acc":         val_acc,
                },
                os.path.join(cfg.checkpoint_dir, "best_model.pth"),
            )
            print(f"  ✓ Checkpoint kaydedildi — val_acc: {val_acc*100:.2f}%")

    writer.close()
    print(f"\nEn iyi val acc: {best_acc*100:.2f}%")


if __name__ == "__main__":
    train(get_config())