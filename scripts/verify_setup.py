"""
Kurulum doğrulama — hızlı, ~30 saniye.
    uv run python scripts/verify_setup.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.resnet_arcface import build_model
from src.losses.arcface import build_arcface_loss
from src.data.dataset import get_dataloaders, CIFAR10_CLASSES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n── Sistem ───────────────────────────────")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Cihaz   : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"\n── Model ────────────────────────────────")
    model     = build_model(embedding_dim=512, pretrained=False).to(device)
    criterion = build_arcface_loss(embedding_dim=512, num_classes=10).to(device)
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")

    x      = torch.randn(8, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)
    emb    = model(x)
    loss   = criterion(emb, labels)
    loss.backward()

    print(f"  Embedding : {tuple(emb.shape)}")
    print(f"  Loss      : {loss.item():.4f}  finite={torch.isfinite(loss).item()}")

    print(f"\n── Veri ─────────────────────────────────")
    train_loader, _ = get_dataloaders(data_dir="./data", batch_size=32, num_workers=0)
    imgs, lbls = next(iter(train_loader))
    print(f"  Batch : {tuple(imgs.shape)}")
    print(f"  Örnek : {[CIFAR10_CLASSES[l] for l in lbls[:4].tolist()]}")

    print(f"\n✓ Hazır. Eğitim için: uv run python train.py\n")


if __name__ == "__main__":
    main()