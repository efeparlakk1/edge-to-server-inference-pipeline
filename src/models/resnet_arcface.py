"""
ResNet18 backbone — torchvision'dan çağrılır, son FC silinir.

timm veya torchvision ikisi de kullanılabilir.
torchvision tercih edildi: ek bağımlılık gerektirmiyor.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(embedding_dim: int = 512, pretrained: bool = True) -> nn.Module:
    """
    ResNet18'in son FC katmanını çıkar, yerine embedding projeksiyonu koy.

    Args:
        embedding_dim : ArcFaceLoss'taki embedding_size ile eşleşmeli
        pretrained    : ImageNet ağırlıkları ile başla (önerilir)
    Returns:
        model : (B, 3, H, W) → (B, embedding_dim)
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    # Son FC'yi embedding projeksiyonuyla değiştir
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, embedding_dim, bias=False),
        nn.BatchNorm1d(embedding_dim),
    )

    return model


if __name__ == "__main__":
    model = build_model(embedding_dim=512, pretrained=False)
    x     = torch.randn(4, 3, 32, 32)
    print(f"Çıkış boyutu: {model(x).shape}")  # (4, 512)