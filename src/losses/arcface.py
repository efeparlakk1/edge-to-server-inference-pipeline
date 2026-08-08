"""
ArcFace Loss — pytorch-metric-learning wrapper.

Elle yazmak yerine battle-tested kütüphaneyi kullanıyoruz.
Kaynak: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss
"""

from pytorch_metric_learning.losses import ArcFaceLoss


def build_arcface_loss(embedding_dim: int, num_classes: int, margin: float = 28.6, scale: float = 64.0):
    """
    Args:
        embedding_dim : Model çıkış boyutu (ResNet18 için 512)
        num_classes   : Sınıf sayısı (CIFAR-10 için 10)
        margin        : Açısal marjin derecesi (pytorch-metric-learning derece cinsinden bekler; 28.6° ≈ 0.5 rad)
        scale         : Logit ölçek faktörü
    Returns:
        ArcFaceLoss instance — .parameters() ile optimizer'a dahil edilmeli
    """
    return ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_dim,
        margin=margin,
        scale=scale,
    )