"""
CIFAR-10 veri seti yükleme ve augmentation pipeline'ı.

Neden CIFAR-10?
  - 10 sınıf, 60.000 görüntü (50k eğitim / 10k test)
  - 32×32 piksel → küçük, hızlı eğitim
  - ArcFace'in avantajını görmek için yeterli sınıf çeşitliliği var
  - GPU'yu boğmadan 1-2 günde tamamlanacak deney

Augmentation stratejisi:
  - Train: RandomCrop + HorizontalFlip → veri çeşitliliği
  - Val  : Yalnızca normalize → adil değerlendirme
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 istatistikleri (önceden hesaplanmış, sabit)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool) -> transforms.Compose:
    """
    Train ve validation için ayrı transform pipeline'ları.

    RandomCrop neden padding ile?
      32×32 görüntüyü 4 piksel pad et (40×40 olur), sonra 32×32 kırp.
      Bu, modelin nesnenin konumuna aşırı öğrenmesini engeller.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 train ve validation DataLoader'larını döndür.

    Args:
        data_dir    : Verinin indirileceği/okunacağı klasör
        batch_size  : RTX 4070 Ti SUPER için 256 rahatça sığar.
                      Bellek sorununda 128'e düşür.
        num_workers : CPU core sayısına göre ayarla (4-8 arası iyi)
        pin_memory  : True olunca CPU→GPU transfer hızlanır (CUDA için önerilir)

    Returns:
        (train_loader, val_loader) tuple'ı
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Her epoch farklı sıra
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,        # Son eksik batch'i at (BatchNorm için önemli)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader