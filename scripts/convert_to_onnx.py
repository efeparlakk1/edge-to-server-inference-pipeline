"""
Adım 1: PyTorch checkpoint → ONNX

Çalıştır:
    uv run python scripts/convert_to_onnx.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.resnet_arcface import build_model

CHECKPOINT = "outputs/checkpoints/best_model.pth"
ONNX_OUT   = "outputs/converted/model.onnx"
IMG_SIZE   = 32   # CIFAR-10

def main():
    os.makedirs("outputs/converted", exist_ok=True)
    device = torch.device("cpu")  # Export CPU'dan yapılır — cihaz bağımsız format üretir

    # Modeli yükle
    model = build_model(embedding_dim=512, pretrained=False).to(device)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Temsili girdi — şekil önemli, değerler değil
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        opset_version=17,           # Geniş operatör desteği için 17+
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={              # Batch boyutunu dinamik bırak
            "input":     {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(ONNX_OUT) / 1e6
    print(f"✓ ONNX kaydedildi: {ONNX_OUT}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()