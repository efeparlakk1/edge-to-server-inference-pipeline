"""
GPU VRAM'ine göre otomatik Triton config üretici.

Çalıştır:
    uv run python scripts/generate_triton_config.py

Başkası projeyi klonladığında kendi GPU'suna göre
otomatik optimize edilmiş config alır.
"""

import subprocess
import sys
import os


def get_vram_gb() -> float:
    """nvidia-smi ile toplam VRAM'i GB cinsinden döndür."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0]
        return int(out) / 1024
    except Exception:
        print("nvidia-smi bulunamadı — CPU modu varsayıldı (4GB)")
        return 4.0


def get_cpu_cores() -> int:
    """Mantıksal CPU çekirdek sayısını döndür."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def compute_config(vram_gb: float, cpu_cores: int) -> dict:
    """
    VRAM ve CPU'ya göre Triton parametrelerini hesapla.

    Mantık:
      - instance_count : Her instance ~1.5GB VRAM tutar (ResNet18+runtime overhead).
                         VRAM'in %60'ını kullan, kalanı sistem için bırak.
      - max_batch_size : Instance başına 32 istek. GPU'yu doldurmak için yeterli.
      - preferred_batch : max_batch'in %25, %50, %100'ü — kademeli doldurma.
      - queue_delay     : Düşük VRAM = az instance = daha agresif batching gerekir.
      - op_threads      : CPU core sayısının yarısı — Triton kendi işleri için yarısına ihtiyaç duyar.
    """
    model_vram_gb     = 1.5
    usable_vram       = vram_gb * 0.60
    instance_count    = max(1, int(usable_vram / model_vram_gb))
    max_batch_size    = instance_count * 32
    max_batch_size    = min(max_batch_size, 256)  # Triton hard limiti

    preferred_batches = sorted(set([
        max(1, max_batch_size // 4),
        max(1, max_batch_size // 2),
        max_batch_size,
    ]))

    # Az instance → daha uzun bekle, batch dolsun
    # Çok instance → hızlı gönder, kuyruk kısa kalır
    queue_delay_us = max(1000, 10000 // instance_count)

    op_threads = max(1, cpu_cores // 2)

    return {
        "instance_count":    instance_count,
        "max_batch_size":    max_batch_size,
        "preferred_batches": preferred_batches,
        "queue_delay_us":    queue_delay_us,
        "op_threads":        op_threads,
    }


def render_config(cfg: dict) -> str:
    preferred_str = ", ".join(str(b) for b in cfg["preferred_batches"])

    return f"""\
name: "arcface_model"
backend: "onnxruntime"
max_batch_size: {cfg["max_batch_size"]}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 32, 32 ]
  }}
]

output [
  {{
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_str} ]
  max_queue_delay_microseconds: {cfg["queue_delay_us"]}
}}

instance_group [
  {{
    count: {cfg["instance_count"]}
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

parameters {{
  key: "intra_op_thread_count"
  value: {{ string_value: "{cfg["op_threads"]}" }}
}}

parameters {{
  key: "inter_op_thread_count"
  value: {{ string_value: "{cfg["op_threads"]}" }}
}}

parameters {{
  key: "execution_mode"
  value: {{ string_value: "1" }}
}}
"""


def main():
    vram_gb   = get_vram_gb()
    cpu_cores = get_cpu_cores()
    cfg       = compute_config(vram_gb, cpu_cores)

    print(f"\n── Sistem ───────────────────────────────")
    print(f"  VRAM       : {vram_gb:.1f} GB")
    print(f"  CPU cores  : {cpu_cores}")
    print(f"\n── Hesaplanan Parametreler ──────────────")
    print(f"  instance_count    : {cfg['instance_count']}")
    print(f"  max_batch_size    : {cfg['max_batch_size']}")
    print(f"  preferred_batches : {cfg['preferred_batches']}")
    print(f"  queue_delay       : {cfg['queue_delay_us']} µs")
    print(f"  op_threads        : {cfg['op_threads']}")

    out_path = "triton_repo/arcface_model/config.pbtxt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write(render_config(cfg))

    print(f"\n✓ Config yazıldı: {out_path}")
    print(f"\nTriton'u başlatmak için:")
    print(f"  docker run --gpus all --rm \\")
    print(f"    -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print(f"    -v $(pwd)/triton_repo:/models \\")
    print(f"    nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print(f"    tritonserver --model-repository=/models")


if __name__ == "__main__":
    main()