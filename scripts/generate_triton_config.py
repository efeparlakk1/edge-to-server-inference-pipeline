"""
GPU VRAM'ine göre otomatik Triton config üretici.

Çalıştır:
    uv run python scripts/generate_triton_config.py
"""

import subprocess
import os


def get_gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0].split(",")
        return {
            "total_mb": int(out[0].strip()),
            "free_mb":  int(out[1].strip()),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError) as err:
        print(f"nvidia-smi çağrısı yapılamadı ({type(err).__name__}) — varsayılan değerler kullanılıyor")
        return {"total_mb": 4096, "free_mb": 4096}


def get_cpu_cores() -> int:
    return os.cpu_count() or 4


def compute_config(gpu: dict, cpu_cores: int) -> dict:
    """
    Parametre mantığı:
      - INSTANCE_VRAM_MB : Gerçek ölçüm (3778MB / 6 instance = ~630MB)
      - VRAM'in %70'ini modellere ayır, %30 aktivasyon+buffer
      - Usable MB hesabında yalnız boş (free) VRAM kullanılır (OOM riski önlenir)
      - instance üst sınırı 4 — daha fazlası context switching overhead yaratır
      - Batch: instance az → büyük batch; çok → küçük batch
      - Queue delay: instance az → uzun bekle, batch dolsun
    """
    TRITON_OVERHEAD_MB = 800
    INSTANCE_VRAM_MB   = 650
    VRAM_UTILIZATION   = 0.70

    # Boş VRAM baz alınır (diğer GPU süreçleri dikkate alınır)
    available_vram = max(0, gpu["free_mb"] - TRITON_OVERHEAD_MB)
    usable_mb      = available_vram * VRAM_UTILIZATION
    instance_count = max(1, min(int(usable_mb / INSTANCE_VRAM_MB), 4))

    batch_per_instance = 64 if instance_count <= 2 else 32
    max_batch_size     = min(instance_count * batch_per_instance, 256)

    preferred_batches = sorted(set([
        max(1, max_batch_size // 4),
        max(1, max_batch_size // 2),
        max_batch_size,
    ]))

    queue_delay_us = 5000 if instance_count <= 2 else 2000
    op_threads     = max(1, min(cpu_cores // 2, 8))

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
    gpu       = get_gpu_info()
    cpu_cores = get_cpu_cores()
    cfg       = compute_config(gpu, cpu_cores)

    print(f"\n── GPU ──────────────────────────────────")
    print(f"  VRAM toplam   : {gpu['total_mb']/1024:.1f} GB")
    print(f"  VRAM boş      : {gpu['free_mb']/1024:.1f} GB")
    print(f"  CPU cores     : {cpu_cores}")
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


if __name__ == "__main__":
    main()
