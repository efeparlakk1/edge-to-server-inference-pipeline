"""
Triton gRPC client — tekil ve yük testi.

Çalıştır:
    uv run python client.py
"""

import time
import numpy as np
import tritonclient.grpc as grpcclient

SERVER   = "localhost:8001"
MODEL    = "arcface_model"
IMG_SIZE = 32


def single_request(client) -> np.ndarray:
    """Tek istek gönder, embedding al."""
    image = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    inp = grpcclient.InferInput("input", image.shape, "FP32")
    inp.set_data_from_numpy(image)

    out    = grpcclient.InferRequestedOutput("embedding")
    result = client.infer(model_name=MODEL, inputs=[inp], outputs=[out])

    return result.as_numpy("embedding")


def load_test(client, n_requests: int = 500):
    """
    Ardışık istekler göndererek Dynamic Batching'i test et.
    Gerçek yük testinde concurrent thread'ler kullanılır,
    bu versiyon latency ölçümü için yeterli.
    """
    latencies = []

    for _ in range(n_requests):
        t0 = time.perf_counter()
        single_request(client)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    print(f"\n── Yük Testi ({n_requests} istek) ──────────────")
    print(f"  Ortalama  : {np.mean(latencies):.2f} ms")
    print(f"  P50       : {latencies[int(n_requests*0.50)]:.2f} ms")
    print(f"  P95       : {latencies[int(n_requests*0.95)]:.2f} ms")
    print(f"  P99       : {latencies[int(n_requests*0.99)]:.2f} ms")
    print(f"  Throughput: {1000/np.mean(latencies):.1f} FPS")


def main():
    client = grpcclient.InferenceServerClient(url=SERVER)

    # Sunucu hazır mı?
    assert client.is_server_live(),  "Triton sunucusu çalışmıyor"
    assert client.is_model_ready(MODEL), f"Model '{MODEL}' yüklenmemiş"
    print(f"✓ Triton bağlantısı kuruldu: {SERVER}")

    # Tekil test
    emb = single_request(client)
    print(f"✓ Embedding boyutu: {emb.shape}")  # (1, 512)

    # Yük testi
    load_test(client, n_requests=500)


if __name__ == "__main__":
    main()
