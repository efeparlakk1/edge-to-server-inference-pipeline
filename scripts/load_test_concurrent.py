"""
Concurrent yük testi — Dynamic Batching'i gerçekten tetikler.
Aynı anda N thread istek gönderir, GPU batch dolmaya başlar.
"""

import time
import numpy as np
import threading
import tritonclient.grpc as grpcclient

SERVER      = "localhost:8001"
MODEL       = "arcface_model"
IMG_SIZE    = 32
N_THREADS   = 32    # Aynı anda 32 paralel istek
N_REQUESTS  = 500   # Thread başına istek sayısı


def worker(results: list, idx: int):
    client = grpcclient.InferenceServerClient(url=SERVER)
    image  = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    inp    = grpcclient.InferInput("input", image.shape, "FP32")
    inp.set_data_from_numpy(image)
    out    = grpcclient.InferRequestedOutput("embedding")

    latencies = []
    for _ in range(N_REQUESTS):
        t0 = time.perf_counter()
        client.infer(model_name=MODEL, inputs=[inp], outputs=[out])
        latencies.append((time.perf_counter() - t0) * 1000)

    results[idx] = latencies


def main():
    results  = [None] * N_THREADS
    threads  = [threading.Thread(target=worker, args=(results, i))
                for i in range(N_THREADS)]

    print(f"  {N_THREADS} thread × {N_REQUESTS} istek = {N_THREADS*N_REQUESTS:,} toplam istek")

    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total_elapsed = time.perf_counter() - t0

    all_latencies = sorted([l for r in results for l in r])
    total_reqs    = len(all_latencies)

    print(f"\n── Concurrent Yük Testi ─────────────────")
    print(f"  Toplam istek  : {total_reqs:,}")
    print(f"  Toplam süre   : {total_elapsed:.2f}s")
    print(f"  Ortalama      : {np.mean(all_latencies):.2f} ms")
    print(f"  P50           : {all_latencies[int(total_reqs*0.50)]:.2f} ms")
    print(f"  P95           : {all_latencies[int(total_reqs*0.95)]:.2f} ms")
    print(f"  P99           : {all_latencies[int(total_reqs*0.99)]:.2f} ms")
    print(f"  Throughput    : {total_reqs/total_elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
