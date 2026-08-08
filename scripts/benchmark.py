import time, os
import numpy as np
import onnxruntime as ort

MODELS = {
    "FP32": "outputs/converted/model.onnx",
    "INT8": "outputs/converted/model_int8.onnx",
}
N_RUNS   = 200
WARMUP   = 10
IMG_SIZE = 32


def benchmark(path: str) -> dict:
    sess  = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name  = sess.get_inputs()[0].name
    data  = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # 10 adım ısınma (warm-up)
    for _ in range(WARMUP):
        sess.run(None, {name: data})

    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {name: data})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    mean_lat  = float(np.mean(latencies))
    return {
        "latency_ms": mean_lat,
        "p50_ms":     latencies[int(N_RUNS * 0.50)],
        "p95_ms":     latencies[int(N_RUNS * 0.95)],
        "p99_ms":     latencies[int(N_RUNS * 0.99)],
        "throughput": 1000 / mean_lat,
        "size_mb":    os.path.getsize(path) / 1e6,
    }


def main():
    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  {name} bulunamadı: {path}")
            continue
        print(f"  {name} ölçülüyor...")
        results[name] = benchmark(path)

    print(f"\n{'Model':<8} {'Boyut (MB)':>12} {'Mean (ms)':>12} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'FPS':>8}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<8} {r['size_mb']:>12.1f} {r['latency_ms']:>12.2f} {r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['throughput']:>8.1f}")

    if "FP32" in results and "INT8" in results:
        size_ratio = results['FP32']['size_mb'] / results['INT8']['size_mb']
        lat_ratio  = results['FP32']['latency_ms'] / results['INT8']['latency_ms']
        if lat_ratio >= 1.0:
            speed_str = f"{lat_ratio:.2f}× daha hızlı"
        else:
            speed_str = f"{(1 / lat_ratio):.2f}× daha yavaş (overhead dolayısıyla)"

        print(f"\nINT8 → {size_ratio:.1f}× daha küçük, CPU ortamında FP32'ye göre {speed_str}")

    os.makedirs("outputs/benchmarks", exist_ok=True)
    with open("outputs/benchmarks/results.md", "w") as f:
        f.write("## Benchmark Sonuçları\n\n")
        f.write("| Model | Boyut (MB) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS |\n")
        f.write("|-------|-----------|-----------|----------|----------|----------|-----|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['size_mb']:.1f} | {r['latency_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['throughput']:.1f} |\n")
    print("\n✓ outputs/benchmarks/results.md kaydedildi")


if __name__ == "__main__":
    main()
