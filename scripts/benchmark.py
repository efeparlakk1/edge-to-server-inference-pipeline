import time, os
import numpy as np
import onnxruntime as ort

MODELS = {
    "FP32": "outputs/converted/model.onnx",
    "INT8": "outputs/converted/model_int8.onnx",
}
N_RUNS   = 200
IMG_SIZE = 32


def benchmark(path: str) -> dict:
    sess  = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name  = sess.get_inputs()[0].name
    data  = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # Isınma
    sess.run(None, {name: data})

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        sess.run(None, {name: data})
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / N_RUNS * 1000
    return {
        "latency_ms": latency_ms,
        "throughput":  1000 / latency_ms,
        "size_mb":     os.path.getsize(path) / 1e6,
    }


def main():
    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  {name} bulunamadı: {path}")
            continue
        print(f"  {name} ölçülüyor...")
        results[name] = benchmark(path)

    print(f"\n{'Model':<8} {'Boyut (MB)':>12} {'Latency (ms)':>14} {'FPS':>10}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<8} {r['size_mb']:>12.1f} {r['latency_ms']:>14.2f} {r['throughput']:>10.1f}")

    if "FP32" in results and "INT8" in results:
        print(f"\nINT8 → {results['FP32']['size_mb']/results['INT8']['size_mb']:.1f}× küçük, "
              f"{results['FP32']['latency_ms']/results['INT8']['latency_ms']:.1f}× hızlı")

    os.makedirs("outputs/benchmarks", exist_ok=True)
    with open("outputs/benchmarks/results.md", "w") as f:
        f.write("## Benchmark Sonuçları\n\n")
        f.write("| Model | Boyut (MB) | Latency (ms) | FPS |\n")
        f.write("|-------|-----------|-------------|-----|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['size_mb']:.1f} | {r['latency_ms']:.2f} | {r['throughput']:.1f} |\n")
    print("\n✓ outputs/benchmarks/results.md kaydedildi")


if __name__ == "__main__":
    main()
