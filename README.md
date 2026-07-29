# Edge-to-Server Inference Pipeline

An end-to-end, production-ready machine learning pipeline featuring **ResNet-18 + ArcFace Metric Learning** training, **ONNX export**, **INT8 dynamic quantization**, and **Triton Inference Server** deployment with **Dynamic Batching** and automated GPU-aware configuration.

---

## 🎯 Purpose

Edge AI deployments and server-side inference engines require drastically different model representations and optimization trade-offs:

- **Edge Deployment**: Strict memory limits, minimal binary footprints, low power consumption.
- **Server Deployment**: High throughput, GPU batching, gRPC-based client communication, and Triton Inference Server interoperability.

This project bridges that gap by providing a fully reproducible workflow: train a metric learning model (ArcFace loss on ResNet-18), export it to ONNX, quantize it for edge deployment, and serve it at scale via Triton with Dynamic Batching.

---

## 📋 Project Scope

### Week 1 — Model Training, Export & Quantization

1. **Metric Learning Training (`train.py`)**
   - Backbone: **ResNet-18** producing 512-dimensional embeddings.
   - Loss: **ArcFace (Additive Angular Margin Loss)** via `pytorch-metric-learning`.
   - Dataset: **CIFAR-10** (60,000 images, 10 classes) with `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`.
   - Best validation accuracy: **87.35%** in 30 epochs on RTX 4070 Ti SUPER (~4.5s/epoch).
   - Logging: TensorBoard + checkpoint saving.

2. **ONNX Export (`scripts/convert_to_onnx.py`)**
   - Exports trained PyTorch backbone to **ONNX Opset 17**.
   - Dynamic batch dimension configured for flexible server batching.

3. **INT8 Quantization (`scripts/quantize_int8.py`)**
   - **Dynamic INT8 quantization** via ONNX Runtime.
   - **4× size reduction**: ~45.8 MB → ~11.5 MB.

4. **Benchmarking (`scripts/benchmark.py`)**
   - Evaluates latency (ms), throughput (FPS), and binary size across FP32 and INT8 ONNX models.

### Week 2 — Triton Inference Server & Production Serving

5. **Auto-Config Generator (`scripts/generate_triton_config.py`)**
   - Queries GPU VRAM and CPU core count via `nvidia-smi` at runtime.
   - Computes optimal `instance_count`, `max_batch_size`, `preferred_batch_size`, and `queue_delay` automatically.
   - Writes `triton_repo/arcface_model/config.pbtxt` — no manual tuning required.

6. **Triton Model Repository (`triton_repo/`)**
   - ONNX Runtime backend with GPU execution.
   - **Dynamic Batching**: queues concurrent requests and merges them into GPU-efficient batches.
   - **Multi-instance execution**: multiple model copies run in parallel on the same GPU.

7. **gRPC Client (`client.py`)**
   - Single-request inference and 500-request sequential load test.

8. **Concurrent Load Test (`scripts/load_test_concurrent.py`)**
   - 32 parallel threads × 500 requests = 16,000 total requests.
   - Reports P50 / P95 / P99 latency and throughput FPS.

---

## 📊 Benchmark Results

### Model Optimization (Week 1)

Benchmarks conducted with ONNX Runtime `CPUExecutionProvider`, 200 iterations:

| Model | Binary Size (MB) | Latency (ms) | Throughput (FPS) | Notes |
|-------|-----------------|-------------|-----------------|-------|
| **FP32** | 45.8 | 0.49 | 2,045 | Baseline |
| **INT8** | 11.5 | 1.61 | 619 | **4× size reduction** |

> **Why INT8 is slower on CPU:** Dynamic quantization quantizes weights to INT8 but dequantizes them back to FP32 at runtime on CPUs without dedicated INT8 SIMD acceleration. Size benefit is real; speed benefit requires static quantization or INT8-capable hardware (NPU/TPU).

### Triton Inference Server (Week 2)

Concurrent load test: 32 threads × 500 requests on RTX 4070 Ti SUPER:

| Metric | Value |
|--------|-------|
| Total requests | 16,000 |
| Total time | 25.71s |
| Average batch size | ~9.2 |
| **P50 latency** | **7.02 ms** |
| P95 latency | 287 ms |
| P99 latency | 798 ms |
| **Throughput** | **622 FPS** |

> **P99 note:** High P99 under 32-thread burst load is expected — concurrent threads create artificial queue spikes absent in real production traffic. P50 at 7ms reflects true steady-state latency.

### Auto-Config Output (RTX 4070 Ti SUPER, 16GB)

```
VRAM total      : 16.0 GB
Instance count  : 4
Max batch size  : 128
Preferred sizes : [32, 64, 128]
Queue delay     : 2000 µs
Op threads      : 6
```

---

## 🏗️ Project Architecture

```
edge-to-server-inference-pipeline/
├── src/
│   ├── data/dataset.py           # CIFAR-10 DataLoader + augmentation
│   ├── losses/arcface.py         # ArcFace loss wrapper
│   └── models/resnet_arcface.py  # ResNet-18 + embedding head
├── scripts/
│   ├── verify_setup.py           # Environment verification
│   ├── convert_to_onnx.py        # PyTorch → ONNX
│   ├── quantize_int8.py          # ONNX FP32 → INT8
│   ├── benchmark.py              # Latency + FPS benchmark
│   ├── generate_triton_config.py # GPU-aware Triton config generator
│   └── load_test_concurrent.py   # Concurrent gRPC load test
├── triton_repo/
│   └── arcface_model/
│       ├── config.pbtxt          # Triton model config (auto-generated)
│       └── 1/
│           └── model.onnx        # ONNX model weights
├── outputs/
│   ├── checkpoints/              # best_model.pth
│   ├── converted/                # model.onnx, model_int8.onnx
│   ├── logs/                     # TensorBoard events
│   └── benchmarks/               # results.md
├── client.py                     # gRPC inference client
├── train.py                      # Main training loop
└── pyproject.toml                # uv dependencies
```

---

## ⚡ Quick Start

### Requirements

- Python 3.11+
- CUDA 12.x + cuDNN
- Docker + NVIDIA Container Toolkit
- [`uv`](https://github.com/astral-sh/uv)

### Installation

```bash
git clone https://github.com/efeparlakk1/edge-to-server-inference-pipeline.git
cd edge-to-server-inference-pipeline
uv sync
```

### Week 1: Train → Export → Quantize

```bash
# 1. Verify environment
uv run python scripts/verify_setup.py

# 2. Train (30 epochs, ~4.5s/epoch on RTX 4070 Ti SUPER)
uv run python train.py --epochs 30 --batch-size 256

# 3. Export to ONNX
uv run python scripts/convert_to_onnx.py

# 4. INT8 quantization
uv run python scripts/quantize_int8.py

# 5. Benchmark FP32 vs INT8
uv run python scripts/benchmark.py
```

### Week 2: Triton Serving

```bash
# 1. Auto-generate Triton config for your GPU
uv run python scripts/generate_triton_config.py

# 2. Copy ONNX model to Triton repo
mkdir -p triton_repo/arcface_model/1
cp outputs/converted/model.onnx triton_repo/arcface_model/1/model.onnx

# 3. Start Triton (Docker)
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# 4. Single inference test
uv run python client.py

# 5. Concurrent load test (32 threads × 500 requests)
uv run python scripts/load_test_concurrent.py
```

---

## 🔑 Key Concepts

| Concept | What it means in this project |
|---------|-------------------------------|
| **ArcFace loss** | Adds angular margin to cosine similarity — tighter clusters, better separation than Cross-Entropy |
| **ONNX Opset 17** | Framework-agnostic model graph — runs on Triton, TensorRT, CoreML without retraining |
| **Dynamic quantization** | Weights → INT8 at export time; activations remain FP32 at runtime |
| **Dynamic Batching** | Triton merges concurrent requests into single GPU batch — maximizes utilization |
| **Multi-instance execution** | Multiple model copies on same GPU — reduces queuing under high concurrency |
| **gRPC vs HTTP** | gRPC uses binary Protocol Buffers; ~3–5× lower latency than HTTP/JSON for inference |
| **P50 / P95 / P99** | Percentile latency — P99 reveals tail latency spikes invisible in averages |
| **Auto-config** | `generate_triton_config.py` queries `nvidia-smi` and computes instance count, batch sizes, queue delay for any GPU |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Training | PyTorch, Torchvision, pytorch-metric-learning |
| Export | ONNX (Opset 17), ONNX Runtime |
| Quantization | onnxruntime.quantization |
| Serving | Triton Inference Server 24.01, ONNX Runtime backend |
| Client | tritonclient[grpc] |
| Environment | uv, Python 3.11, CUDA 12.x |
| GPU | NVIDIA RTX 4070 Ti SUPER (16GB VRAM) |
