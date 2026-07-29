# Edge-to-Server Inference Pipeline

An end-to-end, production-ready machine learning pipeline featuring **ResNet-18 + ArcFace Metric Learning** training, **ONNX export**, **INT8 dynamic quantization**, **LiteRT (TFLite) conversion**, and automated performance benchmarking.

---

## 🎯 Purpose

Edge AI deployments and server-side inference engines often require drastically different model representations and optimization trade-offs:
- **Server Deployment**: High throughput, GPU batching, and ONNX Runtime / Triton Inference Server interoperability.
- **Edge Deployment**: Strict memory limits, minimal binary footprints, low power consumption, and LiteRT (TFLite) runtime compatibility.

This project bridges that gap by providing an end-to-end, reproducible workflow that trains a feature embedding model (ArcFace loss on ResNet-18), exports it to open interoperable formats (ONNX), quantizes it for memory-constrained edge deployment, and evaluates runtime throughput and latency across model variants.

---

## 📋 Project Scope

The pipeline covers the complete lifecycle of metric learning model development and deployment:

1. **Metric Learning Training (`train.py`)**
   - Backbone: **ResNet-18** feature extractor producing 512-dimensional embeddings.
   - Loss Function: **ArcFace (Additive Angular Margin Loss)** via `pytorch-metric-learning` for deep feature discrimination.
   - Dataset: **CIFAR-10** (60,000 images, 10 classes) with data augmentation (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`).
   - Logging & Checkpointing: Validation accuracy tracking and TensorBoard visualization.

2. **ONNX Export (`scripts/convert_to_onnx.py`)**
   - Exports the trained PyTorch backbone model to **ONNX Opset 17**.
   - Configures dynamic batch dimensions (`batch_size`) for flexible server batching.

3. **INT8 Quantization (`scripts/quantize_int8.py`)**
   - Applies **dynamic INT8 quantization** using ONNX Runtime.
   - Reduces model storage size by **4×** (from ~45.8 MB to ~11.5 MB).

4. **LiteRT / TFLite Conversion (`scripts/convert_to_litert.py`)**
   - Converts the ONNX model to **TensorFlow SavedModel** and **LiteRT (`.tflite`)** format using `onnx2tf` for edge mobile and embedded runtimes.

5. **Benchmarking & Evaluation (`scripts/benchmark.py`)**
   - Evaluates CPU inference latency (ms), throughput (FPS), and binary size (MB) across FP32 and INT8 ONNX models using ONNX Runtime.

---

## 📊 Benchmark Results

Benchmarks were conducted using ONNX Runtime (`CPUExecutionProvider`) over 200 iterations:

| Model | Binary Size (MB) | Latency (ms) | Throughput (FPS) | Optimization Impact |
|---|---|---|---|---|
| **FP32** | 45.8 MB | 0.49 ms | 2045.6 FPS | Baseline full precision model |
| **INT8** | 11.5 MB | 1.61 ms | 619.4 FPS | **4.0× size reduction** |

> **Key Takeaways & Quantization Insights:**
> - **Storage Advantage**: INT8 dynamic quantization yields a **4× reduction in binary size** (~45.8 MB → ~11.5 MB), which is critical for edge deployments, mobile app bundle limits, and over-the-air (OTA) model updates.
> - **Execution Speed**: Dynamic quantization quantizes model weights to INT8 while keeping activations in FP32, introducing runtime dequantization/quantization overhead on general-purpose CPUs without dedicated INT8 SIMD acceleration.
> - **Edge Recommendation**: For maximum execution speedups alongside size reduction on edge hardware, static quantization (quantizing both weights and activations with calibration data) or hardware delegate acceleration (e.g., NPU/TPU via LiteRT) should be utilized.

---

## 🏗️ Project Architecture

```
edge-to-server-inference-pipeline/
├── configs/                  # Pipeline configurations
├── data/                     # CIFAR-10 dataset storage
├── outputs/                  # Exported models, checkpoints, logs & benchmark reports
│   ├── benchmarks/           # Benchmark result markdown files
│   ├── checkpoints/          # PyTorch model checkpoints (best_model.pth)
│   ├── converted/            # Exported ONNX, LiteRT (.tflite), and INT8 models
│   └── logs/                 # TensorBoard event files
├── scripts/                  # Processing, conversion, and benchmark scripts
│   ├── verify_setup.py       # Quick pipeline & hardware verification
│   ├── convert_to_onnx.py    # PyTorch → ONNX converter
│   ├── quantize_int8.py      # ONNX FP32 → INT8 dynamic quantizer
│   ├── convert_to_litert.py  # ONNX → LiteRT (.tflite) converter
│   └── benchmark.py          # Latency, FPS, and model size benchmark runner
├── src/                      # Core module implementations
│   ├── data/                 # Data loading and augmentation routines
│   ├── losses/               # ArcFace metric learning loss setup
│   ├── models/               # ResNet-18 feature extractor architecture
│   └── utils/                # Utility functions
├── train.py                  # Main PyTorch training loop
├── pyproject.toml            # Project dependencies and environment specification
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start

### 1. Requirements & Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# Clone the repository
git clone https://github.com/efeparlakk1/edge-to-server-inference-pipeline.git
cd edge-to-server-inference-pipeline

# Sync environment dependencies
uv sync
```

### 2. Verify Setup

Run the setup verification script to confirm CUDA/CPU environment, PyTorch configuration, and dataset loading:

```bash
uv run python scripts/verify_setup.py
```

### 3. Pipeline Workflow

#### Step A: Train Model with ArcFace Loss
Train ResNet-18 backbone with ArcFace loss on CIFAR-10:
```bash
uv run python train.py --epochs 30 --batch-size 256
```

#### Step B: Export to ONNX
Export PyTorch checkpoint to ONNX format (`outputs/converted/model.onnx`):
```bash
uv run python scripts/convert_to_onnx.py
```

#### Step C: Quantize to INT8
Apply dynamic INT8 quantization (`outputs/converted/model_int8.onnx`):
```bash
uv run python scripts/quantize_int8.py
```

#### Step D: Convert to LiteRT (TFLite)
Convert ONNX model to LiteRT format (`outputs/converted/model.tflite`):
```bash
uv run python scripts/convert_to_litert.py
```

#### Step E: Run Benchmarks
Run throughput and latency benchmark across FP32 and INT8 models:
```bash
uv run python scripts/benchmark.py
```

---

## 🛠️ Tech Stack & Dependencies

- **Deep Learning Framework**: PyTorch, Torchvision, `pytorch-metric-learning`
- **Interoperability & Deployment**: ONNX, ONNX Runtime, `onnx2tf`, `ai-edge-litert`
- **Environment & Dependency Management**: `uv`, Python 3.11+