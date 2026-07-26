## Benchmark Sonuçları

| Model | Boyut (MB) | Latency (ms) | FPS    |
|-------|-----------|-------------|--------|
| FP32  | 45.8      | 0.49        | 2045.6 |
| INT8  | 11.5      | 1.61        | 619.4  |

> **Not:** Dynamic quantization yalnızca ağırlıkları INT8'e indirger.
> Hız kazanımı için static quantization gerekir (aktivasyonlar da INT8).
> Boyut avantajı (4×) edge deployment için kritiktir.