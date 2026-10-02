# LLM Architectures Benchmark & Optimization Suite

A high-performance PyTorch benchmark and experimentation codebase analyzing Large Language Model (LLM) attention architectures, training dynamics, `torch.compile` (TorchInductor) optimizations, Key-Value (KV) caching, and CUDA Graph execution strategies.

This repository compares:
* **Attention Architectures**: Multi-Head Attention (**MHA**), Grouped-Query Attention (**GQA-2**, **GQA-4**), and Multi-Query Attention (**MQA**).
* **Residual Topologies**: Sequential Residual vs. Parallel Residual Blocks.
* **Execution Modes**: PyTorch Eager Mode vs. `torch.compile` (`default`, `reduce-overhead`, `max-autotune`).
* **Decoding Mechanics**: Un-bucketed dynamic sequence decoding vs. **Bucket-Shifting Decoding** ($200 \to 256 \to 512 \to 1024 \to 2048 \to 4096$).

---

## ⚡ 1. Training & Architecture Variations (TinyStories 52.76M Params)

* **Dataset:** TinyStories ($5,000$ samples, sequence length $L = 256$, batch size $B = 16$)
* **Model Configuration:** 6 Layers, $d_{\text{model}} = 384$, $H_q = 6$ Heads, Vocab Size = 50,257
* **Training Pipeline:** FP16 Automatic Mixed Precision (`torch.amp`), AdamW ($\text{LR} = 5 \times 10^{-4}$), Cosine LR Schedule with Warmup, Gradient Clipping ($1.0$), 500 Steps.

### PyTorch Eager Mode Execution

| Configuration | Parameters | Val Loss | Val PPL | Throughput (tok/s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (MHA + Sequential)** | $52.76\text{M}$ | **3.1644** | **23.67** | $50,178.9$ | $4.157\text{ GB}$ |
| **2. Variant A (Parallel Residual)** | $52.76\text{M}$ | $3.1742$ | $23.91$ | $49,965.4$ | $4.124\text{ GB}$ |
| **3. Variant B (GQA: 2 KV Heads)** | $51.58\text{M}$ | $3.1966$ | $24.45$ | $51,831.6$ | $4.143\text{ GB}$ |
| **4. Variant C (GQA + Parallel)** | $51.58\text{M}$ | $3.1680$ | $23.76$ | **51,925.7** | **4.109 GB** |

### Compiled Mode Execution (`torch.compile`)

| Configuration | Parameters | Val Loss | Val PPL | Throughput (tok/s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (MHA + Sequential)** | $52.76\text{M}$ | $3.1573$ | $23.51$ | $62,402.5$ | **3.937 GB** |
| **2. Variant A (Parallel Residual)** | $52.76\text{M}$ | **3.1516** | **23.37** | $63,035.5$ | $4.336\text{ GB}$ |
| **3. Variant B (GQA: 2 KV Heads)** | $51.58\text{M}$ | $3.1726$ | $23.87$ | $63,316.2$ | $4.725\text{ GB}$ |
| **4. Variant C (GQA + Parallel)** | $51.58\text{M}$ | $3.1914$ | $24.32$ | **64,374.2** | $5.115\text{ GB}$ |

### Eager vs. Compiled Throughput Comparison

```text
Throughput (Tokens / Second)
========================================================================================
1. Baseline (Eager)    [50,178.9] █████████████████████████
1. Baseline (Compiled) [62,402.5] ███████████████████████████████ (+24.4%)
----------------------------------------------------------------------------------------
4. Variant C (Eager)   [51,925.7] ██████████████████████████
4. Variant C (Compiled)[64,374.2] ████████████████████████████████ (+24.0%)
========================================================================================
```

---

## 🧠 Systems & Compiler Analysis

### 1. The Eager Mode Fallacy & Memory Reclamation
In standard PyTorch Eager Mode, Parallel Residual blocks do not achieve concurrent GPU execution. Operations are queued sequentially on the default CUDA stream (`Stream 0`). Furthermore, parallel branching forces PyTorch to store both Attention and MLP outputs in High-Bandwidth Memory (HBM) simultaneously for the 3-way addition ($x + a + m$), creating extra HBM read/write roundtrips that reduce throughput ($49,965 \text{ vs. } 50,178 \text{ tok/s}$).

However, Eager Mode benefits from dynamic activation freeing: because parallel paths branch from $\text{RMSNorm}(x)$ simultaneously, activation lifetimes are shortened compared to sequential dependencies, allowing Variant C (GQA + Parallel) to achieve the lowest Eager VRAM footprint ($4.109 \text{ GB}$).

### 2. Kernel Fusion via TorchInductor
`torch.compile` allows TorchInductor to generate fused Triton kernels that execute Parallel Attention and MLP additions in single CUDA passes. Intermediate activations stay inside high-speed GPU SRAM registers ($19 \text{ TB/s}$) rather than flushing to main HBM ($2\text{--}3 \text{ TB/s}$), unlocking parallel execution and driving throughput up to **$63,035 \text{ tok/s}$**.

### 3. Static Workspace Memory vs. Graph Complexity
While `torch.compile` accelerates execution by up to $24\%$, it reverses the VRAM hierarchy between models:
* **Baseline (Sequential):** The non-branching, linear graph enables Inductor to aggressively reuse global scratchpad buffers, lowering VRAM to **$3.937 \text{ GB}$**.
* **Complex Variants (GQA / Parallel):** To fuse multi-branch additions ($x + a + m$) and handle GQA key/value broadcasting without GPU register spilling, TorchInductor pre-allocates persistent, static memory workspace buffers. Complex graph topologies add static allocation pools ($\approx 0.39 \text{ GB}$ per structural feature), raising compiled peak VRAM for Variant C up to $5.115 \text{ GB}$.

---

## 📊 2. Decoding & GPU Profiling Benchmarks

All benchmarks were run on an **NVIDIA GeForce RTX 2080 Ti** (Peak Theoretical Bandwidth: $616.0\text{ GB/s}$, $11\text{ GB}$ VRAM, PyTorch `2.10.0+cu128`, FP16 precision).

### 1. Short Sequence Generation Benchmark (128 Tokens)
*Config: Batch Size = 16, Context Length = 2048 $\to$ 2176 tokens*

| Architecture | KV Heads ($H_{\text{kv}}$) | KV Cache Size | Peak GPU VRAM | Total Time | Step Latency | Throughput | Achieved Bandwidth | MBU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MHA (Multi-Head Attention)** | 16 | 4,352.00 MB | 6,694.40 MB | **2.00 s** 🚀 | **15.65 ms** 🚀 | **1,022.46 tok/s** 🚀 | **437.17 GB/s** | **70.97 %** |
| **MQA (Multi-Query Attention)** | 1 | **272.00 MB** 🚀 | **2,761.25 MB** 🚀 | 5.25 s | 40.98 ms | 390.45 tok/s | 59.42 GB/s | 9.65 % |
| **GQA-2 (Grouped-Query Attention)** | 2 | 544.00 MB | 3,323.25 MB | 5.26 s | 41.11 ms | 389.18 tok/s | 66.37 GB/s | 10.77 % |
| **GQA-4 (Grouped-Query Attention)** | 4 | 1,088.00 MB | 6,631.25 MB | 5.60 s | 43.78 ms | 365.42 tok/s | 75.74 GB/s | 12.30 % |

---

### 2. Long Sequence Generation Benchmark (1024 Tokens)
*Config: Batch Size = 4, Context Length = 2048 $\to$ 3072 tokens*

| Architecture | KV Heads ($H_{\text{kv}}$) | KV Cache Size | Peak GPU VRAM | Total Time | Step Latency | Throughput | Achieved Bandwidth | MBU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MHA (Multi-Head Attention)** | 16 | 1,536.00 MB | 3,896.40 MB | **14.03 s** 🚀 | **13.70 ms** 🚀 | **292.00 tok/s** 🚀 | **273.93 GB/s** | **44.47 %** |
| **MQA (Multi-Query Attention)** | 1 | **96.00 MB** 🚀 | **2,408.52 MB** 🚀 | 16.90 s | 16.51 ms | 242.33 tok/s | 135.83 GB/s | 22.05 % |
| **GQA-2 (Grouped-Query Attention)** | 2 | 192.00 MB | 2,618.52 MB | 17.70 s | 17.28 ms | 231.48 tok/s | 135.57 GB/s | 22.01 % |
| **GQA-4 (Grouped-Query Attention)** | 4 | 384.00 MB | 3,814.52 MB | 17.98 s | 17.56 ms | 227.83 tok/s | 144.91 GB/s | 23.52 % |

---

### 3. Extreme Long Sequence Generation Benchmark (2048 Tokens)
*Config: Batch Size = 8, Context Length = 2048 $\to$ 4096 tokens*

| Architecture | KV Heads ($H_{\text{kv}}$) | KV Cache Size | Peak GPU VRAM | Total Time | Step Latency | Throughput | Achieved Bandwidth | MBU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MHA (Multi-Head Attention)** | 16 | 4,096.00 MB | 6,456.44 MB | **28.31 s** 🚀 | **13.82 ms** 🚀 | **578.80 tok/s** 🚀 | **407.48 GB/s** | **66.15 %** |
| **MQA (Multi-Query Attention)** | 1 | **256.00 MB** 🚀 | **2,728.76 MB** 🚀 | 66.31 s | 32.38 ms | 247.08 tok/s | 72.87 GB/s | 11.83 % |
| **GQA-2 (Grouped-Query Attention)** | 2 | 512.00 MB | 3,258.76 MB | 67.12 s | 32.77 ms | 244.10 tok/s | 78.65 GB/s | 12.77 % |
| **GQA-4 (Grouped-Query Attention)** | 4 | 1,024.00 MB | 6,374.76 MB | 66.89 s | 32.66 ms | 244.94 tok/s | 92.28 GB/s | 14.98 % |

---

### 4. Bucket-Shifting Decoding Benchmark (Dynamic Bucket Expansion: Prompt 200 $\to$ 1,900 New Tokens = 2,100 Total Tokens)
*Config: Batch Size = 4, Prompt Length = 200 tokens, Generation = 1,900 new tokens, Buckets = [256, 512, 1024, 2048, 4096]*

| Architecture | KV Heads ($H_{\text{kv}}$) | Total Decode Time (1900 Tokens) | Step Latency | Generation Throughput | Peak GPU VRAM | Theoretical Rank | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MQA (Multi-Query Attention)** | 1 | **27.08 s** 🚀 | **14.25 ms** 🚀 | **280.67 tok/s** 🚀 | **2,333.10 MB** 🚀 | **1st Place** | **PASSED** ✅ |
| **GQA-2 (Grouped-Query Attention)** | 2 | 27.26 s | 14.34 ms | 278.84 tok/s | 2,549.10 MB | 2nd Place | **PASSED** ✅ |
| **GQA-4 (Grouped-Query Attention)** | 4 | 28.52 s | 15.01 ms | 266.48 tok/s | 2,981.10 MB | 3rd Place | **PASSED** ✅ |
| **MHA (Multi-Head Attention)** | 16 | 27.08 s | 14.25 ms | 280.62 tok/s | 5,541.60 MB | 4th Place | **PASSED** ✅ |

---

## 🛠️ Tracing & Deep Compiler Analysis (`compile_trace_study.py`)

To inspect PyTorch Inductor internals, CUDA Graph invalidations, and Triton autotuning, run:

```bash
# Execute trace generation
python compile_trace_study.py

# Inspect graph breaks, recompiles, and CUDA Graph events
TORCH_LOGS="graph_breaks,recompiles,cudagraphs" python compile_trace_study.py

# Generate visual HTML / Perfetto traces
TORCH_TRACE="./torch_trace_html" python compile_trace_study.py
```

### Visual Trace Viewer
Open [torch_trace_html/index.html](file:///home/hussin/llm_archs_benchmark/torch_trace_html/index.html) in your browser or drop [torch_trace_html/trace.json](file:///home/hussin/llm_archs_benchmark/torch_trace_html/trace.json) into `chrome://tracing` / [ui.perfetto.dev](https://ui.perfetto.dev).

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── models/
│   │   ├── layers.py         # RMSNorm, SwiGLU, RoPE (with start_pos offset), GQA
│   │   └── transformer.py    # Transformer Block, Decoder LLM & .generate() with KV cache
│   └── data.py               # Streaming dataset & block tokenizer
├── checkpoints/              # Model weights per experiment
├── results/                  # Metric JSON files & summary reports
├── train.py                  # Core training & evaluation script
├── run_matrix.py             # Eager Mode training benchmark harness
├── run_matrix_with_compile.py # Torch.compile training benchmark harness
├── gpu_profile_kv.py         # Long-sequence KV Cache & Bucket-Shifting Decoding profiler
├── compile_trace_study.py    # PyTorch Dynamo / Inductor compiler tracing study script
├── torch_trace_html/         # Visual Perfetto / Chrome trace HTML & JSON files
└── README.md
```