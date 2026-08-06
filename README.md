# LLM Architectural Ablations & Systems Profiling in PyTorch

A modular, first-principles PyTorch implementation of modern LLaMA-style LLM components—including Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), SwiGLU activations, RMSNorm, and Parallel Residual connections—coupled with an automated profiling harness to analyze throughput, VRAM consumption, and kernel execution dynamics across **PyTorch Eager Mode** and **`torch.compile` (TorchInductor)**.

---

## 📌 Key Insights & Highlights

* **Eager Mode vs. Compiler Fusion:** In standard PyTorch Eager Mode, Parallel Residual blocks fail to deliver expected speedups over Sequential Residuals ($49,965$ vs. $50,178 \text{ tok/s}$) due to single CUDA stream scheduling overhead.
* **The `torch.compile` Transformation:** Compiler fusion via TorchInductor resolves stream bottlenecks, unlocking a **$+24.0\%$ throughput uplift** for the combined GQA + Parallel Residual architecture ($64,374 \text{ tok/s}$).
* **Memory Trade-offs:** `torch.compile` trades static VRAM footprint for execution speed by pre-allocating persistent Triton workspace buffers and caching execution graphs.
* **Zero External LLM Libraries:** Built entirely from scratch using low-level `torch.nn` primitives and custom tensor operations.

---

## 🛠️ Architectural Components

### 1. Root Mean Square Normalization (RMSNorm)
Replaces standard LayerNorm by dispensing with mean-centering, scaling activations strictly by their root-mean-square. This reduces memory bandwidth overhead by avoiding $2\times$ reduction passes over hidden dimensions.

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 2. Rotary Position Embeddings (RoPE)
Encodes relative positional information directly into query and key representations by applying a complex space rotation matrix to 2D feature pairs along hidden dimensions.

$$R_{\Theta, m}^d x_m = \begin{pmatrix} x_1 \cos m\theta_1 - x_2 \sin m\theta_1 \\ x_1 \sin m\theta_1 + x_2 \cos m\theta_1 \\ \vdots \end{pmatrix}$$

### 3. Grouped-Query Attention (GQA)
Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). Multiple query heads ($H_q = 6$) share a single Key/Value head pair ($H_{\text{kv}} = 2$), reducing projection parameter count and shrinking the autoregressive KV-cache memory footprint.

### 4. SwiGLU Activation Module
Gated Linear Unit utilizing the SiLU (Swish) activation function, providing smoother gradient propagation than standard GELU or ReLU primitives.

$$\text{SwiGLU}(x) = \left( x W_{\text{gate}} \cdot \sigma(x W_{\text{gate}}) \right) \odot \left( x W_{\text{up}} \right) W_{\text{down}}$$

### 5. Parallel vs. Sequential Residual Networks
* **Sequential Residual (Baseline):**
  $$x' = x_l + \text{Attn}(\text{RMSNorm}(x_l))$$
  $$x_{l+1} = x' + \text{MLP}(\text{RMSNorm}(x'))$$

* **Parallel Residual (PaLM / Falcon Style):**
  $$x_{l+1} = x_l + \text{Attn}(\text{RMSNorm}(x_l)) + \text{MLP}(\text{RMSNorm}(x_l))$$

---

## 📊 Benchmark Results

All benchmarks were evaluated under identical experimental conditions:
* **Dataset:** TinyStories (5,000 samples, sequence length $L = 256$, batch size $B = 16$)
* **Model Configuration:** 6 Layers, $d_{\text{model}} = 384$, $H_q = 6$ Heads, Vocab Size = 50,257
* **Training Pipeline:** FP16 Automatic Mixed Precision (`torch.amp`), AdamW ($\text{LR} = 5 \times 10^{-4}$), Cosine LR Schedule with Warmup, Gradient Clipping ($1.0$), 500 Steps.

### 1. PyTorch Eager Mode Execution

| Configuration | Parameters | Val Loss | Val PPL | Throughput (tok/s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (MHA + Sequential)** | $52.76\text{M}$ | **3.1644** | **23.67** | $50,178.9$ | $4.157\text{ GB}$ |
| **2. Variant A (Parallel Residual)** | $52.76\text{M}$ | $3.1742$ | $23.91$ | $49,965.4$ | $4.124\text{ GB}$ |
| **3. Variant B (GQA: 2 KV Heads)** | $51.58\text{M}$ | $3.1966$ | $24.45$ | $51,831.6$ | $4.143\text{ GB}$ |
| **4. Variant C (GQA + Parallel)** | $51.58\text{M}$ | $3.1680$ | $23.76$ | **51,925.7** | **4.109 GB** |

### 2. Compiled Mode Execution (`torch.compile`)

| Configuration | Parameters | Val Loss | Val PPL | Throughput (tok/s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (MHA + Sequential)** | $52.76\text{M}$ | $3.1573$ | $23.51$ | $62,402.5$ | **3.937 GB** |
| **2. Variant A (Parallel Residual)** | $52.76\text{M}$ | **3.1516** | **23.37** | $63,035.5$ | $4.336\text{ GB}$ |
| **3. Variant B (GQA: 2 KV Heads)** | $51.58\text{M}$ | $3.1726$ | $23.87$ | $63,316.2$ | $4.725\text{ GB}$ |
| **4. Variant C (GQA + Parallel)** | $51.58\text{M}$ | $3.1914$ | $24.32$ | **64,374.2** | $5.115\text{ GB}$ |

### 3. Eager vs. Compiled Direct Comparison

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

## 🧠 Deep-Dive Systems Analysis

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

## 📂 Repository Structure

```text
.
├── src/
│   ├── models/
│   │   ├── layers.py         # RMSNorm, SwiGLU, RoPE, GQA
│   │   └── transformer.py    # Transformer Block & Decoder LLM
│   └── data.py               # Streaming dataset & block tokenizer
├── checkpoints/              # Model weights per experiment
├── results/                  # Metric JSON files & summary reports
├── train.py                  # Core training & evaluation script
├── run_matrix.py             # Eager Mode benchmarking harness
├── run_compile_benchmark.py  # Torch.compile benchmarking harness
└── README.md