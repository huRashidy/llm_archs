import time
import math
import torch
import torch.nn as nn
from src.models.transformer import Transformer

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_peak_bandwidth():
    if not torch.cuda.is_available():
        return 448.0
    device_name = torch.cuda.get_device_name(0)
    if "2080" in device_name:
        return 448.0
    elif "3090" in device_name:
        return 936.0
    elif "4090" in device_name:
        return 1008.0
    elif "A100" in device_name:
        return 1555.0
    elif "H100" in device_name:
        return 3350.0
    else:
        return 448.0  # Default fallback bandwidth in GB/s

def verify_kv_cache_correctness(device):
    print("==================================================")
    print("1. VERIFYING KV CACHE CORRECTNESS")
    print("==================================================")
    set_seed(42)
    model_args = dict(
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_kv_heads=2,
        vocab_size=1000,
        mlp_ratio=4.0,
        rope=True
    )
    model = Transformer(**model_args).to(device)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 10), device=device)
    max_new_tokens = 20

    set_seed(123)
    out_nocache = model.generate(prompt.clone(), max_new_tokens=max_new_tokens, temperature=1.0, top_k=1, use_cache=False)

    set_seed(123)
    out_cache = model.generate(prompt.clone(), max_new_tokens=max_new_tokens, temperature=1.0, top_k=1, use_cache=True)

    matches = torch.equal(out_nocache, out_cache)
    print(f"No-Cache Output Shape: {out_nocache.shape}")
    print(f"With-Cache Output Shape: {out_cache.shape}")
    print(f"Generated Tokens Match Exactly: {'✅ PASSED' if matches else '❌ FAILED'}\n")
    assert matches, "KV Cache generation output does not match non-cached generation!"

def profile_cache_vs_nocache(model, prompt, new_tokens_list, device):
    print("==================================================")
    print("2. AUTOREGRESSIVE GENERATION: KV CACHE VS NO CACHE")
    print("==================================================")
    print(f"{'New Tokens':<12} | {'No-Cache Time (s)':<18} | {'With-Cache Time (s)':<18} | {'Speedup':<10}")
    print("-" * 66)

    for num_tokens in new_tokens_list:
        # Warmup
        _ = model.generate(prompt.clone(), max_new_tokens=5, use_cache=False)
        _ = model.generate(prompt.clone(), max_new_tokens=5, use_cache=True)
        torch.cuda.synchronize()

        # No cache
        start = time.perf_counter()
        _ = model.generate(prompt.clone(), max_new_tokens=num_tokens, use_cache=False)
        torch.cuda.synchronize()
        time_nocache = time.perf_counter() - start

        # With cache
        start = time.perf_counter()
        _ = model.generate(prompt.clone(), max_new_tokens=num_tokens, use_cache=True)
        torch.cuda.synchronize()
        time_cache = time.perf_counter() - start

        speedup = time_nocache / max(time_cache, 1e-6)
        print(f"{num_tokens:<12} | {time_nocache:<18.4f} | {time_cache:<18.4f} | {speedup:<10.2f}x")
    print()

def profile_mha_vs_gqa_kv_cache(device, target_seq_len=2048, batch_size=4):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
    peak_bw_gbs = get_gpu_peak_bandwidth()

    print("==================================================")
    print("3. ARCHITECTURE COMPARISON AT LONG SEQUENCE LENGTH")
    print(f"   (Batch Size: {batch_size}, Sequence Length: {target_seq_len}, GPU: {gpu_name} [{peak_bw_gbs:.0f} GB/s peak])")
    print("==================================================")

    embed_dim = 2048
    num_heads = 16
    head_dim = embed_dim // num_heads  # 128
    num_layers = 16
    vocab_size = 32000

    configs = [
        ("MHA (Multi-Head Attention)", 16),
        ("GQA-4 (Grouped-Query Attention)", 4),
        ("GQA-2 (Grouped-Query Attention)", 2),
        ("MQA (Multi-Query Attention)", 1),
    ]

    header = f"{'Architecture':<32} | {'KV Heads':<8} | {'KV Cache':<10} | {'Peak VRAM':<12} | {'Step Time':<10} | {'Throughput':<14} | {'Bandwidth':<12} | {'MBU (%)':<8}"
    print(header)
    print("-" * len(header))

    for arch_name, num_kv_heads in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            rope=True
        ).to(device).eval()

        # Calculate bytes for model parameters
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Calculate theoretical KV Cache Memory Size (in bytes and MB)
        # 2 tensors (K, V) * num_layers * batch_size * num_kv_heads * target_seq_len * head_dim * 4 bytes (fp32)
        kv_cache_bytes = 2 * num_layers * batch_size * num_kv_heads * target_seq_len * head_dim * 4
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        total_bytes_moved = weight_bytes + kv_cache_bytes

        # Simulate context at target_seq_len - 1
        start_pos = target_seq_len - 1
        dummy_input = torch.randint(0, vocab_size, (batch_size, 1), device=device)

        # Build dummy past_key_values of length start_pos
        past_key_values = []
        for _ in range(num_layers):
            k_cache = torch.randn(batch_size, num_kv_heads, start_pos, head_dim, device=device)
            v_cache = torch.randn(batch_size, num_kv_heads, start_pos, head_dim, device=device)
            past_key_values.append((k_cache, v_cache))

        # Warmup single decode step
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input, past_key_values=past_key_values, use_cache=True, start_pos=start_pos)
        torch.cuda.synchronize()

        # Benchmark single token decode latency
        num_iters = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = model(dummy_input, past_key_values=past_key_values, use_cache=True, start_pos=start_pos)
        end_event.record()
        torch.cuda.synchronize()

        avg_step_ms = start_event.elapsed_time(end_event) / num_iters
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        decode_seconds = avg_step_ms / 1000.0
        throughput_tok_sec = batch_size / decode_seconds
        achieved_bandwidth_gbs = (total_bytes_moved / 1e9) / decode_seconds
        mbu_percent = (achieved_bandwidth_gbs / peak_bw_gbs) * 100.0

        kv_cache_str = f"{kv_cache_mb:.2f} MB"
        vram_str = f"{peak_vram_mb:.2f} MB"
        step_str = f"{avg_step_ms:.2f} ms"
        tp_str = f"{throughput_tok_sec:.2f} tok/s"
        bw_str = f"{achieved_bandwidth_gbs:.2f} GB/s"
        mbu_str = f"{mbu_percent:.2f}%"

        print(f"{arch_name:<32} | {num_kv_heads:<8} | {kv_cache_str:<10} | {vram_str:<12} | {step_str:<10} | {tp_str:<14} | {bw_str:<12} | {mbu_str:<8}")

        del model, past_key_values
        torch.cuda.empty_cache()

    print()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on Device: {device}\n")

    if device.type != "cuda":
        print("⚠️ CUDA device not available. Running on CPU.")

    verify_kv_cache_correctness(device)

    # Benchmark KV Cache vs No Cache speedup
    small_model = Transformer(
        num_layers=6,
        embed_dim=512,
        num_heads=8,
        num_kv_heads=2,
        vocab_size=1000,
        rope=True
    ).to(device).eval()

    prompt = torch.randint(0, 1000, (1, 16), device=device)
    profile_cache_vs_nocache(small_model, prompt, [32, 64, 128, 256], device)

    # Benchmark MHA vs GQA at long contexts
    if device.type == "cuda":
        profile_mha_vs_gqa_kv_cache(device, target_seq_len=2048, batch_size=4)

if __name__ == "__main__":
    main()
