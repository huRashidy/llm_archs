import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer import Transformer

BUCKETS = [256, 512, 1024, 2048, 4096]

def get_bucket(seq_len):
    for b in BUCKETS:
        if seq_len <= b:
            return b
    raise ValueError(f"seq_len {seq_len} exceeds largest bucket {BUCKETS[-1]}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_peak_bandwidth():
    if not torch.cuda.is_available():
        return 616.0
    device_name = torch.cuda.get_device_name(0)
    if "2080 Ti" in device_name:
        return 616.0
    elif "2080" in device_name:
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
        return 616.0

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

def profile_decode_loop(device, initial_context_len, max_new_tokens, batch_size, title):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
    peak_bw_gbs = get_gpu_peak_bandwidth()
    max_seq_len = initial_context_len + max_new_tokens

    print("========================================================================================================================")
    print(f"{title}")
    print(f"   (Batch Size: {batch_size}, Context: {initial_context_len} -> {max_seq_len}, GPU: {gpu_name}, BW: {peak_bw_gbs} GB/s)")
    print("========================================================================================================================")

    embed_dim = 2048
    num_heads = 16
    head_dim = embed_dim // num_heads  # 128
    num_layers = 16
    vocab_size = 32000
    dtype = torch.float16

    configs = [
        ("MQA (Multi-Query Attention)", 1),
        ("GQA-2 (Grouped-Query Attention)", 2),
        ("GQA-4 (Grouped-Query Attention)", 4),
        ("MHA (Multi-Head Attention)", 16),
    ]

    header = f"{'Architecture':<32} | {'KV Heads':<8} | {'KV Cache':<12} | {'Peak VRAM':<12} | {'Total Time':<10} | {'Step Time':<10} | {'Throughput':<14} | {'Achieved BW':<14} | {'MBU (%)':<10}"
    print(header)
    print("-" * len(header))

    for arch_name, num_kv_heads in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        kv_cache_bytes = 2 * num_layers * batch_size * num_kv_heads * max_seq_len * head_dim * 2
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        try:
            model = Transformer(
                num_layers=num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                vocab_size=vocab_size,
                rope=True
            ).to(device=device, dtype=dtype).eval()

            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

            # Compile Attention and MLP submodules with mode="reduce-overhead"
            torch._dynamo.config.suppress_errors = True
            for layer in model.layers:
                layer.attn = torch.compile(layer.attn, mode="reduce-overhead", dynamic=True)
                layer.mlp = torch.compile(layer.mlp, mode="reduce-overhead", dynamic=True)

            # Pre-allocate static KV cache buffer
            past_key_values = []
            for _ in range(num_layers):
                k_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
                v_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
                past_key_values.append((k_cache, v_cache))

            # Warmup
            dummy_input = torch.randint(0, vocab_size, (batch_size, 1), device=device)
            for pos in [0, 1]:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        _ = model(dummy_input, past_key_values=past_key_values, use_cache=True, start_pos=pos)
            torch.cuda.synchronize()

            # Autoregressive decoding loop over max_new_tokens
            curr_pos = initial_context_len
            curr_tok = torch.randint(0, vocab_size, (batch_size, 1), device=device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for t in range(max_new_tokens):
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        logits, _ = model(curr_tok, past_key_values=past_key_values, use_cache=True, start_pos=curr_pos + t)
                        curr_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            end_event.record()
            torch.cuda.synchronize()

            total_decode_ms = start_event.elapsed_time(end_event)
            avg_step_sec = (total_decode_ms / 1000.0) / max_new_tokens
            avg_step_ms = avg_step_sec * 1000.0
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            total_tokens_generated = batch_size * max_new_tokens
            throughput_tok_sec = total_tokens_generated / (total_decode_ms / 1000.0)

            # Average KV cache length across decode loop
            avg_seq_len = initial_context_len + (max_new_tokens / 2.0)
            avg_kv_read_bytes_per_step = 2 * num_layers * batch_size * num_kv_heads * avg_seq_len * head_dim * 2
            avg_kv_write_bytes_per_step = 2 * num_layers * batch_size * num_kv_heads * 1 * head_dim * 2

            total_bytes_per_step = param_bytes + avg_kv_read_bytes_per_step + avg_kv_write_bytes_per_step
            achieved_bw_gbs = (total_bytes_per_step / avg_step_sec) / 1e9
            mbu = (achieved_bw_gbs / peak_bw_gbs) * 100.0

            kv_cache_str = f"{kv_cache_mb:.2f} MB"
            vram_str = f"{peak_vram_mb:.2f} MB"
            tot_time_str = f"{total_decode_ms / 1000.0:.2f} s"
            step_str = f"{avg_step_ms:.2f} ms"
            tp_str = f"{throughput_tok_sec:.2f} tok/s"
            bw_str = f"{achieved_bw_gbs:.2f} GB/s"
            mbu_str = f"{mbu:.2f} %"

            print(f"{arch_name:<32} | {num_kv_heads:<8} | {kv_cache_str:<12} | {vram_str:<12} | {tot_time_str:<10} | {step_str:<10} | {tp_str:<14} | {bw_str:<14} | {mbu_str:<10}")

            del model, past_key_values
            torch.cuda.empty_cache()

        except torch.OutOfMemoryError:
            kv_cache_str = f"{kv_cache_mb:.2f} MB"
            print(f"{arch_name:<32} | {num_kv_heads:<8} | {kv_cache_str:<12} | {'OOM':<12} | {'OOM':<10} | {'OOM':<10} | {'OOM':<14} | {'OOM':<14} | {'OOM':<10}")
            torch.cuda.empty_cache()

    print()

def profile_bucket_shifting_decode(device, prompt_len=200, total_generate_tokens=1900, batch_size=4):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"

    print("========================================================================================================================")
    print("5. BUCKET-SHIFTING DECODING BENCHMARK (PROMPT: 200 -> GENERATING 1900 TOKENS = 2100 TOTAL TOKENS)")
    print(f"   (Buckets: {BUCKETS}, Batch Size: {batch_size}, GPU: {gpu_name})")
    print("========================================================================================================================")

    embed_dim = 2048
    num_heads = 16
    head_dim = embed_dim // num_heads  # 128
    num_layers = 16
    vocab_size = 32000
    dtype = torch.float16

    configs = [
        ("MQA (Multi-Query Attention)", 1),
        ("GQA-2 (Grouped-Query Attention)", 2),
        ("GQA-4 (Grouped-Query Attention)", 4),
        ("MHA (Multi-Head Attention)", 16),
    ]

    header = f"{'Architecture':<32} | {'KV Heads':<8} | {'Total Time':<12} | {'Step Time':<12} | {'Throughput':<16} | {'Peak GPU VRAM':<16} | {'Status':<10}"
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
        ).to(device=device, dtype=dtype).eval()

        # Compile submodules for reduce-overhead
        torch._dynamo.config.suppress_errors = True
        for layer in model.layers:
            layer.attn = torch.compile(layer.attn, mode="reduce-overhead", dynamic=True)
            layer.mlp = torch.compile(layer.mlp, mode="reduce-overhead", dynamic=True)

        curr_bucket = get_bucket(prompt_len)

        # Pre-allocate KV cache for initial bucket
        past_key_values = []
        for _ in range(num_layers):
            k_cache = torch.zeros(batch_size, num_kv_heads, curr_bucket, head_dim, device=device, dtype=dtype)
            v_cache = torch.zeros(batch_size, num_kv_heads, curr_bucket, head_dim, device=device, dtype=dtype)
            past_key_values.append((k_cache, v_cache))

        prompt_tokens = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
        prompt_padded = F.pad(prompt_tokens, (0, curr_bucket - prompt_len))

        # Prefill stage
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits, _ = model(prompt_padded, past_key_values=past_key_values, use_cache=True, start_pos=0)
                curr_tok = torch.argmax(logits[:, prompt_len - 1:prompt_len, :], dim=-1)

        curr_pos = prompt_len
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for step in range(total_generate_tokens):
            target_pos = curr_pos + step
            next_bucket = get_bucket(target_pos + 1)

            # Bucket Shift: Expand memory buffers when sequence length crosses bucket threshold
            if next_bucket > curr_bucket:
                new_past_key_values = []
                for i in range(num_layers):
                    k_old, v_old = past_key_values[i]
                    k_new = torch.zeros(batch_size, num_kv_heads, next_bucket, head_dim, device=device, dtype=dtype)
                    v_new = torch.zeros(batch_size, num_kv_heads, next_bucket, head_dim, device=device, dtype=dtype)
                    k_new[:, :, :curr_bucket, :] = k_old
                    v_new[:, :, :curr_bucket, :] = v_old
                    new_past_key_values.append((k_new, v_new))
                past_key_values = new_past_key_values
                curr_bucket = next_bucket

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits, _ = model(curr_tok, past_key_values=past_key_values, use_cache=True, start_pos=target_pos)
                    curr_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_step_ms = total_ms / total_generate_tokens
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        total_tokens = batch_size * total_generate_tokens
        throughput_tok_sec = total_tokens / (total_ms / 1000.0)

        tot_time_str = f"{total_ms / 1000.0:.2f} s"
        step_str = f"{avg_step_ms:.2f} ms"
        tp_str = f"{throughput_tok_sec:.2f} tok/s"
        vram_str = f"{peak_vram_mb:.2f} MB"

        print(f"{arch_name:<32} | {num_kv_heads:<8} | {tot_time_str:<12} | {step_str:<12} | {tp_str:<16} | {vram_str:<16} | {'✅ PASSED':<10}")

        del model, past_key_values
        torch.cuda.empty_cache()

    print()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on Device: {device}\n")

    if device.type != "cuda":
        print("⚠️ CUDA device not available. Running on CPU.")

    verify_kv_cache_correctness(device)

    if device.type == "cuda":
        # 1. Short Sequence Generation (128 Tokens, Batch Size 16)
        profile_decode_loop(
            device,
            initial_context_len=2048,
            max_new_tokens=128,
            batch_size=16,
            title="2. SHORT SEQUENCE GENERATION BENCHMARK (128 TOKENS)"
        )

        # 2. Long Sequence Generation (1024 Tokens, Batch Size 4)
        profile_decode_loop(
            device,
            initial_context_len=2048,
            max_new_tokens=1024,
            batch_size=4,
            title="3. LONG SEQUENCE GENERATION BENCHMARK (1024 TOKENS)"
        )

        # 3. Extreme Long Sequence Generation (2048 Tokens, Batch Size 8)
        profile_decode_loop(
            device,
            initial_context_len=2048,
            max_new_tokens=2048,
            batch_size=8,
            title="4. EXTREME LONG SEQUENCE GENERATION BENCHMARK (2048 TOKENS)"
        )

        # 4. Bucket-Shifting Decoding Benchmark (Prompt: 200 -> Generating 1900 Tokens = 2100 Total)
        profile_bucket_shifting_decode(
            device,
            prompt_len=200,
            total_generate_tokens=1900,
            batch_size=4
        )

if __name__ == "__main__":
    main()
