import argparse
import gc
import json
import os
import re
import time
import torch
import torch.nn as nn
from src.data import get_dataloader
from src.models.transformer import Transformer
from train import configure_optimizers, evaluate

def sanitize_slug(name: str) -> str:
    """Converts configuration titles into clean directory and file names."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()

def cleanup_gpu_memory():
    """Flushes Python garbage and CUDA memory pools between benchmark runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def run_experiment(config_name, model_args, train_args, train_loader, val_loader, device):
    print(f"\n==================================================")
    print(f"🚀 RUNNING EXPERIMENT: {config_name}")
    print(f"==================================================")
    
    slug = sanitize_slug(config_name)
    results_dir = "results"
    checkpoints_dir = os.path.join("checkpoints", slug)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Memory hygiene prior to allocation
    cleanup_gpu_memory()

    # 2. Instantiate Model & Optimizer
    model = Transformer(**model_args).to(device)
    optimizer = configure_optimizers(
        model, 
        weight_decay=0.1, 
        learning_rate=train_args["lr"], 
        device_type=device.type
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.2f}M")

    # 3. Warmup CUDA & execution graphs (5 un-timed steps)
    model.train()
    warmup_iter = iter(train_loader)
    for _ in range(5):
        try:
            wx, wy = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            wx, wy = next(warmup_iter)
            
        wx, wy = wx.to(device), wy.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            w_logits = model(wx)
            w_loss = nn.functional.cross_entropy(w_logits.view(-1, w_logits.size(-1)), wy.view(-1))
        w_loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    total_tokens_processed = 0
    step = 0

    # 4. Main Timed Loop
    data_iter = iter(train_loader)
    while step < train_args["max_steps"]:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens_processed += x.numel()
        step += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    throughput = total_tokens_processed / elapsed_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    val_loss, val_ppl = evaluate(model, val_loader, device)

    results = {
        "config": config_name,
        "slug": slug,
        "parameters_M": round(total_params / 1e6, 2),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(val_ppl, 2),
        "throughput_tok_sec": round(throughput, 1),
        "peak_vram_gb": round(peak_vram_gb, 3)
    }

    # 5. Persist Checkpoint & Individual Result File Immediately
    checkpoint_path = os.path.join(checkpoints_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": model_args,
        "train_args": train_args,
        "results": results
    }, checkpoint_path)
    print(f"💾 Checkpoint saved to: {checkpoint_path}")

    json_result_path = os.path.join(results_dir, f"{slug}.json")
    with open(json_result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Metrics saved to: {json_result_path}")

    # 6. Explicit Object Teardown
    del model, optimizer
    cleanup_gpu_memory()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Steps per experiment")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_args = {
        "seq_len": 256,
        "batch_size": 16,
        "max_samples": 5000,
        "lr": 5e-4,
        "max_steps": args.steps
    }

    base_model_args = {
        "num_layers": 6,
        "embed_dim": 384,
        "num_heads": 6,
        "vocab_size": 50257,
        "rope": True
    }

    print("📦 Loading Shared Dataset (Single Execution)...")
    train_loader, _ = get_dataloader(
        seq_len=train_args["seq_len"],
        batch_size=train_args["batch_size"],
        max_samples=train_args["max_samples"],
        split="train"
    )
    val_loader, _ = get_dataloader(
        seq_len=train_args["seq_len"],
        batch_size=train_args["batch_size"],
        max_samples=train_args["max_samples"],
        split="validation"
    )

    matrix = [
        ("1. Baseline (MHA + Sequential Res)", {**base_model_args, "num_kv_heads": 6, "parallel_residual": False}),
        ("2. Variant A (Parallel Residual)", {**base_model_args, "num_kv_heads": 6, "parallel_residual": True}),
        ("3. Variant B (GQA: 2 KV Heads)", {**base_model_args, "num_kv_heads": 2, "parallel_residual": False}),
        ("4. Variant C (GQA + Parallel Res)", {**base_model_args, "num_kv_heads": 2, "parallel_residual": True}),
    ]

    all_results = []
    for config_name, model_args in matrix:
        res = run_experiment(config_name, model_args, train_args, train_loader, val_loader, device)
        all_results.append(res)

    # Master summary file
    with open("results/benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "="*88)
    print("📊 ISOLATED BENCHMARKING RESULTS")
    print("="*88)
    print(f"{'Configuration':<35} | {'Params(M)':<9} | {'Val Loss':<8} | {'Val PPL':<8} | {'Tok/Sec':<9} | {'VRAM (GB)':<9}")
    print("-" * 88)
    for r in all_results:
        print(f"{r['config']:<35} | {r['parameters_M']:<9} | {r['val_loss']:<8} | {r['val_ppl']:<8} | {r['throughput_tok_sec']:<9} | {r['peak_vram_gb']:<9}")
    print("="*88)