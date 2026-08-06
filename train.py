"""
LLM Training Pipeline Script
============================
Handles model training, mixed-precision acceleration (Automatic Mixed Precision),
learning rate scheduling (Cosine Decay with Linear Warmup), gradient clipping,
weight decay separation, and evaluation loop (Perplexity calculation).
"""

import math
import time
import argparse
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data import get_dataloader
from src.models.transformer import Transformer


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Creates a learning rate scheduler that linearly increases the learning rate
    from 0 to max_lr over `num_warmup_steps`, then decays it following a cosine
    curve down to `min_lr_ratio * max_lr` over the remaining steps.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine Annealing Decay Phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        # Rescale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 5e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "cuda"
) -> AdamW:
    """
    Configures AdamW optimizer with proper Weight Decay separation:
    - 2D parameters (weights of Linear and Embedding layers) receive weight decay.
    - 1D parameters (biases, RMSNorm scale parameters) do NOT receive weight decay.
    
    Why? Applying weight decay (L2 regularization) to normalization gains or biases
    distorts signal scaling and degrades training stability.
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D tensors (biases, norm weights) shouldn't be decayed
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    print(
        f"Optimizer Setup: {len(decay_params)} decayed parameter tensors "
        f"({sum(p.numel() for p in decay_params):,} params), "
        f"{len(nodecay_params)} non-decayed parameter tensors "
        f"({sum(p.numel() for p in nodecay_params):,} params)."
    )

    # Use fused AdamW if running on CUDA for speedup
    fused_available = "fused" in torch.optim.AdamW.__doc__
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    return optimizer


@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    Evaluates the model on validation data and returns loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        total_loss += loss.item() * x.size(0)
        total_tokens += x.size(0)
        
    avg_loss = total_loss / total_tokens
    # Perplexity is the exponential of cross-entropy loss
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return avg_loss, perplexity


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> Tuple[float, float]:
    """
    Computes Validation Loss and Perplexity (PPL) on validation dataset split.
    
    Perplexity = exp(CrossEntropyLoss)
    Intuition: Perplexity represents the weighted average branching factor (number of
    equally likely words) the model is choosing between at each step.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    loss_fn = nn.CrossEntropyLoss()

    for idx, (x, y) in enumerate(val_loader):
        if idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(x)
            # Reshape logits: (batch_size * seq_len, vocab_size) and targets: (batch_size * seq_len)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        total_loss += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    
    model.train()
    return mean_loss, perplexity


def train(
    num_layers: int = 6,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    seq_len: int = 256,
    batch_size: int = 16,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    parallel_residual: bool = False,
    overfit_single_batch: bool = False,
    max_samples: int = 5000
):
    """
    Main training execution loop supporting single-batch overfitting sanity checks
    and full epoch iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on Device: {device}")

    # 1. Load Datasets
    print("📦 Loading Data...")
    train_loader, tokenizer = get_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        split="train",
        max_samples=max_samples
    )
    val_loader, _ = get_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        split="validation",
        max_samples=1000
    )

    # 2. Instantiate Model Architecture
    model = Transformer(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        vocab_size=tokenizer.vocab_size,
        rope=True
    ).to(device)

    # Configure Parallel Residuals if requested
    if parallel_residual:
        for block in model.layers:
            block.parallel_residual = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Transformer Model Instantiated | Total Parameters: {total_params / 1e6:.2f}M")

    # 3. Setup Optimizer, LR Scheduler, and Mixed Precision Scaler
    optimizer = configure_optimizers(
        model, weight_decay=weight_decay, learning_rate=learning_rate, device_type=device.type
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    
    # Use GradScaler for fp16 mixed precision (bf16 doesn't strictly need scaling)
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    loss_fn = nn.CrossEntropyLoss()
    model.train()

    step = 0
    start_time = time.time()

    # If debugging / overfit check: grab a single static batch
    if overfit_single_batch:
        print("⚠️ OVERFIT SINGLE BATCH MODE ENABLED (Sanity Check)...")
        x_single, y_single = next(iter(train_loader))
        x_single, y_single = x_single.to(device), y_single.to(device)

    while step < max_steps:
        for x_batch, y_batch in train_loader:
            if step >= max_steps:
                break

            if overfit_single_batch:
                x, y = x_single, y_single
            else:
                x, y = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast(
                enabled=True,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ):
                logits = model(x)  # Shape: (batch_size, seq_len, vocab_size)
                
                # Reshape for Cross Entropy Loss
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)), 
                    y.view(-1)
                )

            # Backward Pass with Gradient Scaling
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step & LR Scheduler step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1

            # Logging Progress
            if step % 20 == 0 or step == 1 or overfit_single_batch:
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = (x.numel()) / (time.time() - start_time + 1e-6)
                print(
                    f"Step {step:04d}/{max_steps} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Throughput: {tokens_per_sec:.1f} tok/s"
                )
                start_time = time.time()

            # Evaluate on Validation Set
            if step % 200 == 0 and not overfit_single_batch:
                val_loss, val_ppl = evaluate_perplexity(model, val_loader, device)
                print(
                    f"\n📊 [EVALUATION @ Step {step}] "
                    f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}\n"
                )

    print("Training Pipeline Run Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom TinyStories Transformer")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--overfit", action="store_true", help="Sanity check: overfit single batch")
    args = parser.parse_args()

    train(max_steps=args.steps, batch_size=args.batch_size, overfit_single_batch=args.overfit)