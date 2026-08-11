import torch
import time
from data import get_dataloader
from models.transformer import Transformer
from transformers import AutoTokenizer

def profile_decode_step(model, input_ids, model_size_gb, num_iters=100):
    # 1. Warm-up GPU allocations and CUDA context 
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()

    # 2. Measure CPU + GPU End-to-End Latency
    start_wall = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    end_wall = time.perf_counter()
    
    avg_wall_ms = ((end_wall - start_wall) / num_iters) * 1000

    # 3. Measure Isolated GPU Kernel Latency via CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    end_event.record()
    torch.cuda.synchronize()

    avg_gpu_ms = start_event.elapsed_time(end_event) / num_iters

    # 4. Calculate MBU Metrics
    peak_bandwidth_gb_s = 448.0  # Single RTX 2080 VRAM Bandwidth
    achieved_bandwidth = model_size_gb / (avg_gpu_ms / 1000.0)
    mbu_percent = (achieved_bandwidth / peak_bandwidth_gb_s) * 100.0

    print(f"--- Single-Token Decode Diagnostic Results ---")
    print(f"End-to-End Latency (CPU + GPU): {avg_wall_ms:.2f} ms")
    print(f"Isolated GPU Latency:           {avg_gpu_ms:.2f} ms")
    print(f"CPU Dispatch Overhead:          {(avg_wall_ms - avg_gpu_ms):.2f} ms")
    print(f"Achieved Memory Bandwidth:      {achieved_bandwidth:.2f} GB/s")
    print(f"Memory Bandwidth Utilization:   {mbu_percent:.2f}%")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/1_baseline_mha_sequential_res/model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

model = Transformer(**checkpoint["model_args"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = Transformer(**checkpoint["model_args"]).to(device)

single_token = torch.tensor([[tokenizer.encode("Hello")[-1]]], device=device)  # Single token input

bytes_per_param = next(model.parameters()).element_size()  # 4 for FP32, 2 for FP16
total_params = sum(p.numel() for p in model.parameters())  # Includes all parameters

model_size_gb = (total_params * bytes_per_param) / (1024 ** 3)
compiled_model = torch.compile(model, mode="reduce-overhead")

profile_decode_step(model=compiled_model, input_ids=single_token, model_size_gb=model_size_gb, num_iters=100)



"""
--- Single-Token Decode Diagnostic Results for eager model---
End-to-End Latency (CPU + GPU): 3.80 ms
Isolated GPU Latency:           3.79 ms
CPU Dispatch Overhead:          0.02 ms
Achieved Memory Bandwidth:      51.86 GB/s
Memory Bandwidth Utilization:   11.58%



"""


"""
--- Single-Token Decode Diagnostic Results for Compiled Model ---
End-to-End Latency (CPU + GPU): 0.43 ms
Isolated GPU Latency:           0.43 ms
CPU Dispatch Overhead:          0.00 ms
Achieved Memory Bandwidth:      455.31 GB/s
Memory Bandwidth Utilization:   101.63% 

"""