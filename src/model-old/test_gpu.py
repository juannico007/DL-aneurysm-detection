import torch
import time

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Create large random tensors
size = 10_000
a = torch.randn(size, size)
b = torch.randn(size, size)

# --- CPU test ---
start = time.time()
c_cpu = torch.mm(a, b)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# --- GPU test (if available) ---
if device == "cuda":
    a = a.to("cuda")
    b = b.to("cuda")
    torch.cuda.synchronize()  # ensure GPU is ready

    start = time.time()
    c_gpu = torch.mm(a, b)
    torch.cuda.synchronize()  # wait for GPU to finish
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x faster 🚀")
else:
    print("GPU not available.")