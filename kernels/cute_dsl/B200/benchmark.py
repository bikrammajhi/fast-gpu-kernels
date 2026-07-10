import torch
import time

print(f"Device: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"Compute Capability: {props.major}.{props.minor}, SMs: {props.multi_processor_count}")

print(f"{'M':>8} {'N':>8} {'K':>8}   {'TFLOPs':>8} {'ms':>8}")

problems = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (5120, 5120, 4096),
    (8192, 8192, 8192),
]

torch.manual_seed(42)

for m, n, k in problems:
    gflops = 2.0 * m * n * k * 1e-9
    iters = 1000 if m < 512 else (200 if m < 2048 else 100)

    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(k, n, dtype=torch.float16, device="cuda")
    C = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    torch.cuda.synchronize()
    C = A @ B
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    tflops = gflops / elapsed / 1000.0
    print(f"{m:>8} {n:>8} {k:>8}   {tflops:>8.1f} {elapsed*1000:>8.3f}")
