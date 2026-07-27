"""Benchmark all H100 CuTe DSL kernels + cuBLAS. Outputs results.json."""
import os, sys, time, json, importlib.util, traceback
import torch

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

sys.path.insert(0, KERNELS_DIR)

import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing

PROBLEMS = [
    (128,128,128),(256,256,256),(512,512,512),(1024,1024,1024),
    (2048,2048,2048),(4096,4096,4096),(5120,5120,4096),(8192,8192,8192),
]
SHAPE_LABELS = ["128","256","512","1K","2K","4K","5K","8K"]

KERNELS = [
    ("cublas","cuBLAS",None),
    ("k1","K1",("1_wgmma_tma.py","HopperGemm",{})),
    ("k2","K2",("2_wgmma_tma_multistage.py","HopperGemm",{})),
    ("k3","K3",("3_wgmma_tma_multistage_epilogue.py","HopperGemm",{})),
    ("k4","K4",("4_wgmma_tma_multistage_WS.py","HopperGemmWarpSpecialized",{})),
    ("k5","K5",("5_wgmma_tma_multistage_WS2.py","HopperGemmWarpSpecialized",{})),
    ("k6","K6",("6_wgmma_tma_multistage_WS2_multicast.py","HopperGemmWarpSpecialized",{"cluster_shape_mn":(2,1)})),
]
KERNELS = [k for k in KERNELS if k[2] is None or os.path.exists(os.path.join(KERNELS_DIR, k[2][0]))]

def benchmark_kernel(cls, M, N, K, mod=None, config_kwargs=None):
    config_kwargs = config_kwargs or {}
    warmup, iters = 5, 20
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")
    if mod is not None and hasattr(mod, "GemmConfig"):
        obj = cls(config=mod.GemmConfig(**config_kwargs))
    else:
        obj = cls(**config_kwargs)
    a_c = from_dlpack(A, assumed_align=16)
    b_c = from_dlpack(B, assumed_align=16)
    c_c = from_dlpack(C, assumed_align=16)
    compiled = cute.compile(obj, a_c, b_c, c_c)
    compiled(a_c, b_c, c_c)
    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=5e1, rtol=5e-1), f"Correctness failed {M}x{N}x{K}"
    def gen():
        a=torch.randn(M,K,device="cuda",dtype=torch.float16)
        b=torch.randn(N,K,device="cuda",dtype=torch.float16)
        c=torch.empty(M,N,device="cuda",dtype=torch.float16)
        return testing.JitArguments(from_dlpack(a,16),from_dlpack(b,16),from_dlpack(c,16))
    one_ws = (A.numel()+B.numel()+C.numel())*A.element_size()
    wc = testing.get_workspace_count(one_ws, warmup, iters)
    exec_time = testing.benchmark(compiled, workspace_generator=gen, workspace_count=wc, warmup_iterations=warmup, iterations=iters)
    tflops = (2.0*M*N*K*1e-9)/(exec_time*1e-3)
    return tflops, exec_time

def benchmark_cublas(M, N, K):
    warmup, iters = 5, 20
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    C = A @ B
    torch.cuda.synchronize()
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    tflops = (2.0*M*N*K*1e-9) / elapsed / 1000.0
    return tflops, elapsed * 1000.0

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"CC: {props.major}.{props.minor}, SMs: {props.multi_processor_count}")
    print(f"Kernels: {[k[0] for k in KERNELS]}")

    results = {k[0]: {} for k in KERNELS}
    ms_results = {k[0]: {} for k in KERNELS}

    for M, N, K in PROBLEMS:
        sl = SHAPE_LABELS[PROBLEMS.index((M,N,K))]
        print(f"\n=== {M}x{N}x{K} ===", flush=True)
        for key, desc, mod_info in KERNELS:
            print(f"  {key:8s} ... ", end="", flush=True)
            try:
                if key == "cublas":
                    tf, ms = benchmark_cublas(M, N, K)
                else:
                    fname, cls_name, cfg_kwargs = mod_info
                    fpath = os.path.join(KERNELS_DIR, fname)
                    spec = importlib.util.spec_from_file_location("kmod", fpath)
                    kmod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kmod)
                    cls = getattr(kmod, cls_name)
                    tf, us = benchmark_kernel(cls, M, N, K, mod=kmod, config_kwargs=cfg_kwargs)
                    ms = us / 1000.0
                results[key][sl] = round(tf, 1)
                ms_results[key][sl] = round(ms, 3)
                print(f"{tf:>7.1f} TFLOPS  ({ms:.3f} ms)")
            except Exception as e:
                results[key][sl] = None
                ms_results[key][sl] = None
                print(f"FAIL: {e}")
                traceback.print_exc()

    out = {"results": results, "ms": ms_results}
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== TFLOPS ===")
    header = f"{'Shape':>8s}" + "".join(f"  {k[0]:>8s}" for k in KERNELS)
    print(header)
    for sl in SHAPE_LABELS:
        row = f"{sl:>8s}"
        for key, _, _ in KERNELS:
            v = results[key][sl]
            row += f"  {v:>8.1f}" if v is not None else f"  {'FAIL':>8s}"
        print(row)

if __name__ == "__main__":
    main()
