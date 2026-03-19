import torch
from quack import rmsnorm

def run(**params):
    M, N = 4096, 4096

    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.ones(N,     device="cuda", dtype=torch.float16)

    out = rmsnorm(x, w)

    # reference
    rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    ref = (x.float() * rms * w.float()).half()

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    return f"quack rmsnorm: M={M} N={N}  PASSED"
