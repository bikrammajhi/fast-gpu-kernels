import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a    = tl.load(a_ptr + offs, mask=mask)
    b    = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def run(**params):
    n = 1 << 20
    a = torch.ones(n,  device="cuda", dtype=torch.float32)
    b = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
    c = torch.empty(n, device="cuda", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    vector_add_kernel[grid](a, b, c, n, BLOCK=1024)

    assert torch.all(c == 3.0), "mismatch!"
    return f"triton vector_add: n={n}  c[0]={c[0].item():.1f}  c[-1]={c[-1].item():.1f}"
