# gpulab 🔬

A personal GPU kernel lab for running kernels across every major framework on NVIDIA Blackwell (B200) using [Modal](https://modal.com).

One command to run anything:

```bash
modal run scripts/run.py --task kernels/cuda/vector_add.cu
modal run scripts/run.py --task kernels/triton/vector_add.py
modal run scripts/run.py --task kernels/cute/vector_add.cu
modal run scripts/run.py --task kernels/cute_dsl/vector_add.py
modal run scripts/run.py --task kernels/cutlass/gemm.cu
modal run scripts/run.py --task kernels/quack/rmsnorm.py
modal run scripts/run.py --task repos/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py
```

---

## What's Inside

| Framework | Directory | Type | Notes |
|---|---|---|---|
| CUDA | `kernels/cuda/` | `.cu` | Raw CUDA, compiled with `nvcc` |
| CuTe C++ | `kernels/cute/` | `.cu` | Layout algebra, part of CUTLASS 3.x |
| CUTLASS | `kernels/cutlass/` | `.cu` | High-performance GEMM/conv primitives |
| Triton | `kernels/triton/` | `.py` | OpenAI's Python GPU kernel DSL |
| CuTe DSL | `kernels/cute_dsl/` | `.py` | NVIDIA's Python JIT compiler for CuTe |
| Quack | `kernels/quack/` | `.py` | Production CuTe DSL kernels (rmsnorm, softmax, etc.) |

CUTLASS repo examples can be run directly from `repos/cutlass/` — they resolve against the image's cloned copy, not your local files.

---

## Project Structure

```
gpulab/
├── kernels/                  # your kernels — mounted fresh on every run
│   ├── cuda/
│   ├── cute/
│   ├── cute_dsl/
│   ├── cutlass/
│   ├── triton/
│   └── quack/
├── repos/                    # local copies for reading — never uploaded to Modal
│   └── cutlass/
├── src/
│   └── gpulab/
│       ├── __init__.py
│       ├── compiler.py       # compile flags per backend
│       ├── runner.py         # dispatch logic
│       └── modal_app.py      # Modal image + remote function
├── scripts/
│   └── run.py                # CLI entrypoint
├── pyproject.toml
└── .gitignore
```

---

## Prerequisites

- Python 3.12+
- [conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- A [Modal](https://modal.com) account (free tier works)

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/your-handle/gpulab.git
cd gpulab

conda create -n gpulab python=3.12
conda activate gpulab
```

### 2. Install dependencies

```bash
pip install -e .
pip install modal
```

### 3. Log in to Modal

```bash
modal setup
```

This opens a browser window to authenticate. Once done, your credentials are saved locally.

Verify it works:

```bash
modal profile list
```

### 4. Verify Modal is working

```bash
modal run scripts/run.py --task kernels/cuda/vector_add.cu
```

The first run builds the container image — this takes a few minutes. Every run after that is fast. You should see:

```
╭──────────────────── kernels/cuda/vector_add.cu ────────────────────╮
│ cuda vector_add: n=1048576  c[0]=3.0  PASSED                       │
╰────────────────────────────────────────────────────────────────────╯
```

---

## Running Kernels

```bash
# CUDA
modal run scripts/run.py --task kernels/cuda/vector_add.cu

# CuTe C++
modal run scripts/run.py --task kernels/cute/vector_add.cu

# CUTLASS
modal run scripts/run.py --task kernels/cutlass/gemm.cu

# Triton
modal run scripts/run.py --task kernels/triton/vector_add.py

# CuTe DSL
modal run scripts/run.py --task kernels/cute_dsl/vector_add.py

# Quack
modal run scripts/run.py --task kernels/quack/rmsnorm.py

# Official CUTLASS examples from the repo
modal run scripts/run.py --task repos/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py

# Extra nvcc flags (e.g. for profiling)
modal run scripts/run.py --task kernels/cuda/vector_add.cu --flags="-lineinfo"
```

---

## Writing Your Own Kernels

### `.cu` kernels

Write a standard `int main()`. Drop the file in the right directory and run it — the backend is auto-detected from the folder name.

```c
// kernels/cuda/my_kernel.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void my_kernel(...) { ... }

int main() {
    // allocate, launch, verify
    printf("PASSED\n");
    return 0;
}
```

```bash
modal run scripts/run.py --task kernels/cuda/my_kernel.cu
```

### `.py` kernels (Triton, CuTe DSL, Quack)

Define a `run(**params)` function that returns a string. The runner calls this function.

```python
# kernels/triton/my_kernel.py
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    ...

def run(**params):
    # set up tensors, launch kernel, verify
    return "my_kernel: PASSED"
```

```bash
modal run scripts/run.py --task kernels/triton/my_kernel.py
```

If your script runs computation at the module level (no `run()` function), that is fine too — the runner handles it gracefully.

---

## How the Backend is Detected

The runner infers which backend to use from the file path:

| Path contains | Backend | What happens |
|---|---|---|
| `cuda/` | cuda | `nvcc -arch=sm_100` |
| `cute/` | cutlass | `nvcc` + CUTLASS includes |
| `cutlass/` | cutlass | `nvcc` + CUTLASS includes + `-std=c++17` |
| `cute_dsl/` | cute_dsl | Python import |
| `triton/` | triton | Python import |
| `quack/` | quack | Python import |
| `repos/` | repos_cutlass | Resolved against image's `/root/cutlass` |

---

## How the Image Works

The container image is built once and cached. It contains:

- `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` (base)
- `torch`, `triton`, `numpy`, `rich`, `jax[cuda12]`, `nvidia-cutlass-dsl[cu13]`, `quack-kernels[cu13]`
- CUTLASS cloned at `/root/cutlass`

Only `kernels/` and `src/` are uploaded on every run — both are small and fast. `repos/` is never uploaded.

**When does the image rebuild?** Only when you change `modal_app.py` — adding a new pip package, a new `run_commands`, etc. Changes to `kernels/` and `src/` never trigger a rebuild.

---

## Adding a New Framework

Follow these four steps:

**1. Install it in the image** — edit `src/gpulab/modal_app.py`:
```python
.pip_install("new-package")
# or
.run_commands("git clone https://github.com/org/repo.git /root/repo")
```

**2. Add include paths** — edit `src/gpulab/compiler.py` if it is a C++ framework:
```python
INCLUDES["new_backend"] = ["-I/root/repo/include"]
```

**3. Add a compile function** — edit `src/gpulab/compiler.py`:
```python
def compile_new_backend(src, binary, extra_flags):
    _nvcc(src, binary, INCLUDES["new_backend"], extra_flags)
```

**4. Add backend detection** — edit `src/gpulab/runner.py`:
```python
if "new_backend" in parts: return "new_backend"
```

This triggers one image rebuild. After that, every run with the new framework is fast.

---

## Adding an External Repo

For repos you want to clone locally and read but run from the image:

```bash
# Clone locally for reading
git clone https://github.com/org/repo repos/repo
```

Add to image in `modal_app.py`:
```python
.run_commands("git clone --depth 1 https://github.com/org/repo.git /root/repo")
```

The local copy in `repos/` is for reading the code. The container uses its own clone.

---

## Common Issues

**`modal` command not found or wrong version**

Make sure `which modal` and `which python` both point to your conda env:
```bash
conda activate gpulab
which modal   # should contain gpulab
which python  # should contain gpulab
```

If `modal` points to `~/.local/bin/modal`, add your conda env to PATH:
```bash
echo 'export PATH="$CONDA_PREFIX/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Files not found in container**

If you get `FileNotFoundError` for a file that exists locally, the most common cause is a nested `.git` directory blocking Modal's file sync. Fix:
```bash
rm -rf repos/myrepo/.git
```

Or better: do not mount `repos/` at all — clone repos into the image instead.

**`cute/tensor.hpp: No such file or directory`**

Your kernel is in `kernels/cuda/` but includes CuTe headers. Move it to `kernels/cute/` — the runner will automatically add CUTLASS include paths.

**CUTLASS compile errors about C++17**

The `compile_cutlass` function already passes `-std=c++17`. If you are calling `compile_cuda` on a CUTLASS kernel, make sure the file is in `kernels/cutlass/` or `kernels/cute/`.

**Quack crashes with `No module named 'jax.numpy'`**

JAX is missing. Make sure `jax[cuda12]` is in your `pip_install()` in `modal_app.py` and rebuild the image.

---

## Target Hardware

All kernels are compiled for `sm_100` (NVIDIA Blackwell, B200/B300). To target a different GPU, update `ARCH` in `src/gpulab/compiler.py`:

```python
# H100 (Hopper)
ARCH = ["-arch=sm_90", "-gencode", "arch=compute_90,code=sm_90"]

# A100 (Ampere)
ARCH = ["-arch=sm_80", "-gencode", "arch=compute_80,code=sm_80"]
```

And update `gpu="B200"` in `src/gpulab/modal_app.py` to match.

---

## Acknowledgements

- [Modal](https://modal.com) for the GPU infrastructure
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) for CuTe and CUTLASS
- [Dao-AILab Quack](https://github.com/Dao-AILab/quack) for production CuTe DSL kernels
- [OpenAI Triton](https://github.com/openai/triton) for the Python GPU kernel DSL