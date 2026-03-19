# Running GPU Kernels on NVIDIA B200 with Modal: A Complete Tutorial

*How to build a personal GPU kernel lab that runs CUDA, CuTe, CUTLASS, Triton, CuTe DSL, and Quack — from scratch, step by step.*

---

## Who This Is For

You want to write and run GPU kernels. You have heard of CUDA, Triton, CUTLASS, CuTe — maybe you have read some papers or watched some talks — but you have never actually set up an environment to run them yourself. You do not own a high-end GPU. You want to run things on real hardware (a B200 in this case) without spending weeks on environment setup.

This tutorial walks through exactly how to do that using Modal, a cloud platform that lets you run code on GPUs from your laptop. We will build everything from scratch, explain every decision, and by the end you will have a single command that can run any kernel across any framework.

---

## Before We Start: Understanding the Pieces

There are a few concepts worth understanding before writing any code.

### What is Modal?

Modal is a cloud platform where you define your compute environment in Python and run functions remotely on GPUs. Instead of SSHing into a server and configuring CUDA manually, you describe what you want in a Python file and Modal handles the rest. The key insight is that your code runs in a container on a remote machine, not on your laptop.

### What is a Container?

A container is an isolated environment — think of it as a clean virtual machine that starts fresh every time. It has its own filesystem, its own Python installation, its own CUDA toolkit. When your kernel runs, it runs inside this container. When it finishes, the container disappears. This is why we need to think carefully about what files to put inside it and how.

### The Two Ways to Get Files Into a Modal Container

This is the most important concept in this whole tutorial. There are two ways:

**Baked into the image.** When Modal builds your container image, it can run commands, install packages, and clone repos. These operations happen once and get cached. Every subsequent run reuses this cached image — so these operations are fast after the first build. Use this for anything large, slow to download, or that rarely changes: CUDA toolkit, pip packages, CUTLASS headers.

**Mounted at runtime.** Every time you run `modal run`, Modal can upload files from your laptop into the container. This happens on every run — so it should be small and fast. Use this only for files you are actively editing: your own kernels, your own code.

The mistake most people make is putting everything in the mount. This makes every run slow because you are uploading megabytes of files every single time. The goal is to keep mounts small.

### What Are All These Frameworks?

Before we write any code, let's understand what each framework actually is:

**CUDA** is NVIDIA's programming model for GPUs. You write `.cu` files in a C++ dialect, compile them with `nvcc`, and run the resulting binary. This is the lowest level and gives you the most control.

**PTX** (Parallel Thread Execution) is NVIDIA's intermediate assembly language. When you compile CUDA, it first compiles to PTX, then to actual machine code. You can write PTX directly for maximum control, but most people write CUDA instead.

**CuTe** is a layout algebra library that is part of CUTLASS 3.x. It gives you abstractions for describing how data is arranged in memory — shapes, strides, tiles. You still write `.cu` files and compile with `nvcc`, but you use CuTe's tensor types instead of raw pointers. This makes tiling and memory access patterns much easier to reason about.

**CUTLASS** is NVIDIA's library of high-performance GEMM (matrix multiplication) and convolution primitives. It is built on top of CuTe. Instead of writing the inner loop yourself, you configure the operation (tile shape, data types, pipeline schedule) and CUTLASS generates the optimised kernel for you.

**Triton** is OpenAI's Python-based GPU kernel DSL. You write kernels in Python with special decorators, and Triton compiles them to GPU code at runtime. Much easier to get started with than raw CUDA, and competitive in performance for many operations.

**CuTe DSL** is NVIDIA's Python JIT compiler that lets you write CuTe kernels in Python instead of C++. You install it with pip (`nvidia-cutlass-dsl`) and it compiles to CUDA device code using MLIR under the hood. No `nvcc` needed.

**Quack** (QuACK: A Quirky Assortment of CuTe Kernels) is a library from Dao-AILab that provides production-quality kernels (RMSNorm, softmax, cross entropy, etc.) written using the CuTe DSL. You install it with pip and import it like any Python library.

---

## Part 0: Setting Up Modal

Before we write a single kernel, we need to get Modal working. This section takes you from a fresh machine to running your first remote GPU function.

### Step 1: Install VS Code

If you do not have VS Code, download it from [code.visualstudio.com](https://code.visualstudio.com). Install the **Python extension** from the Extensions panel (Ctrl+Shift+X, search for "Python" by Microsoft).

### Step 2: Set up a Python environment

We recommend conda to manage your environment. If you do not have it, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

Open your terminal (in VS Code: Terminal → New Terminal) and run:

```bash
conda create -n gpulab python=3.12
conda activate gpulab
```

You will see `(gpulab)` at the start of your terminal prompt. This means you are inside the environment. Every time you open a new terminal, run `conda activate gpulab` again.

In VS Code, select this as your Python interpreter: press Ctrl+Shift+P → "Python: Select Interpreter" → choose `gpulab`.

### Step 3: Install Modal

```bash
pip install modal
modal --version
```

You should see a version number like `1.3.5` or higher.

**Important:** Make sure `modal` in your terminal comes from the same environment as your Python. Run:

```bash
which modal
which python
```

Both paths should contain `gpulab`. If `modal` points somewhere else (like `~/.local/bin/modal`), it means you have an old Modal install from before. Fix it by adding your conda env to the front of your PATH:

```bash
echo 'export PATH="$CONDA_PREFIX/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Create a Modal account and log in

Go to [modal.com](https://modal.com) and create a free account. Then run:

```bash
modal setup
```

This opens a browser window asking you to log in and authorise your terminal. Once done, your credentials are saved and you are ready to run things remotely.

Verify you are logged in:

```bash
modal profile list
```

You should see your account name listed.

### Step 5: Run your first Modal function

Let's verify everything works end to end using Modal's official hello world example.

```bash
git clone https://github.com/modal-labs/modal-examples
cd modal-examples
modal run 01_getting_started/hello_world.py
```

You should see:

```
✓ Initialized. View run at https://modal.com/apps/...
✓ Created objects.
└── 🔨 Created function hello_world.

hello world
```

If you see "hello world" printed, Modal is working. The function ran on a remote server, not on your laptop.

### Step 6: Understand what just happened

Open `01_getting_started/hello_world.py` and look at the code:

```python
import modal

app = modal.App("hello-world")

@app.function()
def hello_world():
    print("hello world")

@app.local_entrypoint()
def main():
    hello_world.remote()
```

A few things to notice:

- `modal.App` defines your application — think of it as a named project.
- `@app.function()` marks a Python function to run remotely. When you call `.remote()` on it, Modal sends it to a server.
- `@app.local_entrypoint()` marks the function that runs on your laptop when you do `modal run`. This is your entry point.
- `.remote()` is what tells Modal "run this on a server, not here".

This pattern is the core of everything we will build.

### Step 7: Verify GPU access

Create a file called `test_gpu.py` anywhere and run it:

```python
import modal

app = modal.App("test-gpu")

@app.function(gpu="any")
def check_gpu():
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def main():
    check_gpu.remote()
```

```bash
modal run test_gpu.py
```

You should see `nvidia-smi` output showing a real NVIDIA GPU. The GPU you get with `gpu="any"` will vary. When we build our kernel lab, we will specify `gpu="B200"` to always get a Blackwell GPU.

Now you are ready. Let's build the kernel lab.

---

## Part 1: Setting Up the Project

### Step 1: Create the folder structure

We are going to create a folder called `gpulab`. Everything lives here.

```bash
mkdir gpulab
cd gpulab

mkdir -p kernels/cuda
mkdir -p kernels/cutlass
mkdir -p kernels/cute
mkdir -p kernels/cute_dsl
mkdir -p kernels/triton
mkdir -p kernels/quack
mkdir -p src/gpulab
mkdir -p scripts
mkdir -p repos
touch repos/.gitkeep
```

**Why this structure?**

`kernels/` is where your kernel files live, organised by framework. This is the folder that gets uploaded to Modal on every run. Everything in here should be small.

`src/gpulab/` is your Python package — the runner, compiler, and Modal app definition. This also gets uploaded on every run.

`scripts/` holds the CLI entrypoint — the command you actually type.

`repos/` is for cloning external repos locally so you can read the code. It never gets uploaded to Modal. The container has its own copies.

### Step 2: Set up the Python package

```bash
touch src/gpulab/__init__.py
touch src/gpulab/compiler.py
touch src/gpulab/runner.py
touch src/gpulab/modal_app.py
touch scripts/run.py
touch pyproject.toml
touch .gitignore
```

### Step 3: Write `pyproject.toml`

```bash
cat > pyproject.toml << 'EOF'
[project]
name = "gpulab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "modal>=1.0",
    "torch",
    "triton",
    "numpy",
    "rich",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF
```

**Why `pyproject.toml`?** This is the modern Python standard for describing a package. It lets you run `pip install -e .` to install your package in editable mode, so you can import `gpulab` from anywhere on your machine.

### Step 4: Write `.gitignore`

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.egg-info/
dist/
.venv/
.modal/
EOF
```

**Note about `repos/`:** We do not gitignore `repos/` itself, but we also never commit large repos into it. It is just a local working area. The container does not use it.

### Step 5: Install the package

```bash
pip install -e .
```

---

## Part 2: The Compiler

The compiler is responsible for one thing: taking a source file and producing a runnable binary (for `.cu` files) or doing nothing (for `.py` files, which Python handles at import time).

### Why do we need a compiler module at all?

Different frameworks need different compile flags. A plain CUDA kernel just needs `nvcc -arch=sm_100`. A CUTLASS kernel needs CUTLASS include paths and `-std=c++17`. Each framework has its own requirements, so we centralise all of that in one place.

### Write `src/gpulab/compiler.py`

```bash
cat > src/gpulab/compiler.py << 'EOF'
import subprocess
import time
from pathlib import Path
from rich.console import Console

console = Console()

# Target architecture — B200 is sm_100 (Blackwell)
# This is different from H100 (sm_90) and A100 (sm_80)
# Using the wrong arch will either fail or silently run slower
ARCH = ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"]

# CUTLASS lives at /root/cutlass in the container — cloned at image build time
CUTLASS_ROOT = "/root/cutlass"

INCLUDES = {
    "cuda": [],
    "cutlass": [
        f"-I{CUTLASS_ROOT}/include",
        f"-I{CUTLASS_ROOT}/tools/util/include",
    ],
}

# CUTLASS 3.x uses C++17 features — without this flag, templates fail
CXX17_FLAGS = ["-std=c++17"]


def _nvcc(src: str, binary: str, includes: list[str], extra_flags: list[str]) -> None:
    cmd = ["nvcc", "-O3", *includes, *ARCH, "-lcublas", *extra_flags, "-o", binary, src]
    console.log(f"[dim]compile[/dim]  {src}")
    t0 = time.perf_counter()
    cc = subprocess.run(cmd, capture_output=True, text=True)
    if cc.returncode != 0:
        raise RuntimeError(f"compile failed:\n{cc.stderr}")
    console.log(f"[dim]done[/dim]     {(time.perf_counter()-t0)*1000:.0f}ms")


def compile_cuda(src: str, binary: str, extra_flags: list[str]) -> None:
    _nvcc(src, binary, INCLUDES["cuda"], extra_flags)


def compile_cutlass(src: str, binary: str, extra_flags: list[str]) -> None:
    # Also add the source file's own directory and examples/common/
    # because CUTLASS examples include local headers like "helper.h"
    src_dir   = str(Path(src).parent)
    extra_inc = [f"-I{src_dir}", f"-I{CUTLASS_ROOT}/examples/common"]
    _nvcc(src, binary, [*INCLUDES["cutlass"], *extra_inc], [*CXX17_FLAGS, *extra_flags])
EOF
```

**Key decisions explained:**

`sm_100` — B200 is Blackwell architecture, which is compute capability 10.0, expressed as `sm_100`. This is different from H100 (`sm_90`) and A100 (`sm_80`). If you use the wrong architecture flag, `nvcc` will either refuse to compile or compile for the wrong target and run slowly or incorrectly.

`-std=c++17` for CUTLASS — CUTLASS 3.x uses C++17 features like `if constexpr`, structured bindings, and certain template metaprogramming patterns. Without this flag, the CUTLASS headers will fail to compile with cryptic errors.

`extra_inc` for CUTLASS examples — the official CUTLASS examples include local headers like `#include "helper.h"`. These live in the source file's own directory and in `examples/common/`. We add both include paths so they resolve.

---

## Part 3: The Runner

The runner is the brain of the operation. It looks at the task path, figures out which backend to use, calls the right compile function, and runs the result.

### Write `src/gpulab/runner.py`

```bash
cat > src/gpulab/runner.py << 'EOF'
import subprocess
import importlib
import sys
import os
from pathlib import Path
from rich.console import Console
from gpulab.compiler import compile_cuda, compile_cutlass

console = Console()

# ROOT is where your kernels and src land in the container
ROOT = "/root/gpulab"

# CUTLASS is cloned into the image at build time — not mounted
CUTLASS_ROOT = "/root/cutlass"


def _detect_backend(task: str) -> str:
    """
    Infer which backend to use from the task path.
    We use the directory name as a signal:
      kernels/cuda/foo.cu      → cuda
      kernels/cute/foo.cu      → cutlass (CuTe C++ needs CUTLASS headers)
      kernels/cutlass/foo.cu   → cutlass
      kernels/cute_dsl/foo.py  → cute_dsl
      kernels/triton/foo.py    → triton
      kernels/quack/foo.py     → quack
      repos/cutlass/...        → repos_cutlass (resolved against image)
    """
    parts = set(Path(task).parts)
    if "repos"    in parts: return "repos_cutlass"
    if "cutlass"  in parts: return "cutlass"
    if "cute"     in parts: return "cutlass"   # CuTe C++ uses same headers as CUTLASS
    if "cute_dsl" in parts: return "cute_dsl"
    if "triton"   in parts: return "triton"
    if "quack"    in parts: return "quack"
    return "cuda"


def _run_binary(binary: str) -> str:
    """Run a compiled binary and return its stdout."""
    out = subprocess.run([binary], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runtime error:\n{out.stderr}")
    return out.stdout


def _run_python(task: str, params: dict) -> str:
    """
    Import a Python module and call its run() function.
    If no run() function exists, the module runs at import time
    (which is fine for standalone scripts like CuTe DSL examples).
    """
    sys.path.insert(0, ROOT)
    module = task.removesuffix(".py").replace("/", ".")
    console.log(f"[dim]import[/dim]   {module}")
    mod = importlib.import_module(module)
    if hasattr(mod, "run"):
        return str(mod.run(**params))
    return f"{task} completed"


def _run_script(script: str) -> str:
    """
    Execute a Python script directly as a subprocess.
    Used for repo examples that are standalone scripts,
    not importable modules.
    """
    if not os.path.exists(script):
        raise FileNotFoundError(f"{script} not found")
    console.log(f"[dim]exec[/dim]     {script}")
    env = {
        **os.environ,
        "PYTHONPATH": f"{Path(script).parent}:{Path(script).parent.parent}",
    }
    out = subprocess.run(["python3", script], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"script failed:\n{out.stderr}")
    return out.stdout


def _resolve(task: str) -> str:
    """
    Convert a task path to an absolute path in the container.

    kernels/cuda/foo.cu → /root/gpulab/kernels/cuda/foo.cu
    repos/cutlass/examples/... → /root/cutlass/examples/...

    Note: repos/cutlass maps to /root/cutlass (not /root/gpulab/repos/cutlass)
    because CUTLASS is cloned into the image, not mounted.
    """
    parts = Path(task).parts
    if parts[0] == "repos":
        # Strip "repos/cutlass" prefix, resolve against image's /root/cutlass
        return os.path.join(CUTLASS_ROOT, *parts[2:])
    return os.path.join(ROOT, task)


def dispatch(task: str, params: dict, extra_flags: list[str] | None = None) -> str:
    extra_flags = extra_flags or []
    backend = _detect_backend(task)
    console.log(f"backend: [bold]{backend}[/bold]")

    # .cu files: compile with nvcc, run binary
    if task.endswith(".cu"):
        src    = _resolve(task)
        binary = src.removesuffix(".cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} not found")
        if backend in ("cutlass", "repos_cutlass"):
            compile_cutlass(src, binary, extra_flags)
        else:
            compile_cuda(src, binary, extra_flags)
        return _run_binary(binary)

    # repos .py files: run as standalone script
    if backend == "repos_cutlass":
        return _run_script(_resolve(task))

    # all other .py files: import as module, call run()
    return _run_python(task, params)
EOF
```

**Key decisions explained:**

**Why detect backend from path?** We could ask the user to specify `--backend cuda` every time, but that is annoying. The folder name is a reliable enough signal — if you put a file in `kernels/cute/`, it is a CuTe kernel and needs CUTLASS headers. This keeps the CLI simple.

**Why does CuTe C++ map to the `cutlass` backend?** CuTe is part of CUTLASS. The headers for CuTe (`cute/tensor.hpp`, etc.) live inside the CUTLASS repo under `include/cute/`. So any kernel that uses CuTe needs CUTLASS include paths. They are the same backend from a compilation standpoint.

**Why two different Python execution paths (`_run_python` vs `_run_script`)?** Your own kernels in `kernels/` are proper Python modules with a `run()` function that the runner can call. External repo examples (like the official CUTLASS Python examples) are standalone scripts — they run their whole computation at the module level, not inside a function. We handle both cases.

**The `_resolve` function** is important. Your task paths look like `repos/cutlass/examples/...`, but in the container there is no `/root/gpulab/repos/`. CUTLASS is at `/root/cutlass/` because we cloned it there at image build time. `_resolve` translates between the two.

---

## Part 4: The Modal App

This is where we define the container environment and the remote function.

### Write `src/gpulab/modal_app.py`

```bash
cat > src/gpulab/modal_app.py << 'EOF'
import modal
from rich.console import Console
from rich.panel import Panel

console = Console()

image = (
    modal.Image.from_registry(
        # Start from NVIDIA's official CUDA 13.1.1 development image
        # "devel" means it includes the full CUDA toolkit including nvcc
        # "ubuntu24.04" is the OS
        "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "cmake", "ninja-build")
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "rich",
        # JAX is required by quack-kernels — without it, import fails at runtime
        "jax[cuda12]",
        # CuTe DSL — NVIDIA's Python JIT compiler for CuTe kernels
        # [cu13] variant is for CUDA 13.x
        "nvidia-cutlass-dsl[cu13]",
        # Quack — production CuTe DSL kernels from Dao-AILab
        "quack-kernels[cu13]",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu130",
    )
    # Clone CUTLASS into the image at build time
    # --depth 1 means only the latest commit — much faster than full history
    # This runs once, gets cached, never uploaded again
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /root/cutlass",
    )
    # These two directories mount fresh on every run
    # They are small (just your kernel files and runner code)
    # so this is fast
    .add_local_dir("./kernels", remote_path="/root/gpulab/kernels")
    .add_local_dir("./src",     remote_path="/root/gpulab/src")
)

app = modal.App("gpulab", image=image)


@app.function(gpu="B200", timeout=3600)
def run(task: str, params: dict, extra_flags: list[str] | None = None):
    import subprocess
    import sys

    # Make sure Python can find our gpulab package
    sys.path.insert(0, "/root/gpulab/src")

    from gpulab.runner import dispatch

    # Print GPU info — useful for confirming you got the right hardware
    subprocess.run(["nvidia-smi"], check=True)

    result = dispatch(task, params, extra_flags)

    # Print the result in a nice panel
    console.print(Panel(result, title=f"[green]{task}[/green]", border_style="green"))
    return result
EOF
```

**Key decisions explained:**

**Why `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`?** This is NVIDIA's official base image. The `devel` tag includes `nvcc` and all the CUDA development headers — without it, you cannot compile `.cu` files. The `cudnn` part includes cuDNN for deep learning ops. `13.1.1` matches the CUDA version on the B200 nodes Modal provides.

**Why clone CUTLASS rather than mount it?** CUTLASS is about 36MB of headers, examples, and tools. If we mounted it from our laptop on every run, we would upload 36MB every single time. By cloning it at image build time, we pay this cost once. The image is cached and reused.

**Why `--depth 1`?** Git repos contain the full history of every commit. For CUTLASS, which has years of history, the full repo is much larger than just the latest code. `--depth 1` gives us only the most recent snapshot — all the headers and examples, none of the history.

**Why `add_local_dir` for kernels and src?** These are the files you edit constantly. You write a new kernel, you want to run it immediately without rebuilding the image. `add_local_dir` uploads them fresh on every run. Since they are small (just your `.cu` and `.py` files), this is fast.

---

## Part 5: The CLI

### Write `scripts/run.py`

```bash
cat > scripts/run.py << 'EOF'
import sys
import os

# Make sure Python can find the gpulab package locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpulab.modal_app import run, app


@app.local_entrypoint()
def main(task: str, params: str = "{}", flags: str = ""):
    """
    Run a GPU kernel on Modal.

    task:   path to .cu or .py file
    params: JSON string of parameters for Python kernels (optional)
    flags:  extra nvcc flags (optional), e.g. --flags="-lineinfo"
    """
    import json
    extra_flags = flags.split() if flags else []
    run.remote(task, json.loads(params), extra_flags)
EOF
```

**Why `@app.local_entrypoint()`?** In Modal 1.x, this decorator makes the function a CLI entrypoint. Any typed arguments automatically become CLI options — so `task: str` becomes `--task`, `flags: str` becomes `--flags`, etc. No argparse needed. You run it with `modal run scripts/run.py --task kernels/cuda/foo.cu`.

---

## Part 6: Writing Kernels

Now we have the infrastructure. Let's write kernels for each framework.

### Convention: the `run()` function

For Python kernels (Triton, CuTe DSL, Quack), we follow one convention: the file must contain a `run(**params)` function that does the work and returns a string. The runner calls this function.

For `.cu` kernels, there is no convention — write a standard `int main()`.

### CUDA kernel

```bash
cat > kernels/cuda/vector_add.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vector_add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("cuda vector_add: n=%d  c[0]=%.1f  PASSED\n", n, h_c[0]);

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
EOF
```

### CuTe C++ kernel

CuTe lets you address tensor elements by logical indices instead of computing raw pointer offsets manually.

```bash
cat > kernels/cute/vector_add.cu << 'EOF'
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cute;

__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    // make_tensor wraps a raw pointer with a shape and stride descriptor
    // This lets us index it like tA(i) instead of *(a + i)
    auto tA = make_tensor(make_gmem_ptr(a), make_shape(n));
    auto tB = make_tensor(make_gmem_ptr(b), make_shape(n));
    auto tC = make_tensor(make_gmem_ptr(c), make_shape(n));

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tC(i) = tA(i) + tB(i);
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vector_add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("cute vector_add: n=%d  c[0]=%.1f  PASSED\n", n, h_c[0]);

    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
EOF
```

### CUTLASS kernel

For a first CUTLASS kernel, we use CuTe tensors to write a naive GEMM. This is not the high-performance collective builder GEMM — that requires more setup — but it demonstrates the CUTLASS programming model.

```bash
cat > kernels/cutlass/gemm.cu << 'EOF'
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace cute;

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    auto tA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto tB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto tC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += tA(row, k) * tB(k, col);
    tC(row, col) = acc;
}

int main() {
    const int M = 256, N = 256, K = 256;

    float *h_A = new float[M*K], *h_B = new float[K*N], *h_C = new float[M*N];
    for (int i = 0; i < M*K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < M*N; i++)
        if (fabs(h_C[i] - K) > 1e-3) { pass = false; break; }

    printf("cutlass gemm: M=%d N=%d K=%d  %s\n", M, N, K, pass ? "PASSED" : "FAILED");

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
EOF
```

### Triton kernel

```bash
cat > kernels/triton/vector_add.py << 'EOF'
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
    return f"triton vector_add: n={n}  c[0]={c[0].item():.1f}  PASSED"
EOF
```

### CuTe DSL kernel

CuTe DSL lets you write CuTe kernels in Python. The kernel function is decorated with `@cute.kernel` and the host launch function with `@cute.jit`.

```bash
cat > kernels/cute_dsl/vector_add.py << 'EOF'
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n
    gC[mi, ni] = gA[mi, ni] + gB[mi, ni]


@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    num_threads = 256
    m, n = mA.shape
    kernel = elementwise_add_kernel(mA, mB, mC)
    kernel.launch(
        grid=((m * n) // num_threads, 1, 1),
        block=(num_threads, 1, 1),
    )


M, N = 16384, 8192
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

compiled = cute.compile(elementwise_add, a_, b_, c_)
compiled(a_, b_, c_)

torch.testing.assert_close(c, a + b)
print(f"cute_dsl vector_add: M={M} N={N}  PASSED")
EOF
```

**Note:** This kernel runs its computation at the module level (not inside a `run()` function). That is fine — the runner handles this case gracefully.

### Quack kernel

```bash
mkdir -p kernels/quack
cat > kernels/quack/rmsnorm.py << 'EOF'
import torch
from quack import rmsnorm


def run(**params):
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.ones(N,     device="cuda", dtype=torch.float16)

    out = rmsnorm(x, w)

    # Reference implementation
    rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    ref = (x.float() * rms * w.float()).half()

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    return f"quack rmsnorm: M={M} N={N}  PASSED"
EOF
```

---

## Part 7: Running Everything

Install the package:
```bash
pip install -e .
```

Now run each kernel. The first run will build the image — this takes several minutes. Every run after that is fast.

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
```

To run official CUTLASS examples from the cloned repo:
```bash
modal run scripts/run.py --task repos/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py
```

To pass extra nvcc flags (e.g. for profiling):
```bash
modal run scripts/run.py --task kernels/cuda/vector_add.cu --flags="-lineinfo"
```

---

## Part 8: Adding a New Framework

The pattern is always the same four steps:

**Step 1 — install it in the image.** If it is a pip package, add it to `pip_install()`. If it needs a git clone, add it to `run_commands()`.

**Step 2 — add include paths** to `INCLUDES` in `compiler.py` if it is a C++ framework.

**Step 3 — add backend detection** — one line in `_detect_backend()` in `runner.py`.

**Step 4 — add a compile function** in `compiler.py` if it needs special compile flags.

This triggers one image rebuild. After that, every run with the new framework is fast.

---

## Common Pitfalls and How to Fix Them

### "module 'modal' has no attribute 'Mount'"

You have two Modal installations on your machine — an old one and a new one. Run `which modal` and `python -c "import modal; print(modal.__file__)"` and check that they point to the same environment. If not, activate your conda env and run `pip install --upgrade modal`.

### Files not landing in the container

If you get `FileNotFoundError` and you are sure the file exists locally, the most common cause is a nested `.git` directory. If you `git clone` a repo into your project folder and then mount the parent, Modal will skip files inside the nested `.git`. Fix: delete the `.git` folder from the cloned repo (`rm -rf repos/myrepo/.git`) or, better, do not mount `repos/` at all and clone into the image instead.

### Image rebuild is taking forever

The image only rebuilds when something in the image definition changes — new pip packages, new `run_commands`, etc. Changes to `kernels/` or `src/` do not trigger a rebuild because they are mounted, not baked. If you are waiting a long time, it is because you changed something in `modal_app.py`. Be deliberate about image changes — batch them together so you pay the rebuild cost once.

### "compile failed: fatal error: cute/tensor.hpp: No such file or directory"

Your kernel is in `kernels/cuda/` but it includes CuTe headers. Move it to `kernels/cute/` and the runner will automatically pass the CUTLASS include paths.

### Quack crashes with "No module named 'jax.numpy'"

Quack depends on JAX. Add `"jax[cuda12]"` to your `pip_install()` list and rebuild the image.

### CUTLASS examples fail with "helper.h: No such file or directory"

CUTLASS examples include local headers. The `compile_cutlass` function already handles this by adding `-I{src_dir}` and `-I{CUTLASS_ROOT}/examples/common` — but only if you use the `cutlass` or `repos_cutlass` backend. Make sure your file is in a path that triggers one of those backends.

---

## Final Project Structure

After following this tutorial, your project looks like this:

```
gpulab/
├── kernels/
│   ├── cuda/
│   │   └── vector_add.cu
│   ├── cute/
│   │   └── vector_add.cu
│   ├── cute_dsl/
│   │   └── vector_add.py
│   ├── cutlass/
│   │   └── gemm.cu
│   ├── triton/
│   │   └── vector_add.py
│   └── quack/
│       └── rmsnorm.py
├── repos/
│   └── cutlass/          ← local copy for reading, never uploaded
├── src/
│   └── gpulab/
│       ├── __init__.py
│       ├── compiler.py   ← compile flags per backend
│       ├── runner.py     ← dispatch logic
│       └── modal_app.py  ← container definition
├── scripts/
│   └── run.py            ← CLI entrypoint
├── pyproject.toml
└── .gitignore
```

---
