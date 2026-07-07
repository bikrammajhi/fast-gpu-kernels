"""gpulab — GPU kernel benchmark runner on Modal.

Usage:
    modal run scripts/run.py --task kernels/cuda/A100/benchmark.cu --gpu A100
    modal run scripts/run.py --task kernels/cute/H100/benchmark.cu --gpu H100
"""

import subprocess
import os
import time
from pathlib import Path

import modal
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = "/root/gpulab"
CUTLASS_ROOT = "/root/cutlass"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# GPU → arch mapping
# ---------------------------------------------------------------------------
GPU_ARCH = {
    "A100":        ["-arch=sm_80",  "-gencode", "arch=compute_80,code=sm_80"],
    "H100":        ["-arch=sm_90",  "-gencode", "arch=compute_90,code=sm_90"],
    "H200":        ["-arch=sm_90",  "-gencode", "arch=compute_90,code=sm_90"],
    "B200":        ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "B100":        ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "L40S":        ["-arch=sm_89",  "-gencode", "arch=compute_89,code=sm_89"],
    "L4":          ["-arch=sm_89",  "-gencode", "arch=compute_89,code=sm_89"],
    "A10":         ["-arch=sm_86",  "-gencode", "arch=compute_86,code=sm_86"],
    "T4":          ["-arch=sm_75",  "-gencode", "arch=compute_75,code=sm_75"],
    "RTXPRO6000":  ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
}


def _get_arch(gpu: str) -> list[str]:
    return GPU_ARCH.get(gpu.upper().replace(" ", ""), GPU_ARCH["H100"])


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
def _compile(src: str, binary: str, gpu: str, extra_flags: list[str], cutlass: bool = False) -> None:
    arch = _get_arch(gpu)
    includes = []
    flags = []
    if cutlass:
        includes = [
            f"-I{CUTLASS_ROOT}/include",
            f"-I{CUTLASS_ROOT}/tools/util/include",
            f"-I{Path(src).parent}",
            f"-I{CUTLASS_ROOT}/examples/common",
        ]
        flags = ["-std=c++17"]
    cmd = ["nvcc", "-O3", *includes, *arch, "-lcublas", *flags, *extra_flags, "-o", binary, src]
    console.log(f"[dim]compile[/dim]  {src} [dim]({gpu})[/dim]")
    t0 = time.perf_counter()
    cc = subprocess.run(cmd, capture_output=True, text=True)
    if cc.returncode != 0:
        raise RuntimeError(f"compile failed:\n{cc.stderr}")
    console.log(f"[dim]done[/dim]     {(time.perf_counter() - t0) * 1000:.0f}ms")


def _run_binary(binary: str) -> str:
    out = subprocess.run([binary], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runtime error:\n{out.stderr}")
    return out.stdout


def dispatch(task: str, gpu: str, extra_flags: list[str] | None = None) -> str:
    extra_flags = extra_flags or []
    src = os.path.join(ROOT, task)
    binary = src.removesuffix(".cu")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found")
    cutlass = "cute" in task or "cutlass" in task
    _compile(src, binary, gpu, extra_flags, cutlass=cutlass)
    return _run_binary(binary)


# ---------------------------------------------------------------------------
# Modal app — change gpu= to run on a different GPU by default
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("rich")
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /root/cutlass",
    )
    .add_local_dir(str(PROJECT_ROOT / "kernels"), remote_path="/root/gpulab/kernels")
)

app = modal.App("gpulab", image=image)


@app.function(gpu="H100", timeout=3600)
def run(task: str, gpu: str = "H100", extra_flags: list[str] | None = None):
    subprocess.run(["nvidia-smi"], check=True)
    result = dispatch(task, gpu, extra_flags)
    console.print(Panel(result, title=f"[green]{task}[/green] on [bold]{gpu}[/bold]", border_style="green"))
    return result


@app.local_entrypoint()
def main(task: str, gpu: str = "H100", flags: str = ""):
    extra_flags = flags.split() if flags else []
    run.remote(task, gpu, extra_flags)
