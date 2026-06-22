import subprocess
import time
from pathlib import Path
from rich.console import Console

console = Console()

ARCH         = ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"]
ARCH_A100    = ["-arch=sm_80", "-gencode", "arch=compute_80,code=sm_80"]
CUTLASS_ROOT = "/root/cutlass"

INCLUDES = {
    "cuda": [],
    "cutlass": [
        f"-I{CUTLASS_ROOT}/include",
        f"-I{CUTLASS_ROOT}/tools/util/include",
    ],
}

CXX17_FLAGS = ["-std=c++17"]


def _detect_arch(src: str) -> list[str]:
    if "A100" in src or "a100" in src:
        return ARCH_A100
    return ARCH


def _nvcc(src: str, binary: str, includes: list[str], extra_flags: list[str]) -> None:
    arch = _detect_arch(src)
    cmd = ["nvcc", "-O3", *includes, *arch, "-lcublas", *extra_flags, "-o", binary, src]
    console.log(f"[dim]compile[/dim]  {src}")
    t0 = time.perf_counter()
    cc = subprocess.run(cmd, capture_output=True, text=True)
    if cc.returncode != 0:
        raise RuntimeError(f"compile failed:\n{cc.stderr}")
    console.log(f"[dim]done[/dim]     {(time.perf_counter()-t0)*1000:.0f}ms")


def compile_cuda(src: str, binary: str, extra_flags: list[str]) -> None:
    _nvcc(src, binary, INCLUDES["cuda"], extra_flags)


def compile_cutlass(src: str, binary: str, extra_flags: list[str]) -> None:
    src_dir   = str(Path(src).parent)
    extra_inc = [f"-I{src_dir}", f"-I{CUTLASS_ROOT}/examples/common"]
    _nvcc(src, binary, [*INCLUDES["cutlass"], *extra_inc], [*CXX17_FLAGS, *extra_flags])
