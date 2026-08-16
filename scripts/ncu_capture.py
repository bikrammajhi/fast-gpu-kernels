"""gpulab — plain-ncu report capture on Modal.

Runs a kernel under NVIDIA Nsight Compute with the full metric set and
writes ONLY a `.ncu-rep` report file to the `gpulab-cute-dsl-traces` volume
(`captures/<name>.ncu-rep`). Download just that file with:

    modal volume get gpulab-cute-dsl-traces captures/<name>.ncu-rep <local-dest>

then open it in your local Nsight Compute GUI (`ncu-ui <file>.ncu-rep`) —
no local GPU needed.

Usage:
    modal run scripts/ncu_capture.py --src kernels/cute_dsl/B200/matmul_v5.py --gpu B200
    modal run scripts/ncu_capture.py --src kernels/cuda/A100/benchmark.cu --gpu A100-40GB --clock-control boost
    modal run scripts/ncu_capture.py --src kernels/cute/H100/matmul_v4.cu --gpu H100 --launch-skip 1 --launch-count 3
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT = "/root/gpulab"
CUTLASS_ROOT = "/root/cutlass"

GPU_ARCH = {
    "A100": ["-arch=sm_80", "-gencode", "arch=compute_80,code=sm_80"],
    "A100-40GB": ["-arch=sm_80", "-gencode", "arch=compute_80,code=sm_80"],
    "A100-80GB": ["-arch=sm_80", "-gencode", "arch=compute_80,code=sm_80"],
    "H100": ["-arch=sm_90", "-gencode", "arch=compute_90,code=sm_90"],
    "H200": ["-arch=sm_90", "-gencode", "arch=compute_90,code=sm_90"],
    "B200": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "B200+": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "B100": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "L40S": ["-arch=sm_89", "-gencode", "arch=compute_89,code=sm_89"],
    "L4": ["-arch=sm_89", "-gencode", "arch=compute_89,code=sm_89"],
    "A10": ["-arch=sm_86", "-gencode", "arch=compute_86,code=sm_86"],
    "T4": ["-arch=sm_75", "-gencode", "arch=compute_75,code=sm_75"],
    "RTXPRO6000": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
    "RTX-PRO-6000": ["-arch=sm_100", "-gencode", "arch=compute_100,code=sm_100"],
}

DSL_ARCH = {
    "A100": "sm_80",
    "A100-40GB": "sm_80",
    "A100-80GB": "sm_80",
    "H100": "sm_90a",
    "H200": "sm_90a",
    "B200": "sm_100a",
    "B200+": "sm_100a",
    "B100": "sm_100a",
    "L40S": "sm_89",
    "L4": "sm_89",
    "A10": "sm_86",
    "T4": "sm_75",
    "RTXPRO6000": "sm_100a",
    "RTX-PRO-6000": "sm_100a",
}


def _build_run_cmd(src: str, gpu: str) -> list[str]:
    if src.endswith(".py"):
        return ["python3", src]
    if src.endswith(".cu"):
        binary = src.removesuffix(".cu")
        cutlass = "cute" in src or "cutlass" in src
        includes = []
        if cutlass:
            includes = [
                f"-I{CUTLASS_ROOT}/include",
                f"-I{CUTLASS_ROOT}/tools/util/include",
                f"-I{Path(src).parent}",
                f"-I{CUTLASS_ROOT}/examples/common",
            ]
        cmd = ["nvcc", "-O3", *includes, *GPU_ARCH.get(gpu, GPU_ARCH["H100"]),
               "-lcublas", "-std=c++17", "-o", binary, src]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return [binary]
    raise RuntimeError(f"unsupported source: {src} (use --build-cmd)")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git")
    .apt_install("cuda-nsight-compute-13-0")
    .pip_install("rich")
    .pip_install("ncu-report")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl[cu13]")
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /root/cutlass",
    )
    .add_local_dir(str(PROJECT_ROOT / "kernels"), remote_path="/root/gpulab/kernels")
)

volume = modal.Volume.from_name("gpulab-cute-dsl-traces", create_if_missing=True)
app = modal.App("gpulab-ncu-capture", image=image)


@app.function(gpu="H100", timeout=3600, volumes={"/out": volume})
def capture(
    src: str,
    gpu: str = "H100",
    out: str | None = None,
    clock_control: str = "base",
    launch_skip: int = 1,
    launch_count: int = 1,
    build_cmd: str | None = None,
) -> str:
    full = os.path.join(ROOT, src)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{full} not found")
    run_cmd = (build_cmd or " ").split() if build_cmd else _build_run_cmd(full, gpu)
    name = Path(src).name.removesuffix(".cu").removesuffix(".py")
    key = out or f"captures/{name}.ncu-rep"
    rep = f"/out/{key}"
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    env = dict(os.environ)
    env["CUTE_DSL_ARCH"] = DSL_ARCH.get(gpu, "sm_90a")
    cmd = [
        "ncu",
        "--set", "full",
        "--warp-sampling-interval", "auto",
        "--clock-control", clock_control,
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "-o", rep,
        *run_cmd,
    ]
    subprocess.run(cmd, check=True, env=env, timeout=3000)
    return key


@app.local_entrypoint()
def main(
    src: str,
    gpu: str = "H100",
    out: str | None = None,
    clock_control: str = "base",
    launch_skip: int = 1,
    launch_count: int = 1,
    build_cmd: str | None = None,
):
    t0 = time.perf_counter()
    key = capture.remote(src, gpu, out, clock_control, launch_skip, launch_count, build_cmd)
    print(f"[{time.perf_counter() - t0:6.1f}s] captured {key}")
    print(f"download the report only:  modal volume get gpulab-cute-dsl-traces {key} <local-dest>")
    print(f"open it locally:           ncu-ui <local-dest>/<file>.ncu-rep")