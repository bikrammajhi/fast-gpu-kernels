"""Run CuTe DSL kernel benchmarks on Modal."""
import os
import subprocess
from pathlib import Path

import modal
from rich.console import Console
from rich.panel import Panel

console = Console()

ROOT = "/root/gpulab"
KERNEL_TYPE = "cute_dsl"

# Default GPU when neither `--gpu` nor a GPU-named task folder is given.
GPU_FALLBACK = "B200"

# Known GPU folder names. Used only as a fallback when `--gpu` is not passed.
KNOWN_GPUS = {
    "H100", "H100-80GB", "H200", "A100", "A100-40GB", "A100-80GB",
    "B200", "B100", "L40S", "L4", "A10", "A10G", "T4", "RTXPRO6000",
}


def get_cute_dsl_arch(gpu_name: str) -> str:
    """Map Modal GPU names to the correct CuTe DSL architecture string."""
    # Normalize the GPU name for easy matching
    gpu = gpu_name.upper().replace("-", "").replace(" ", "")
    
    if gpu in ("B200", "B100"):
        return "sm_100a"  # Blackwell
    if gpu in ("H100", "H10080GB", "H200"):
        return "sm_90a"   # Hopper
    if gpu in ("A100", "A10040GB", "A10080GB"):
        return "sm_80"    # Ampere
    if gpu in ("L40S", "L4", "A10", "A10G", "RTXPRO6000"):
        return "sm_89"    # Ada / Lovelace
    if gpu == "T4":
        return "sm_75"    # Turing
        
    # Fallback to Hopper if the GPU is unrecognized
    return "sm_90a"


def resolve_gpu(task: str, gpu: str | None) -> str:
    # Explicit `--gpu` flag always wins and accepts any Modal GPU architecture.
    if gpu:
        return gpu
    head = task.split("/", 1)[0]
    if head in KNOWN_GPUS:
        return head
    return GPU_FALLBACK


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("rich")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl[cu13]")
    .add_local_dir(str(Path(__file__).resolve().parent.parent.parent / "kernels"), remote_path="/root/gpulab/kernels")
)

app = modal.App(f"gpulab-{KERNEL_TYPE}", image=image)


@app.cls(gpu=GPU_FALLBACK, timeout=3600)
class Runner:
    @modal.method()
    def run(self, task: str, gpu: str = GPU_FALLBACK, args: str = ""):
        src = os.path.join(ROOT, "kernels", KERNEL_TYPE, task)
        if not src.endswith(".py"):
            raise ValueError(f"{KERNEL_TYPE} runner only supports .py tasks")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} not found")

        gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        console.print(Panel(gpu_info.stdout, title="[bold]GPU Info[/bold]", border_style="cyan"))

        env = os.environ.copy()
        # FIX: Dynamically set the architecture based on the resolved GPU
        env["CUTE_DSL_ARCH"] = get_cute_dsl_arch(gpu)

        console.log(f"[dim]run python[/dim]  {src} [dim]({gpu} -> {env['CUTE_DSL_ARCH']})[/dim]")
        cmd = ["python3", src] + (args.split() if args else [])
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if out.returncode != 0:
            console.print(Panel(out.stderr, title="[bold red]Error[/bold red]", border_style="red"))
            raise RuntimeError(f"runtime error:\n{out.stderr}")
        console.print(Panel(out.stdout, title=f"[green]{task}[/green] on [bold]{gpu}[/bold]", border_style="green"))
        return out.stdout


@app.local_entrypoint()
def main(task: str, gpu: str = None, args: str = ""):
    resolved_gpu = resolve_gpu(task, gpu)
    Runner.with_options(gpu=resolved_gpu)().run.remote(task, resolved_gpu, args)