import modal
from rich.console import Console
from rich.panel import Panel

console = Console()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "cmake", "ninja-build")
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "rich",
        "jax[cuda12]",
        "nvidia-cutlass-dsl[cu13]",
        "quack-kernels[cu13]",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu130",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /root/cutlass",
    )
    .add_local_dir("./kernels", remote_path="/root/gpulab/kernels")
    .add_local_dir("./src",     remote_path="/root/gpulab/src")
)

app = modal.App("gpulab", image=image)


@app.function(gpu="A100", timeout=3600)
def run(task: str, params: dict, extra_flags: list[str] | None = None):
    import subprocess
    import sys
    sys.path.insert(0, "/root/gpulab/src")

    from gpulab.runner import dispatch

    subprocess.run(["nvidia-smi"], check=True)
    result = dispatch(task, params, extra_flags)
    console.print(Panel(result, title=f"[green]{task}[/green]", border_style="green"))
    return result
