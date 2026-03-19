import subprocess
import importlib
import sys
import os
from pathlib import Path
from rich.console import Console
from gpulab.compiler import compile_cuda, compile_cutlass

console = Console()

ROOT         = "/root/gpulab"
CUTLASS_ROOT = "/root/cutlass"


def _detect_backend(task: str) -> str:
    parts = set(Path(task).parts)
    if "repos"    in parts: return "repos_cutlass"
    if "cutlass"  in parts: return "cutlass"
    if "cute"     in parts: return "cutlass"
    if "cute_dsl" in parts: return "cute_dsl"
    if "triton"   in parts: return "triton"
    if "quack"    in parts: return "quack"
    return "cuda"


def _run_binary(binary: str) -> str:
    out = subprocess.run([binary], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runtime error:\n{out.stderr}")
    return out.stdout


def _run_python(task: str, params: dict) -> str:
    sys.path.insert(0, ROOT)
    module = task.removesuffix(".py").replace("/", ".")
    console.log(f"[dim]import[/dim]   {module}")
    mod = importlib.import_module(module)
    if hasattr(mod, "run"):
        return str(mod.run(**params))
    return f"{task} completed"


def _run_script(script: str) -> str:
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
    parts = Path(task).parts
    if parts[0] == "repos":
        return os.path.join(CUTLASS_ROOT, *parts[2:])
    return os.path.join(ROOT, task)


def dispatch(task: str, params: dict, extra_flags: list[str] | None = None) -> str:
    extra_flags = extra_flags or []
    backend = _detect_backend(task)
    console.log(f"backend: [bold]{backend}[/bold]")

    if task.endswith(".cu"):
        src    = _resolve(task)
        binary = src.removesuffix(".cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} not found")
        if backend in ("cutlass", "repos_cutlass"): compile_cutlass(src, binary, extra_flags)
        else:                                       compile_cuda(src, binary, extra_flags)
        return _run_binary(binary)

    if backend == "repos_cutlass":
        return _run_script(_resolve(task))

    return _run_python(task, params)
