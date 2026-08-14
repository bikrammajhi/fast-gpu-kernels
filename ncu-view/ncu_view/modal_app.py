"""ncu-view on Modal: profile any CUDA source with ncu and build the report.

    ncu-view profile kernels/cute_dsl/B200 -o out/

mounts the source directory into a Modal container, guesses how to build
and run it, captures a full ncu report (`.ncu-rep` + raw CSV), downloads
the artifacts and renders the HTML report locally. Works on any NVIDIA GPU
Modal offers (--modal-gpu), for any CUDA code: cutlass, cute, raw .cu,
python drivers, Makefile projects.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

from .profile import NCU_DETAIL_SECTIONS, _sec_filename

APP_NAME = "ncu-view-profile"
VOLUME_NAME = "gpulab-cute-dsl-traces"
REMOTE_SRC = "/root/src"


def _rewrite_cutlass_include(cmd: str) -> str:
    """Point -I flags that don't exist in this container at the container's
    own cutlass/cute headers (the client embeds its local path)."""
    if "-I" not in cmd:
        return cmd
    flags = _container_cutlass_flags()
    if not flags:
        return cmd
    parts = re.split(r"\s+&&\s+", cmd)
    parts[0] = re.sub(r"\s+-I\S+", "", parts[0]) + flags
    return " && ".join(parts)


def _container_cutlass_flags() -> str:
    root = _container_cutlass_include()
    if not root:
        return ""
    flags = f" -I{root}"
    util = f"{Path(root).parent}/tools/util/include"
    if Path(util).is_dir():
        flags += f" -I{util}"
    return flags


def _container_cutlass_include() -> str | None:
    if (Path("/opt/cutlass/include/cutlass/cluster_launch.hpp").exists()):
        return "/opt/cutlass/include"
    import glob
    import site
    for sp in site.getsitepackages():
        for d in glob.glob(sp + "/**/include", recursive=True):
            if (Path(d) / "cutlass" / "cluster_launch.hpp").exists() and \
                    (Path(d) / "cute").exists():
                return d
    return None
REMOTE_OUT = "/root/out"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("cuda-nsight-compute-13-0", "git", "cmake", "ninja-build", "g++")
    .run_commands([
        "git clone --depth 1 --branch v4.2.2 "
        "https://github.com/NVIDIA/cutlass /opt/cutlass"
    ])
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl", "nvidia-cutlass", "ncu-report")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _run_ncu_sections(rep: str, run_id: str, cwd: str | None = None) -> dict[str, bytes]:
    """NVIDIA's own detailed section tables via `ncu --import --section`.

    `--page details` shows the full tables (a bare `--csv` skips the
    details part of a section); if a section still yields the
    "No metrics to show" warning it is retried without the flag.
    """
    out = {}
    for sid in NCU_DETAIL_SECTIONS:
        for flag in (["--page", "details"], []):
            try:
                r = subprocess.run(
                    ["ncu", "--import", rep, "--section", sid, *flag, "--csv"],
                    cwd=cwd, check=True, capture_output=True)
                if b"WARNING" in r.stdout:
                    raise RuntimeError("no metrics for this part")
                out[_sec_filename(run_id, sid)] = r.stdout
                break
            except Exception:
                continue
    return out


def _cublas_bench_source(precision: str, shape: int, iters: int = 5,
                         warmup: int = 5) -> str:
    typ, cublas_t = (("half", "CUDA_R_16F") if precision == "fp16"
                     else ("__nv_bfloat16", "CUDA_R_16BF"))
    return f"""#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
int main() {{
    const int N = {shape};
    {typ} *A, *B, *C;
    cudaMalloc(&A, (size_t)N * N * sizeof({typ}));
    cudaMalloc(&B, (size_t)N * N * sizeof({typ}));
    cudaMalloc(&C, (size_t)N * N * sizeof({typ}));
    cudaMemset(C, 0, (size_t)N * N * sizeof({typ}));
    cublasHandle_t h;
    cublasCreate(&h);
    {typ} alpha = 1, beta = 0;
    for (int i = 0; i < {warmup}; ++i)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, A, {cublas_t}, N, B, {cublas_t}, N,
                     &beta, C, {cublas_t}, N, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    for (int i = 0; i < {iters}; ++i)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, A, {cublas_t}, N, B, {cublas_t}, N,
                     &beta, C, {cublas_t}, N, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C);
    printf("cublas gemm done\\n");
    return 0;
}}
"""


def _profile_source_body(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    """Run `run_cmd` under ncu (full set), then re-export raw CSV + sections.

    `launch_skip` lands the capture past the app's own warm-up loop so
    steady-state launches are timed (a cold single launch reads ~1.5x slower
    on data-center parts). Cascades: (skip,count) → (1,1) → (0,1) so
    single-launch apps still profile. `clock_control` (default boost) is the
    ncu --clock-control setting: ncu's own default 'base' locks the GPU to
    base clock and understates throughput; boost gives the reproducible peak
    and none lets the app's warm-up drive clocks. If the host refuses boost,
    the capture retries with none and a note rides back under "__note__".
    `bench` (dict with precision/shape) additionally profiles a cuBLAS GEMM
    in the same run under identical flags, so the report series compares
    your kernel against cuBLAS at the same clock.
    """
    subprocess.run(["mkdir", "-p", REMOTE_SRC], check=True)
    for rel, data in files.items():
        p = Path(REMOTE_SRC) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    subprocess.run(["mkdir", "-p", REMOTE_OUT], check=True)
    rep = f"{REMOTE_OUT}/{run_id}.ncu-rep"
    env = None
    cap = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    nvcc_arch = {"7.5": "sm_75", "8.0": "sm_80", "8.6": "sm_86",
                 "8.9": "sm_89", "9.0": "sm_90a", "10.0": "sm_100a",
                 "10.3": "sm_103a", "11.0": "sm_110a", "12.0": "sm_120a"}.get(cap)
    if nvcc_arch:
        run_cmd = run_cmd.replace("-arch=native", f"-arch={nvcc_arch}")
        dsl_arch = {"8.0": "sm_80", "8.9": "sm_89", "9.0": "sm_90a",
                    "10.0": "sm_100a", "10.3": "sm_103a", "11.0": "sm_110a",
                    "12.0": "sm_120a"}.get(cap)
        if dsl_arch and os.environ.get("CUTE_DSL_ARCH") is None:
            env = {**os.environ, "CUTE_DSL_ARCH": dsl_arch}
    run_cmd = _rewrite_cutlass_include(run_cmd)

    def _capture_into(rrep: str, cmd: str, skip: int, count: int,
                      cc: str) -> subprocess.CompletedProcess:
        return subprocess.run(["ncu", "--set", "full", "--clock-control", cc,
                               "--launch-skip", str(skip), "--launch-count",
                               str(count), "-o", rrep, "sh", "-c", cmd],
                              cwd=REMOTE_SRC, env=env, capture_output=True)

    def _select(rrep: str, cmd: str, skip: int, count: int
                ) -> tuple[subprocess.CompletedProcess, str | None]:
        note = None
        r = _capture_into(rrep, cmd, skip, count, clock_control)
        if r.returncode != 0 and clock_control == "boost":
            r2 = _capture_into(rrep, cmd, skip, count, "none")
            if r2.returncode == 0 and os.path.exists(rrep):
                r, note = r2, ("ncu refused --clock-control boost on this "
                               "host; captured at natural clocks")
        if r.returncode == 0 and not os.path.exists(rrep):
            r = _capture_into(rrep, cmd, 1, 1, clock_control)
        if r.returncode == 0 and not os.path.exists(rrep):
            r = _capture_into(rrep, cmd, 0, 1, clock_control)
        return r, note

    ncu, note = _select(rep, run_cmd, launch_skip, launch_count)
    if ncu.returncode != 0 or not os.path.exists(rep):
        return {"error": f"ncu rc={ncu.returncode}\n"
                         f"{ncu.stdout.decode(errors='replace')[-4000:]}\n"
                         f"{ncu.stderr.decode(errors='replace')[-4000:]}"}
    raw = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"],
                         cwd=REMOTE_SRC, capture_output=True)
    if raw.returncode != 0:
        return {"error": f"ncu --import raw rc={raw.returncode}\n"
                         f"{raw.stdout.decode(errors='replace')[-3000:]}\n"
                         f"{raw.stderr.decode(errors='replace')[-3000:]}"}
    files = {f"{run_id}.ncu-rep": Path(rep).read_bytes(),
             f"{run_id}.raw.csv": raw.stdout}
    files.update(_run_ncu_sections(rep, run_id, cwd=REMOTE_SRC))
    if bench:
        cid = f"{run_id}-cublas"
        crep = f"{REMOTE_OUT}/{cid}.ncu-rep"
        arch = nvcc_arch or "native"
        bench_cmd = (f"nvcc -arch={arch} -O3 -lcublas "
                     f"-o /tmp/ncu-view-cublas-bench "
                     f"ncu-view-cublas-bench.cu && "
                     f"/tmp/ncu-view-cublas-bench")
        bncu, bnote = _select(crep, bench_cmd, launch_skip, launch_count)
        if bncu.returncode != 0 or not os.path.exists(crep):
            note = (note + "\n" if note else "") + \
                   f"cuBLAS baseline failed: rc={bncu.returncode}\n" + \
                   bncu.stderr.decode(errors='replace')[-2000:]
        else:
            braw = subprocess.run(["ncu", "--import", crep, "--page", "raw",
                                   "--csv"], cwd=REMOTE_SRC,
                                  capture_output=True)
            if braw.returncode != 0:
                note = (note + "\n" if note else "") + \
                       f"cuBLAS baseline raw export failed: rc={braw.returncode}"
            else:
                files[f"{cid}.ncu-rep"] = Path(crep).read_bytes()
                files[f"{cid}.raw.csv"] = braw.stdout
                files.update(_run_ncu_sections(crep, cid, cwd=REMOTE_SRC))
    if note:
        files["__note__"] = note.encode()
    return files


# One module-level entrypoint per Modal GPU type (modal serializes functions
# by importing the module and looking up the qualname, so closures/factories
# won't hydrate). All delegate to the shared _profile_source_body.


@app.function(image=image, gpu="T4", timeout=3600)
def _profile_source_t4(run_cmd: str, run_id: str,
                       files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="L4", timeout=3600)
def _profile_source_l4(run_cmd: str, run_id: str,
                       files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A10", timeout=3600)
def _profile_source_a10(run_cmd: str, run_id: str,
                        files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="L40S", timeout=3600)
def _profile_source_l40s(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100", timeout=3600)
def _profile_source_a100(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100-40GB", timeout=3600)
def _profile_source_a100_40gb(run_cmd: str, run_id: str,
                              files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def _profile_source_a100_80gb(run_cmd: str, run_id: str,
                              files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="RTX-PRO-6000", timeout=3600)
def _profile_source_rtx_pro_6000(run_cmd: str, run_id: str,
                                 files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="H100", timeout=3600)
def _profile_source_h100(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="H200", timeout=3600)
def _profile_source_h200(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B200", timeout=3600)
def _profile_source_b200(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B200+", timeout=3600)
def _profile_source_b200p(run_cmd: str, run_id: str,
                          files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B300", timeout=3600)
def _profile_source_b300(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int = 10,
                         launch_count: int = 1,
                         clock_control: str = "boost",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


PROFILE_SOURCES = {
    "T4": _profile_source_t4, "L4": _profile_source_l4,
    "A10": _profile_source_a10, "L40S": _profile_source_l40s,
    "A100": _profile_source_a100, "A100-40GB": _profile_source_a100_40gb,
    "A100-80GB": _profile_source_a100_80gb,
    "RTX-PRO-6000": _profile_source_rtx_pro_6000,
    "H100": _profile_source_h100, "H200": _profile_source_h200,
    "B200": _profile_source_b200, "B200+": _profile_source_b200p,
    "B300": _profile_source_b300,
}


@app.function(image=image, cpu=2, timeout=900,
              volumes={REMOTE_OUT: volume})
def _extract_sections(rep_key: str, run_id: str) -> list[str]:
    """Re-extract NVIDIA's detailed sections from a report already on the
    volume (CPU-only, no GPU): `ncu --import rep --section <X> --csv`.
    Writes each table to the volume under ncu-golden/, returns what landed.
    """
    local_rep = f"/tmp/{run_id}.ncu-rep"
    with open(local_rep, "wb") as f:
        f.write(b"".join(volume.read_file(rep_key)))
    artifacts = _run_ncu_sections(local_rep, run_id)
    subprocess.run(["mkdir", "-p", f"{REMOTE_OUT}/ncu-golden"], check=True)
    for name, data in artifacts.items():
        with open(f"{REMOTE_OUT}/ncu-golden/{name}", "wb") as f:
            f.write(data)
    print(f"wrote {len(artifacts)} sections: {sorted(artifacts)}", flush=True)
    return sorted(artifacts)


def extract_sections(rep_key: str, run_id: str,
                     dest_dir: str | None = None) -> list[str]:
    """Client side: extract detailed sections (volume-to-volume)."""
    names = _extract_sections.remote(rep_key, run_id)
    if dest_dir and names:
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            (d / name).write_bytes(
                b"".join(volume.read_file(f"ncu-golden/{name}")))
    return names


def profile_on_modal(source_dir: Path, run_cmd: str, run_id: str,
                     gpu: str, timeout: int,
                     launch_skip: int = 10,
                     launch_count: int = 1,
                     clock_control: str = "boost",
                     bench: dict | None = None,
                     app_iters: int | None = None,
                     app_warmup: int | None = None) -> dict[str, bytes]:
    """Run the profile on Modal and return {filename: bytes} artifacts."""
    fn = PROFILE_SOURCES.get(gpu, PROFILE_SOURCES["H100"])
    if gpu not in PROFILE_SOURCES:
        print(f"note: --modal-gpu {gpu} not in the Modal catalog "
              f"({sorted(PROFILE_SOURCES)}); falling back to H100.")
    files = {}
    if source_dir.is_file():
        files[source_dir.name] = source_dir.read_bytes()
        for p in sorted(source_dir.parent.iterdir()):
            if p.is_file() and p.suffix in (".cu", ".cuh", ".h", ".hpp",
                                            ".c", ".cpp", ".py"):
                files[p.name] = p.read_bytes()
    else:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(source_dir))] = p.read_bytes()
    if app_iters is not None and app_iters > 0:
        warmup = app_warmup if app_warmup is not None else 1
        from .profile import _rewrite_launch_counts
        rewritten = False
        for rel in list(files):
            if rel.endswith(".py"):
                new = _rewrite_launch_counts(
                    files[rel].decode(errors="replace"), app_iters, warmup)
                if new != files[rel].decode(errors="replace"):
                    files[rel] = new.encode()
                    rewritten = True
        if rewritten:
            print(f"note: profiling {app_iters} timed launch(s), "
                  f"{warmup} warm-up (app launch counts rewritten)")
        else:
            print("note: --app-iters has no effect here: no "
                  "warmup_iterations=/iterations= kwargs in the driver; "
                  "app keeps its own launches")
    if bench:
        files["ncu-view-cublas-bench.cu"] = _cublas_bench_source(
            bench.get("precision", "fp16"), int(bench.get("shape", 8192)),
            int(bench.get("iters", 5)), int(bench.get("warmup", 5))
        ).encode()
    with app.run():
        return fn.remote(run_cmd, run_id, files, launch_skip, launch_count,
                         clock_control, bench)
