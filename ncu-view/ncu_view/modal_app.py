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
import sys
import threading
import time
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


def _progress(run_id: str, line: str) -> None:
    """Append a progress line for the client's watcher (volume-mounted)."""
    try:
        with open(f"{REMOTE_OUT}/{run_id}.progress", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


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


def _cublas_bench_source(precision: str, shape: int) -> str:
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
    for (int i = 0; i < 5; ++i)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, A, {cublas_t}, N, B, {cublas_t}, N,
                     &beta, C, {cublas_t}, N, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    for (int i = 0; i < 20; ++i)
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
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    """Run `run_cmd` under ncu (full set), then re-export raw CSV + sections.

    `launch_skip` defaults to None: skip the warmup launch and profile ONE
    kernel (launch 1) — works for any app with no launch-order knowledge.
    Pass an explicit skip to land on a specific launch, or a launch_count > 1
    to average over steady-state launches. `clock_control` (default base) is
    the ncu --clock-control setting: ncu's own default 'base' locks the GPU
    to base clock; boost gives the reproducible
    peak and none lets the app's warm-up drive clocks. If the host refuses
    boost, the capture retries with none and a note rides back under
    "__note__". `bench` (dict with precision/shape) additionally profiles a
    cuBLAS GEMM in the same run under identical flags, so the report series
    compares your kernel against cuBLAS at the same clock.
    """
    subprocess.run(["mkdir", "-p", REMOTE_SRC], check=True)
    for rel, data in files.items():
        p = Path(REMOTE_SRC) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    subprocess.run(["mkdir", "-p", REMOTE_OUT], check=True)
    rep = f"{REMOTE_OUT}/{run_id}.ncu-rep"
    env = None
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if smi.returncode == 0:
        for ln in smi.stdout.splitlines():
            if ln.strip():
                _progress(run_id, "smi:" + ln.rstrip())
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
        proc = subprocess.Popen(
            ["ncu", "--set", "full", "--warp-sampling-interval", "auto",
             "--clock-control", cc,
             "--launch-skip", str(skip), "--launch-count",
             str(count), "-o", rrep, "sh", "-c", cmd],
            cwd=REMOTE_SRC, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        chunks: list[bytes] = []

        def _fwd(stream):
            for line in iter(stream.readline, b""):
                chunks.append(line)
                if b"%" in line or b"PROF" in line or b"Profiling" in line:
                    _progress(run_id, "msg:" +
                              line.decode(errors="replace").strip())
            stream.close()

        t1 = threading.Thread(target=_fwd, args=(proc.stdout,), daemon=True)
        t2 = threading.Thread(target=_fwd, args=(proc.stderr,), daemon=True)
        t1.start()
        t2.start()
        proc.wait()
        t1.join()
        t2.join()
        return subprocess.CompletedProcess(proc.args, proc.returncode,
                                           b"", b"".join(chunks))

    def _select(rrep: str, cmd: str, skip: int | None, count: int
                ) -> tuple[subprocess.CompletedProcess, str | None]:
        note = None
        if skip is None:
            skip, count = 1, 1  # one kernel: skip warmup, capture launch 1
        r = _capture_into(rrep, cmd, skip, count, clock_control)
        if r.returncode != 0 and clock_control == "boost":
            r2 = _capture_into(rrep, cmd, skip, count, "none")
            if r2.returncode == 0 and os.path.exists(rrep):
                r, note = r2, ("ncu refused --clock-control boost on this "
                               "host; captured at natural clocks")
        return r, note

    _progress(run_id, "step:ncu capture "
              f"(skip {1 if launch_skip is None else launch_skip}, "
              f"count {1 if launch_skip is None else launch_count}, "
              f"clock {clock_control})")
    ncu, note = _select(rep, run_cmd, launch_skip, launch_count)
    if ncu.returncode != 0 or not os.path.exists(rep):
        return {"error": f"ncu rc={ncu.returncode}\n"
                         f"{ncu.stdout.decode(errors='replace')[-4000:]}\n"
                         f"{ncu.stderr.decode(errors='replace')[-4000:]}"}
    _progress(run_id, "step:export raw CSV")
    raw = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"],
                         cwd=REMOTE_SRC, capture_output=True)
    if raw.returncode != 0:
        return {"error": f"ncu --import raw rc={raw.returncode}\n"
                         f"{raw.stdout.decode(errors='replace')[-3000:]}\n"
                         f"{raw.stderr.decode(errors='replace')[-3000:]}"}
    _progress(run_id, "step:export NVIDIA detail sections "
              f"({len(NCU_DETAIL_SECTIONS)})")
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
        _progress(run_id, "step:cuBLAS baseline (same clock)")
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
    _progress(run_id, "step:done — returning artifacts")
    return files


# One module-level entrypoint per Modal GPU type (modal serializes functions
# by importing the module and looking up the qualname, so closures/factories
# won't hydrate). All delegate to the shared _profile_source_body.


@app.function(image=image, gpu="T4", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_t4(run_cmd: str, run_id: str,
                       files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="L4", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_l4(run_cmd: str, run_id: str,
                       files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A10", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_a10(run_cmd: str, run_id: str,
                        files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="L40S", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_l40s(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_a100(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100-40GB", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_a100_40gb(run_cmd: str, run_id: str,
                              files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="A100-80GB", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_a100_80gb(run_cmd: str, run_id: str,
                              files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="RTX-PRO-6000", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_rtx_pro_6000(run_cmd: str, run_id: str,
                                 files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="H100", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_h100(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="H200", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_h200(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B200", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_b200(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B200+", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_b200p(run_cmd: str, run_id: str,
                          files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
                         bench: dict | None = None) -> dict[str, bytes]:
    return _profile_source_body(run_cmd, run_id, files, launch_skip,
                                launch_count, clock_control, bench)


@app.function(image=image, gpu="B300", timeout=3600,
              volumes={REMOTE_OUT: volume})
def _profile_source_b300(run_cmd: str, run_id: str,
                         files: dict[str, bytes],
                         launch_skip: int | None = None,
                         launch_count: int = 1,
                         clock_control: str = "base",
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


def _print_smi_box(lines: list[str], clear: bool = False) -> None:
    if clear:
        print("\r" + " " * 80 + "\r", end="", flush=True)
    width = max(len(ln) for ln in lines)
    print("┌" + "─" * (width + 2) + "┐")
    for ln in lines:
        print("│ " + ln + " " * (width - len(ln)) + " │")
    print("└" + "─" * (width + 2) + "┘")


def _watch_progress(run_id: str, total_steps: int,
                    volume: modal.Volume) -> threading.Event:
    """Poll the container's progress file and render a live status bar.

    TTY: one in-place `\\r` status line. Piped (agent runs): no spinner
    flood — one plain line per step change, plus the smi box once."""
    stop = threading.Event()
    tty = sys.stdout.isatty()
    width = 20
    smi_shown = False
    last = None

    def _render() -> None:
        nonlocal smi_shown, last
        start = time.monotonic()
        while not stop.is_set():
            lines: list[str] = []
            try:
                data = b"".join(volume.read_file(f"{run_id}.progress"))
                lines = data.decode(errors="replace").splitlines()
            except Exception:
                lines = []
            elapsed = int(time.monotonic() - start)
            mm, ss = divmod(elapsed, 60)
            if not smi_shown:
                smi_lines = [ln[4:] for ln in lines if ln.startswith("smi:")]
                if smi_lines:
                    _print_smi_box(smi_lines, clear=tty)
                    smi_shown = True
            steps = [ln for ln in lines if ln.startswith("step:")]
            cur = steps[-1][5:] if steps else "waiting for Modal container"
            done = min(len(steps), total_steps)
            msgs = [ln for ln in lines if ln.startswith("msg:")]
            status = msgs[-1][4:] if msgs else ""
            m = re.search(r"(\d+)%", status)
            if m:
                pct = int(m.group(1))
            elif done > 0 and total_steps > 1:
                pct = done * 100 // (total_steps - 1)
            else:
                pct = 0
            if tty:
                filled = pct * width // 100
                bar = "█" * filled + "░" * (width - filled)
                print(f"\r[{done}/{total_steps}] {cur} — {mm}:{ss:02d} — "
                      f"[{bar}] {pct:3d}% {status[:70]}" + " " * 4,
                      end="", flush=True)
            elif (done, cur) != last:
                print(f"[{done}/{total_steps}] {cur} — {pct}% "
                      f"{status[:70]}", flush=True)
                last = (done, cur)
            time.sleep(1)
        if tty:
            print()

    t = threading.Thread(target=_render, daemon=True)
    t.start()
    return stop


def profile_on_modal(source_dir: Path, run_cmd: str, run_id: str,
                     gpu: str, timeout: int,
                     launch_skip: int | None = None,
                     launch_count: int = 1,
                     clock_control: str = "base",
                     bench: dict | None = None) -> dict[str, bytes]:
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
    if bench:
        files["ncu-view-cublas-bench.cu"] = _cublas_bench_source(
            bench.get("precision", "fp16"), int(bench["shape"])
        ).encode()
    total_steps = 5 + (1 if bench else 0)
    stop = _watch_progress(run_id, total_steps, volume)
    try:
        with app.run():
            return fn.remote(run_cmd, run_id, files, launch_skip,
                             launch_count, clock_control, bench)
    except Exception as e:
        msg = str(e).strip()
        raise SystemExit(f"modal run failed:\n{msg[-2000:]}") from None
    finally:
        stop.set()
