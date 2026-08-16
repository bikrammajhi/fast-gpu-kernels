"""The `profile` command: source directory -> ncu report -> HTML report.

One command turns any CUDA source tree into a full ncu report:

    ncu-view profile kernels/cute_dsl/B200 -o out/

The source is uploaded to Modal (or profiled locally with --no-modal),
built and run under ncu's full metric set, and the report is rendered
into outdir. `--build-cmd` overrides the build/run heuristic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# NVIDIA sections whose own tables we export next to the report so the
# report page can show NVIDIA's detailed numbers instead of our derivation.
NCU_DETAIL_SECTIONS = [
    "SchedulerStats",
    "WarpStateStats",
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "MemoryWorkloadAnalysis_Chart",
    "Occupancy",
    "SourceCounters",
    "SpeedOfLight_RooflineChart",
    "PmSampling",
]
# ncu section ids that don't match our own sid naming; the CSV overlay
# keys on OUR sid, so the exported file must be named with it.
NCU_SID_ALIAS = {"PmSampling": "PM Sampling"}


def _sec_filename(run_id: str, ncu_sid: str) -> str:
    sid = NCU_SID_ALIAS.get(ncu_sid, ncu_sid)
    return f"{run_id}.sec-{sid}.csv"


def _export_sections_locally(rep: Path, outdir: Path, cwd: Path) -> None:
    import subprocess

    for sid in NCU_DETAIL_SECTIONS:
        out = outdir / _sec_filename(rep.stem, sid)
        try:
            with open(out, "wb") as f:
                subprocess.run(["ncu", "--import", str(rep), "--section", sid,
                                "--page", "details", "--csv"],
                               cwd=cwd, check=True, stdout=f)
        except Exception:
            continue


def _cutlass_include() -> str:
    """-I flags for the cutlass/cute headers when nvidia-cutlass is installed.

    Tries the importable module first, then a filesystem search of
    site-packages or /opt/cutlass (3.x ships the util helpers separately
    under tools/util/include).
    """
    flags = ""
    for inc in _cutlass_roots():
        flags += f" -I{inc}"
        util = Path(inc).parent / "tools" / "util" / "include"
        if util.is_dir():
            flags += f" -I{util}"
    return flags


def _cutlass_roots() -> list[Path]:
    roots = []
    for cand in ["/opt/cutlass/include"]:
        p = Path(cand)
        if (p / "cutlass" / "cluster_launch.hpp").exists():
            roots.append(p)
    try:
        import nvidia.cutlass
        p = Path(nvidia.cutlass.__file__).parent / "include"
        if p.is_dir():
            roots.append(p)
    except Exception:
        pass
    if not roots:
        import glob
        import site as site_mod
        for sp in site_mod.getsitepackages():
            for d in glob.glob(sp + "/**/include", recursive=True):
                p = Path(d)
                if (p / "cutlass" / "cluster_launch.hpp").exists() and \
                        (p / "cute").exists():
                    roots.append(p)
    return roots


def _guess_run_cmd(src: Path) -> str:
    """Return the shell command that builds and runs the kernel."""
    if src.is_dir():
        cus = sorted(src.rglob("*.cu"))
        pys = sorted(src.glob("*.py"))
        if (src / "run.py").exists():
            return "python3 run.py"
        if cus:
            bin_path = f"/tmp/{src.name}-run"
            return f"nvcc -arch=native -O3{_cutlass_include()} -lcublas " \
                   f"-o {bin_path} " \
                   + " ".join(str(c) for c in cus) + f" && {bin_path}"
        if len(pys) == 1:
            return f"python3 {pys[0].name}"
        if (src / "Makefile").exists():
            raise SystemExit("Makefile project: pass --build-cmd "
                             "(e.g. 'make && ./run')")
        raise SystemExit("cannot guess how to run this source: pass "
                         "--build-cmd (e.g. 'python3 matmul_v1.py')")
    if src.suffix == ".py":
        return f"python3 {src.name}"
    if src.suffix == ".cu":
        bin_path = f"/tmp/{src.stem}-run"
        return f"nvcc -arch=native -O3{_cutlass_include()} -lcublas " \
               f"-o {bin_path} {src.name} && {bin_path}"
    raise SystemExit(f"cannot guess how to run {src}: pass --build-cmd "
                     "(e.g. 'python3 matmul_v1.py')")


def run_profile(source: str, outdir: Path | None, build_cmd: str | None,
                timeout: int, no_modal: bool,
                modal_gpu: str = "H100",
                launch_skip: int | None = None, launch_count: int = 1,
                clock_control: str = "boost", compare_cublas: bool = False,
                bench_precision: str = "fp16",
                bench_shape: int | None = None) -> int:
    """Profile a source tree with ncu and write the report into outdir.

    `launch_skip` defaults to None: every launch is profiled in one pass and
    the report stars the dominant kernel (max total time) — works for any
    app with no launch-order knowledge.
    """
    import subprocess

    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"source {src} does not exist")

    outdir = outdir or src.parent / f"{src.stem if src.is_file() else src.name}-ncu-report"
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{src.name}-{int(time.time())}"

    run_cmd = build_cmd or _guess_run_cmd(src)
    bench = ({"precision": bench_precision,
              "shape": bench_shape or 8192} if compare_cublas else None)

    if no_modal:
        rep = outdir / f"{run_id}.ncu-rep"
        csv = outdir / f"{run_id}.raw.csv"
        cwd = src if src.is_dir() else src.parent
        skip, count = (0, 100000) if launch_skip is None \
            else (launch_skip, launch_count)
        try:
            subprocess.run(["ncu", "--set", "full", "--clock-control",
                            clock_control, "--launch-skip",
                            str(skip), "--launch-count",
                            str(count), "-o", str(rep), "sh", "-c",
                            run_cmd], cwd=cwd, check=True)
        except FileNotFoundError:
            raise SystemExit("ncu not found on PATH: install Nsight Compute, "
                             "or drop --no-modal to profile on Modal") from None
        with open(csv, "wb") as f:
            subprocess.run(["ncu", "--import", str(rep), "--page", "raw",
                            "--csv"], cwd=cwd, check=True, stdout=f)
        _export_sections_locally(rep, outdir, cwd)
        if bench:
            from .modal_app import _cublas_bench_source
            bench_src = outdir / "ncu-view-cublas-bench.cu"
            bench_src.write_text(_cublas_bench_source(
                bench["precision"], bench["shape"]))
            bench_bin = outdir / "ncu-view-cublas-bench"
            subprocess.run(["nvcc", "-arch=native", "-O3", "-lcublas",
                            "-o", str(bench_bin), str(bench_src)],
                           cwd=cwd, check=True)
            crep = outdir / f"{run_id}-cublas.ncu-rep"
            subprocess.run(["ncu", "--set", "full", "--clock-control",
                            clock_control, "--launch-skip",
                            str(skip), "--launch-count",
                            str(count), "-o", str(crep), "sh", "-c",
                            str(bench_bin)], cwd=cwd, check=True)
            with open(outdir / f"{run_id}-cublas.raw.csv", "wb") as f:
                subprocess.run(["ncu", "--import", str(crep), "--page",
                                "raw", "--csv"], cwd=cwd, check=True,
                               stdout=f)
            _export_sections_locally(crep, outdir, cwd)
        report_input = [rep, outdir / f"{run_id}-cublas.ncu-rep"] if bench \
            else rep
    else:
        try:
            from .modal_app import profile_on_modal
        except ImportError:
            raise SystemExit("the profile command needs the modal package: "
                             "pip install modal") from None
        artifacts = profile_on_modal(src, run_cmd, run_id, modal_gpu, timeout,
                                     launch_skip, launch_count, clock_control,
                                     bench)
        if artifacts.get("error"):
            raise SystemExit(artifacts["error"])
        for name, data in artifacts.items():
            if name.startswith("__"):
                print(f"note: {data.decode()}")
                continue
            (outdir / name).write_bytes(data)
        rep, csv = outdir / f"{run_id}.ncu-rep", outdir / f"{run_id}.raw.csv"
        report_input = ([rep, outdir / f"{run_id}-cublas.ncu-rep"]
                        if bench and (outdir / f"{run_id}-cublas.ncu-rep")
                        .exists() else rep)

    from .html import render_html
    from .report import build

    report = build(report_input)
    html = outdir / f"{run_id}.html"
    html.write_text(render_html(report))
    (outdir / f"{run_id}.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {html}")
    print(f"wrote {csv}")
    return 0
