"""Command line interface for ncu-view.

    ncu-view report  <input>... [--outdir DIR] [--open]      build the HTML report
    ncu-view summary <input>...                               terminal verdict table
    ncu-view serve   <input>... [--port P]                    serve the report over HTTP
    ncu-view view    <input>...                               build + open in browser
    ncu-view profile <source> [--outdir DIR] [--build-cmd C]  profile code on Modal + report

Inputs are results/ncu_counters.json, a raw ncu CSV (`ncu --page raw --csv`),
or .ncu-rep report files (requires NVIDIA's ncu_report module). The command
word may be omitted entirely: `ncu-view --path foo.ncu-rep` is `report`.

Every derived number is NVIDIA's own: DRAM % and occupancy come from the
profile's pct_of_peak_sustained counters (see the Nsight Compute Profiling
Guide, https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .html import render_html
from .report import build

SEV_COLOR = {"critical": "\x1b[31m", "warning": "\x1b[33m",
             "suggestion": "\x1b[36m", "info": "\x1b[34m"}
RESET = "\x1b[0m"

COMMANDS = ("report", "summary", "serve", "view", "profile")


def _fmt(v, width, decimals=1, comma=False):
    """Format a possibly-None metric for the terminal table (None -> '—')."""
    if v is None:
        return f"{'—':>{width}}"
    if comma:
        return f"{v:>{width},.{decimals}f}"
    return f"{v:>{width}.{decimals}f}"


def _print_summary(report: dict) -> None:
    meta = report["meta"]
    dev = meta.get("device") or {}
    print(f"ncu-view {__version__} — {meta['input']} ({meta['source']})")
    print(f"device: {dev.get('name') or 'unknown'}")
    print()
    hdr = (f"{'kernel':<34}{'time (µs)':>12}"
           f"{'pipe %':>8}{'stall':>9}  top stall")
    print(hdr)
    for k in report["series"]:
        top = f"{k['top_stall']:.1f}" if k["top_stall"] is not None else "—"
        print(f"{k['name'][:33]:<34}{_fmt(k['time_us'], 12, 0, comma=True)}"
              f"{_fmt(k['pipe_pct'], 8, 1)}"
              f"{_fmt(k['stall_cycles'], 9, 2)}  {top}")
    print()
    for k in report["kernels"]:
        v = k["verdict"]
        if v:
            color = SEV_COLOR.get(v["severity"], "")
            print(f"{k['name'][:33]:<34}{color}{v['name']}{RESET}")
            print(f"{'':<34}  {v['message']}")
    print()


def _serve(report: dict, port: int) -> None:
    payload = json.dumps(report, indent=1).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/report.html"):
                body = render_html(report).encode()
                ctype = "text/html"
            elif self.path == "/report.json":
                body = payload
                ctype = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http.server.HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def _kernel_name(input_path: Path) -> str:
    """The input's kernel name: file name minus known profile suffixes."""
    name = input_path.name
    for suffix in (".ncu-rep", ".raw.csv", ".csv", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return input_path.stem


def _default_outdir(outdir: str | None, inputs: list[str]) -> Path:
    if outdir:
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = Path(inputs[0]).parent / f"{_kernel_name(Path(inputs[0]))}-ncu-report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_report(args: argparse.Namespace) -> dict:
    kfilter = None
    if getattr(args, "kernel_regex", None):
        kfilter = re.compile(args.kernel_regex)

    report = build(args.inputs[0])
    for extra in args.inputs[1:]:
        more = build(extra)
        report["kernels"] += more["kernels"]
        report["series"] += more["series"]
        report["meta"]["kernels"] = len(report["kernels"])
        report["meta"]["input"] += f" + {extra}"
    if kfilter:
        report["kernels"] = [k for k in report["kernels"] if kfilter.search(k["name"])]
        report["series"] = [k for k in report["series"] if kfilter.search(k["name"])]
        report["meta"]["kernels"] = len(report["kernels"])
    return report


def _run_profile(args: argparse.Namespace) -> int:
    from .profile import run_profile

    return run_profile(
        source=args.inputs[0],
        outdir=Path(args.outdir) if args.outdir else None,
        build_cmd=args.build_cmd,
        timeout=args.timeout,
        no_modal=args.no_modal,
        modal_gpu=args.modal_gpu,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        clock_control=args.clock_control,
        compare_cublas=args.compare_cublas,
        bench_precision=args.bench_precision,
        bench_shape=args.M,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "report"
    if argv and argv[0] in COMMANDS:
        command = argv.pop(0)

    ap = argparse.ArgumentParser(prog="ncu-view", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"ncu-view {__version__}")
    ap.add_argument("inputs", nargs="*", metavar="INPUT",
                    help=".json probe, ncu CSV dump, or .ncu-rep (also --path)")
    ap.add_argument("--path", dest="path", action="append", default=None,
                    metavar="FILE",
                    help="input file (repeatable; positional works too)")
    ap.add_argument("-o", "--outdir", "--out", "--path-to-report",
                    default=None, metavar="DIR",
                    help="directory for the report files (default: "
                         "<kernel>-ncu-report/ next to the input)")
    ap.add_argument("--kernel-regex", default=None,
                    help="only analyze kernels matching this regex")
    ap.add_argument("--M", type=int, default=None, metavar="N",
                    help="profile: square GEMM size for the --compare-cublas "
                         "reference run (required with --compare-cublas; "
                         "irrelevant otherwise)")
    ap.add_argument("--open", action="store_true",
                    help="open the report in the browser")
    ap.add_argument("--port", type=int, default=8000,
                    help="serve port (default 8000)")
    ap.add_argument("--build-cmd", default=None, metavar="CMD",
                    help="profile: command that runs the kernel binary "
                         "(default: guess from the source)")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="profile: Modal run timeout in seconds (default 1200)")
    ap.add_argument("--modal-gpu", default="H100", metavar="ACCELERATOR",
                    help="profile: Modal accelerator, e.g. H100, B200 "
                         "(default H100)")
    ap.add_argument("--launch-skip", type=int, default=None, metavar="N",
                    help="profile: ncu launch skip (default 1: skip the "
                         "first warmup launch and profile ONE kernel; pass "
                         "N to land on a specific launch)")
    ap.add_argument("--launch-count", type=int, default=1, metavar="N",
                    help="profile: ncu launches to capture (default 1 — one "
                         "kernel at a time); >1 averages "
                         "gpu__time_duration over steady-state launches "
                         "(default 1)")
    ap.add_argument("--clock-control", default="base",
                    choices=("none", "base", "boost"),
                    help="profile: ncu clock control for the capture. ncu's "
                         "own default 'base' locks the GPU to its base clock; "
                         "'boost' locks the max boost clock (the reproducible "
                         "peak reviewers expect, where the host allows it) "
                         "and 'none' lets the app's warm-up drive clocks "
                         "(default base — ncu's own default)")
    ap.add_argument("--compare-cublas", action="store_true",
                    help="profile: also profile a cuBLAS GEMM in the same "
                         "run under identical ncu flags, so the report's "
                         "series and chart compare your kernel against "
                         "cuBLAS at the same clock")
    ap.add_argument("--bench-precision", default="fp16", choices=("fp16", "bf16"),
                    help="profile: precision of the cuBLAS comparison GEMM, "
                         "matched to your kernel's io dtype (default fp16)")
    ap.add_argument("--no-modal", action="store_true",
                    help="profile: run ncu locally instead of on Modal")
    args = ap.parse_args(argv)
    args.command = command
    args.inputs = list(args.inputs) + list(args.path or [])
    if not args.inputs:
        ap.error("an input file is required (--path or positional)")

    if args.command == "profile":
        return _run_profile(args)

    try:
        report = _build_report(args)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: input not found: {e.filename or args.inputs[0]}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"error: {args.inputs[0]} is not valid JSON: {e}\n")
        return 1
    except KeyError as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    except ImportError:
        sys.stderr.write("error: reading .ncu-rep needs NVIDIA's ncu_report; "
                         "install with: pip install 'ncu-view[ncu]'\n")
        return 1
    if args.command == "summary":
        _print_summary(report)
        return 0
    if args.command == "serve":
        print(f"serving on http://localhost:{args.port}/report.html")
        _serve(report, args.port)
        return 0

    outdir = _default_outdir(args.outdir, args.inputs)
    stem = Path(args.inputs[0]).stem.replace(".ncu-rep", "")
    html_path = outdir / f"{stem}.html"
    json_path = outdir / f"{stem}.json"
    html_path.write_text(render_html(report))
    json_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {html_path}")
    print(f"wrote {json_path}")
    if args.open or args.command == "view":
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
