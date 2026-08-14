"""Diff harness: prove ncu-view's numbers against Nsight Compute itself.

Two ways to run:

1. Against a profile you already captured (the golden .ncu-rep from the
   Modal session, plus the raw CSV re-export of the same report):

       python3 tests/test_against_ncu.py ~/golden/matmul_v1.ncu-rep ~/golden/raw.csv

2. Generate both fresh: requires ncu + the kernel's profiling command:

       python3 tests/test_against_ncu.py --collect \
           -- ncu -o report.ncu-rep --set full --page raw --csv ... <kernel cmd>

The harness ingests both views of the SAME profile — the .ncu-rep via
ncu_view.ingest_report and the raw CSV via ingest_csv — and requires every
shared counter to agree within tolerance, plus the derived sections/rules
to render identically. Divergence = a bug in ncu-view, not in ncu.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ncu_view.ingest import STALL_BASES, ingest
from ncu_view.rules import rules_for
from ncu_view.sections import tma_pct

TOL = {
    "time": 1e-9,          # exact: same report file
    "default": 1e-6,       # floats from the same source must match near-exactly
}

CORE_METRICS = [
    "gpu__time_duration.avg",
    "sm__cycles_elapsed.avg",
    "smsp__inst_executed.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # sm__ctas_active.* is not in the 2025 full-set metric export; the
    # grid-size rule degrades gracefully when it is absent.
    "dram__bytes.sum.per_second",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
]


def _ingest_pair(report_path: str, csv_path: str):
    from_report = ingest(report_path)
    from_csv = ingest(csv_path)
    by_name = {}
    for kp in from_report:
        by_name.setdefault(kp.name, []).append(kp)
    mismatched = []
    for kp in from_csv:
        cands = by_name.get(kp.name)
        if not cands:
            mismatched.append(f"kernel {kp.name!r} missing from .ncu-rep ingest")
            continue
        # match by name; take the first report kernel with that name
        rp = cands[0]
        for metric in CORE_METRICS:
            a, b = rp.metrics.get(metric), kp.metrics.get(metric)
            if a is None or b is None:
                mismatched.append(f"{kp.name}: {metric}: missing from one ingest "
                                  f"(report={a is not None}, csv={b is not None})")
                continue
            tol = TOL.get(metric, TOL["default"])
            if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
                mismatched.append(
                    f"{kp.name}: {metric}: ncu-rep {a:.6g} vs csv {b:.6g}"
                )
        # The warp-stall counters are the heart of the analysis: both ingests
        # must carry all of them and agree, or the sections and verdicts are
        # not comparing like with like.
        for base in STALL_BASES:
            a, b = rp.metrics.get(base), kp.metrics.get(base)
            if a is None or b is None:
                mismatched.append(f"{kp.name}: stall counter {base} missing "
                                  f"(report={a is not None}, csv={b is not None})")
                continue
            if abs(a - b) > TOL["default"] * max(1.0, abs(a), abs(b)):
                mismatched.append(f"{kp.name}: {base}: ncu-rep {a:.6g} vs csv {b:.6g}")
        ta, tb = tma_pct(rp)[0], tma_pct(kp)[0]
        if ta is None or tb is None or abs(ta - tb) > 1e-6 * max(1.0, abs(ta), abs(tb)):
            mismatched.append(f"{kp.name}: TMA pct: ncu-rep {ta} vs csv {tb}")
        if mismatched:
            continue
        r_verdict = next((r for r in rules_for(rp, None, {}) if r.rid == "verdict"), None)
        c_verdict = next((r for r in rules_for(kp, None, {}) if r.rid == "verdict"), None)
        rv = r_verdict.name if r_verdict else None
        cv = c_verdict.name if c_verdict else None
        if rv != cv:
            mismatched.append(f"{kp.name}: verdict differs: {rv} vs {cv}")
    return mismatched


def _collect(ncu_args: list[str]) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rep = td / "report.ncu-rep"
        csvp = td / "raw.csv"
        cmd = ncu_args + ["-o", str(rep)]
        subprocess.run(cmd, check=True)
        subprocess.run(
            ["ncu", "--import", str(rep), "--page", "raw", "--csv", "--export", str(csvp)],
            check=True,
        )
        return rep, csvp


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--collect":
        args = args[1:]
        sep = args.index("--")
        ncu_args = args[sep + 1:]
        if not ncu_args:
            print("usage: --collect -- <ncu args>... <kernel command>")
            return 2
        rep, csvp = _collect(ncu_args)
    else:
        if len(args) < 2:
            print("usage: test_against_ncu.py <report.ncu-rep> <raw.csv>")
            return 2
        rep, csvp = Path(args[0]), Path(args[1])

    mismatches = _ingest_pair(str(rep), str(csvp))
    if mismatches:
        print(f"{len(mismatches)} divergence(s):")
        for m in mismatches:
            print(f"  - {m}")
        return 1
    print("OK: .ncu-rep ingest, CSV ingest, sections and verdicts agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
