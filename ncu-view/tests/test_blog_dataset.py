"""Pin the blog's exact numbers against ncu-view's rendering.

Every figure asserted here is a value published in the "Beating cuBLAS on
B200" write-up (blog/beating-cublas-on-b200-with-cute-dsl.md), so a change
in either the data or the tool's math shows up immediately.

Run: python3 tests/test_blog_dataset.py   (no pytest needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ncu_view.html import render_html
from ncu_view.report import build
from ncu_view.sections import section_stall_total

JSON = HERE.parent.parent / "kernels" / "cute_dsl" / "B200" / "results" / "ncu_counters.json"

TMA = {"v1": 15.22, "v2": 30.00, "v3": 51.07, "v4": 43.37, "v5": 46.88, "v6": 46.87, "cuBLAS": 50.68}
STALL = {"v1": 116.16, "v2": 57.44, "v3": 38.48, "v4": 37.83, "v5": 52.91, "v6": 52.92, "cuBLAS": 46.48}
VERDICT = {
    "v1": "LATENCY-BOUND",
    "v2": "COMPILER-OPACITY",
    "v3": "PIPE-BOUND",
    "v4": "ISSUE-SERIALIZATION",
    "v5": "CONVERGING",
    "v6": "CONVERGED",
    "cuBLAS": "REFERENCE",
}


def approx(a: float, b: float, tol: float = 0.06) -> bool:
    return abs(a - b) <= tol


def test_kernel_order_and_count():
    rep = build(str(JSON))
    names = [k["name"] for k in rep["kernels"]]
    assert names == ["v1", "v2", "v3", "v4", "v5", "v6", "cuBLAS"], names
    assert rep["meta"]["kernels"] == 7


def test_tma_derived():
    rep = build(str(JSON))
    for k in rep["kernels"]:
        sec = next(s for s in k["sections"] if s["sid"] == "memoryworkload")
        tma = next(r for r in sec["rows"] if r["label"] == "TMA active")
        assert approx(float(tma["value"]), TMA[k["name"]]), (k["name"], tma["value"])
        assert tma["derived"], "TMA row must be marked derived"


def test_stall_totals():
    from ncu_view.ingest import ingest

    profs = ingest(str(JSON))
    for p in profs:
        assert approx(section_stall_total(p) or 0.0, STALL[p.name]), (p.name, section_stall_total(p))


def test_tflops_and_per_active_cycle():
    rep = build(str(JSON))
    v6 = next(k for k in rep["series"] if k["name"] == "v6")
    cb = next(k for k in rep["series"] if k["name"] == "cuBLAS")
    assert approx(v6["tflops"], 1788.4, 1.0)
    assert approx(cb["tflops"], 1576.8, 1.0)
    eff_v6 = v6["tflops"] / (v6["pipe_pct"] / 100.0)
    eff_cb = cb["tflops"] / (cb["pipe_pct"] / 100.0)
    assert approx(eff_v6, 1978.0, 6.0), eff_v6
    assert approx(eff_cb, 1590.0, 6.0), eff_cb
    assert eff_cb < eff_v6, "the busier kernel must not be the faster per active cycle"


def test_verdicts_reproduce_the_blog():
    rep = build(str(JSON))
    for k in rep["kernels"]:
        assert k["verdict"] is not None
        assert VERDICT[k["name"]] in k["verdict"]["name"], (k["name"], k["verdict"]["name"])
    v5 = next(k for k in rep["kernels"] if k["name"] == "v5")
    assert "denominator" in v5["verdict"]["message"], "v5 must carry the stall-counter caveat"


def test_html_deterministic_and_complete():
    rep = build(str(JSON))
    html1 = render_html(rep)
    html2 = render_html(rep)
    assert html1 == html2, "HTML output must be deterministic"
    for name in ("v1", "v6", "cuBLAS", "LATENCY-BOUND", "Warp State Statistics",
                 "Speed Of Light", "Memory Workload Analysis"):
        assert name in html1, name


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
