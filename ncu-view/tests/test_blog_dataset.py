"""Pin the blog's exact numbers against ncu-view's rendering.

Every figure asserted here is a value published in the "Beating cuBLAS on
B200" write-up (blog/beating-cublas-on-b200-with-cute-dsl.md), so a change
in either the data or the tool's math shows up immediately.

The blog input is a bare counters JSON (no NVIDIA sections or rules
exported), so it must also prove the honest fallback: chips only, no
sections, no verdict — nothing invented.

Run: python3 tests/test_blog_dataset.py   (no pytest needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ncu_view.html import render_html
from ncu_view.report import build

JSON = HERE.parent.parent / "kernels" / "cute_dsl" / "B200" / "results" / "ncu_counters.json"

STALL = {"v1": 116.16, "v2": 57.44, "v3": 38.48, "v4": 37.83, "v5": 52.91, "v6": 52.92, "cuBLAS": 46.48}


def approx(a: float, b: float, tol: float = 0.06) -> bool:
    return abs(a - b) <= tol


def test_kernel_order_and_count():
    rep = build(str(JSON))
    names = [k["name"] for k in rep["kernels"]]
    assert names == ["v1", "v2", "v3", "v4", "v5", "v6", "cuBLAS"], names
    assert rep["meta"]["kernels"] == 7


def test_no_derived_data_invented():
    rep = build(str(JSON))
    for k in rep["kernels"]:
        assert k["sections"] == [], (k["name"], "no NVIDIA sections for a counters-only input")
        assert k["rules"] == [], (k["name"], "no NVIDIA rules for a counters-only input")
        assert k["verdict"] is None, (k["name"], "no verdict without NVIDIA's rule engine")


def test_stall_totals():
    from ncu_view.ingest import ingest, stall_total

    profs = ingest(str(JSON))
    for p in profs:
        assert approx(stall_total(p) or 0.0, STALL[p.name]), (p.name, stall_total(p))


def test_html_deterministic_and_complete():
    rep = build(str(JSON))
    html1 = render_html(rep)
    html2 = render_html(rep)
    assert html1 == html2, "HTML output must be deterministic"
    for name in ("v1", "v6", "cuBLAS", "Duration", "SM clock", "Tensor pipe",
                 "DRAM Throughput", "Achieved occupancy", "Stall / issue"):
        assert name in html1, name
    assert "show derived" not in html1


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
