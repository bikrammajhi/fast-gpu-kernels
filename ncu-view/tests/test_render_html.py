"""Render smoke tests: the HTML must actually render (not just be valid).

Extracts the embedded JS payload from the rendered HTML and executes it in
node against a minimal DOM stub, asserting that the kernel page, summary
view and sidebar produce their expected elements. Skipped when node is
unavailable.

Run:  python3 tests/test_render_html.py
"""

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncu_view.html import render_html  # noqa: E402
from ncu_view.report import build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
BLOG_JSON = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "ncu_counters.json"

STUB = """
function el() { return { _html:'', textContent:'', style:{}, dataset:{},
  classList:{toggle(){},add(){},remove(){}},
  set innerHTML(v){this._html=v}, get innerHTML(){return this._html},
  addEventListener(){}, scrollIntoView(){}, id:'', parentElement:null,
  querySelector(){return el()}, querySelectorAll(){return []} }; }
const els = {};
const document = { addEventListener(t,f){ if(t==='DOMContentLoaded') this.ready=f; },
  querySelector(s){ if(!els[s]) els[s]=el(); return els[s]; },
  querySelectorAll(){ return []; } };
globalThis.document = document;
globalThis.IntersectionObserver = class { observe(){} };
"""


def _js(html: str) -> str:
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    assert m, "no <script> payload in rendered HTML"
    return m.group(1)


def _run_node(program: str) -> str:
    r = subprocess.run(
        [shutil.which("node") or "node", "-e", program],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise AssertionError(f"node failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


CHECKS = """
document.ready();
const main = els['#main'].innerHTML;
const sb = els['#sidebar'].innerHTML;
const KERNELS = ['v1','v2','v3','v4','v5','v6','cuBLAS'];
const checks = {};
checks['stats strip'] = main.includes('Duration') && main.includes('Tensor pipe');
checks['no verdict invented'] = !main.includes('verdict sev-');
checks['no nvidia sections'] = !main.includes('Recommendations (NVIDIA rule engine)');
checks['honest fallback note'] = main.includes('No NVIDIA sections for this input');
checks['achieved occupancy chip'] = main.includes('Achieved occupancy');
checks['all kernels in sidebar'] = KERNELS.every(n => sb.includes(n));
checks['summary nav'] = sb.includes('Summary');
checks['no derived toggle'] = !main.includes('show derived');
curView = 'summary'; renderMain();
const summ = els['#main'].innerHTML;
checks['summary series table'] = summ.includes('Kernel series') &&
  KERNELS.every(n => summ.includes(n));
checks['summary recs'] = summ.includes('Prioritized recommendations');
checks['summary best'] = summ.includes('Best kernel');
let ok = true;
for (const [k, v] of Object.entries(checks)) {
  console.log((v ? 'PASS' : 'FAIL') + '  ' + k); if (!v) ok = false; }
console.log(ok ? '\\nALL RENDER CHECKS PASS' : '\\nRENDER FAILURES');
"""

GOLDEN_CHECKS = """
document.ready();
const main = els['#main'].innerHTML;
const sb = els['#sidebar'].innerHTML;
const checks = {};
checks['sol rows from nvidia'] = ['SM Frequency', 'DRAM Throughput']
  .every(s => main.includes(s));
checks['tensor pipe row'] = main.includes('Tensor pipe') ||
  main.includes('Tensor Pipe Utilization');
checks['warp stalls'] = main.includes('Stall Cycles Per Issued Instruction') ||
  main.includes('Warp State');
checks['nvidia rules block'] = main.includes('Recommendations (NVIDIA rule engine)');
checks['bottleneck banner'] = main.includes('Bottleneck') &&
  main.includes('verdict sev-');
checks['pm nvidia detailed'] = ['Maximum Buffer Size', 'Maximum Sampling Interval']
  .every(s => main.includes(s));
checks['no derived toggle'] = !main.includes('show derived') &&
  !main.includes('derived-sec');
checks['no our verdicts'] = !main.includes('PIPE-BOUND') &&
  !main.includes('LATENCY-BOUND');
checks['sections tree nvidia'] = sb.includes('Speed Of Light') &&
  sb.includes('Warp State Statistics');
checks['zero not-collected'] = !main.includes('not collected');
let ok = true;
for (const [k, v] of Object.entries(checks)) {
  console.log((v ? 'PASS' : 'FAIL') + '  ' + k); if (!v) ok = false; }
console.log(ok ? '\\nGOLDEN RENDER CHECKS PASS' : '\\nGOLDEN RENDER FAILURES');
"""

GOLDEN_REP = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "golden" / "matmul_v1.ncu-rep"


def test_render_html() -> None:
    if shutil.which("node") is None:
        print("SKIP test_render_html: node not available")
        return
    assert BLOG_JSON.exists(), f"missing {BLOG_JSON}"
    report = build(str(BLOG_JSON))
    html = render_html(report)
    js = _js(html)
    assert "__DATA__" not in js, "payload placeholder left in output"
    out = _run_node(STUB + js + CHECKS)
    assert "ALL RENDER CHECKS PASS" in out, out


def test_render_deterministic() -> None:
    report = build(str(BLOG_JSON))
    assert render_html(report) == render_html(report), \
        "HTML must be deterministic across renders"


def test_render_html_golden() -> None:
    if shutil.which("node") is None:
        print("SKIP test_render_html_golden: node not available")
        return
    if not GOLDEN_REP.exists():
        print(f"SKIP test_render_html_golden: missing {GOLDEN_REP}")
        return
    if importlib.util.find_spec("ncu_report") is None:
        print("SKIP test_render_html_golden: ncu-report not installed")
        return
    report = build(str(GOLDEN_REP))
    html = render_html(report)
    out = _run_node(STUB + _js(html) + GOLDEN_CHECKS)
    assert "GOLDEN RENDER CHECKS PASS" in out, out


if __name__ == "__main__":
    test_render_html()
    test_render_deterministic()
    test_render_html_golden()
    print("render tests OK")
