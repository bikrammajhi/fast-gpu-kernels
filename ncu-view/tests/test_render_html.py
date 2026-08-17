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
  querySelectorAll(){ return []; },
  createElement(){ return el(); } };
globalThis.document = document;
globalThis.IntersectionObserver = class { observe(){} };
"""


def _js(html: str) -> str:
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    assert m, "no <script> payload in rendered HTML"
    return m.group(1)


def _run_node(program: str) -> str:
    r = subprocess.run(
        [shutil.which("node") or "node", "-"],
        input=program, capture_output=True, text=True, timeout=120,
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
checks['derived card ours on counters input'] = main.includes('Derived metrics') &&
  main.includes('src-tag ours') && main.includes('DRAM bytes moved') &&
  !main.includes('CTAs launched');
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
checks['top rec banner'] = (() => {
  const vname = (main.match(/class="verdict sev-[a-z]+">([^<]+)</) || [])[1];
  const r1 = (main.match(/<div class="recno">1<\\/div>\\s*<div class="rectext">([^<]+)</) || [])[1];
  return !!vname && vname === r1 &&
    main.includes('Top recommendation · NVIDIA rule engine'); })();
checks['pm nvidia detailed'] = ['Maximum Buffer Size', 'Maximum Sampling Interval']
  .every(s => main.includes(s));
checks['no derived toggle'] = !main.includes('show derived') &&
  !main.includes('derived-sec');
checks['no our verdicts'] = !main.includes('PIPE-BOUND') &&
  !main.includes('LATENCY-BOUND');
checks['sections tree nvidia'] = sb.includes('Speed Of Light') &&
  sb.includes('Warp State Statistics');
checks['zero nvidia not-collected one-liners'] =
  !main.includes('==WARNING== No metrics to show');
checks['derived card marked ours'] = main.includes('Derived metrics') &&
  main.includes('src-tag ours') && main.includes('Calculated by ncu-view, not by NVIDIA');
checks['derived values present'] = ['Kernel-wide IPC', 'Instructions per thread',
  'CTAs launched', 'Occupancy utilization'].every(s => main.includes(s));
checks['derived descs'] = main.includes('ddesc') && main.includes('dgroup');
checks['derived click details'] = main.includes('ddetails') &&
  main.includes("this.classList.toggle('open')");
checks['derived tensor tile'] = main.includes('Tensor FLOPS (NVIDIA ops-path)');
checks['roofline chart computed'] = main.includes('rl-pt') &&
  main.includes('>Achieved<') && main.includes('Ridge Point');
checks['roofline note ours'] = main.includes('calculated by ncu-view') &&
  main.includes('SASS FMA FLOP counters') &&
  main.includes('NVIDIA tensor ops-path');
checks['pm sampling real config'] = !main.includes('Illustrative') &&
  main.includes('Sampling configuration') && main.includes('Warp-sampling period');
checks['hierarchy tables'] = main.includes('Memory hierarchy') &&
  main.includes('FLOP accounting by precision') && main.includes('L1 (SM\u2194L2)');
checks['memory chart rendered'] = main.includes('Memory Chart') &&
  main.includes('memchart') && main.includes('mc-unit') &&
  main.includes('Link % of') && main.includes('L1/TEX Cache') &&
  main.includes('Device Memory');
checks['memory chart links'] = main.includes('sectors') &&
  main.includes('· %') || main.includes('req · ') || main.includes('· 16.3%');
checks['memory tables all'] = ['Shared Memory', 'L1/TEX Cache', 'L2 Cache',
  'L2 Eviction Policies', 'Device Memory'].every(s => main.includes(s));
checks['memory table rows'] = main.includes('GPU Total') &&
  main.includes('Bank Conflicts') && main.includes('Sector Misses to L2') &&
  main.includes('TEX Op') && main.includes('Normal Demote');
checks['search palette built'] = typeof buildSearchIndex === 'function' &&
  typeof searchGo === 'function' && typeof bindSearch === 'function';
checks['search index nonempty'] = typeof SEARCH !== 'undefined' && SEARCH.length > 0;
checks['metric ref wrapper'] = main.includes('Metric reference') &&
  main.includes('Profiling Guide §2.4 metric families');
checks['metric ref cards'] = ['sec-mr-launch', 'sec-mr-occupancy', 'sec-mr-device',
  'sec-mr-pcsamp', 'sec-mr-pcsamp-not-issued', 'sec-mr-warpidsamp',
  'sec-mr-warpidsamp-not-issued', 'sec-mr-source', 'sec-mr-evict']
  .every(id => main.includes('id="' + id + '"'));
checks['metric ref launch rows'] = main.includes('launch__grid_size') &&
  main.includes('Maximum total number of blocks in a grid') && main.includes('2,048') &&
  main.includes('13.84');
checks['metric ref values'] = main.includes('NVIDIA B200') && main.includes('143.9K') &&
  main.includes('6.25%') && main.includes('PolicySpread');
checks['metric ref honest absent'] = main.includes('Not collected in this profile') &&
  main.includes('warpidsamp metric family is absent');
checks['metric ref sidebar'] = sb.includes('METRIC REFERENCE') &&
  sb.includes('Launch Metrics');
checks['metric ref search rows'] = typeof SEARCH !== 'undefined' &&
  SEARCH.some(x => x.kind === 'r' && x.label === 'launch__grid_size' && x.sid === 'mr-launch') &&
  SEARCH.some(x => x.kind === 's' && x.sid === 'mr-evict');
checks['long rule messages wrap'] = main.includes('class="num wrap">The memory access pattern for global stores') &&
  main.includes('class="num wrap">Out of the 2147588800.0 bytes') &&
  main.includes('class="num wrap">The ratio of peak float (FP32)');
checks['mref names wrap'] = main.includes('class="mref-name">') &&
  !main.includes('class="l mref-name"');
checks['short kernel names'] = main.includes('class="kname" title="click to show full name"') &&
  main.includes('class="ks">kernel_cutlass') && main.includes('…CopyAtom_ThrI_0</span>') &&
  main.includes('class="kf">kernel_cutlass_kernel_TiledMMA_ThrLayoutVMNK11110000');
curView = 'summary'; renderMain();
const summ = els['#main'].innerHTML;
checks['recs rule name first'] = summ.includes('<span class="rec-name">Warp Stall</span>') &&
  !summ.includes('rec-k');
checks['recs no kernel names'] = !summ.includes('kernel_cutlass') &&
  !summ.includes('class="kname"');
checks['series no kernel column single kernel'] = !summ.includes('<th>Kernel</th>') &&
  summ.includes('Kernel series');
checks['recs est desc'] = summ.indexOf('<span class="rec-name">Uncoalesced Global Accesses</span>') <
  summ.indexOf('<span class="rec-name">Issue Slot Utilization</span>') &&
  summ.indexOf('<span class="rec-name">Issue Slot Utilization</span>') <
  summ.indexOf('<span class="rec-name">Theoretical Occupancy</span>') &&
  summ.includes('est. 86.3x') && !summ.includes('est. 64.2xx') &&
  summ.includes('>est. 9.1x<');
checks['recs all 11 shown'] = summ.includes('<span class="rec-name">Bottleneck</span>') &&
  summ.includes('<span class="rec-name">Roofline Analysis</span>') &&
  summ.includes('<span class="rec-name">High Pipe Utilization</span>');
checks['matmul prefix stripped'] = kdisp('matmul_v1.cu-1786855259') === 'v1.cu-1786855259' &&
  kdisp('matmul_v1.py-1786885115') === 'v1.py-1786885115' &&
  kdisp('matmul') === 'matmul' && kdisp('v1_kern') === 'v1_kern';
checks['big numbers humanized'] = main.includes('>4.49 <') && main.includes('M cycle') &&
  main.includes('>4.37 <') && main.includes('35.34M') &&
  main.includes('1.13G') && main.includes('191.5G') &&
  main.includes('1.05') && main.includes('M inst') && main.includes('1.100 TFLOP') &&
  !main.includes('35,339,164') && !main.includes('4.49396e+06');
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
