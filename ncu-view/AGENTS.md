# AGENTS.md

Operating notes for AI agents working in this repo. Read before making changes.

## Project overview

- `ncu-view/` — pip-installable package: any NVIDIA Nsight-Compute (ncu) profile
  input (`.ncu-rep`, raw CSV, or our JSON) → device-accurate Nsight-Compute-style
  HTML report. CLI: `ncu-view report <input> -o out/` (default command) and
  `ncu-view profile <source> -o out/` (runs any CUDA source on the user's Modal
  server, then renders the report locally).
- `kernels/cute_dsl/B200/results/golden/` — the golden artifacts used by tests:
  `matmul_v1.ncu-rep` + `matmul_v1.raw.csv` + 7 NVIDIA section CSVs +
  rebuilt `matmul_v1.html`/`matmul_v1.json`. Mirrored on Modal volume
  `gpulab-cute-dsl-traces` under `ncu-golden/`.

## Session-critical facts

- **This model cannot read images.** Never verify UI by looking at screenshots;
  verify via jsdom DOM checks (node + jsdom in `/tmp/opencode/node_modules`) and
  tell the user screenshots are for their eyes. Playwright screenshots live in
  `/tmp/opencode/demo/`.
- **`.ncu-rep` is NVIDIA's proprietary `NVR\x00\x02` container** — not zip/sqlite.
  Full section tables are only available via the ncu CLI
  (`ncu --import <rep> --section <sid> --csv`) or the `ncu_report` PyPI module
  (metrics/rules/samples only — no tables). Do not try to parse it directly.
- **Section CSV format (ncu 2025+/13.x):** long format — header row contains
  "Metric Name"; columns `Section Name, Metric Name, Metric Unit, Metric Value`
  (indices 11-14). One metric per row; values may contain commas; `%` unit →
  bar in the UI. A multi-kernel rep exports ALL kernels' rows in ONE file
  (full mangled names in the `Kernel Name` column, shared header) — the
  overlay filters rows by short-name substring (`_sec_csv_kernels` + the
  `kernel_name` param of `_parse_sec_csv`) and `_dedupe_rows` keeps only
  the last row block per label (multi-launch exports repeat each metric
  once per launch). Older format (label/value/unit
  rows) is the parser fallback.
- **The detail flag is `--page details`, NOT `--print-details`.** The latter is
  invalid and makes ncu fail. A section that still emits
  `==WARNING== No metrics to show` is retried without the flag (see
  `modal_app._run_ncu_sections`). MemoryWorkloadAnalysis_Tables/Chart and
  SpeedOfLight_RooflineChart legitimately have no table rows on the golden rep;
  they keep NVIDIA one-liners.
- **Section id `PmSampling` ≠ our sid `PM Sampling`** — alias map `NCU_SID_ALIAS`
  in `profile.py`; the exported CSV must be named with OUR sid (that's what the
  overlay keys on).
- **Modal volume quirks (client 1.4.2):** `Volume.put_file` does not exist
  anywhere; `Volume.batch_upload` silently no-ops (client and container);
  `commit()` is container-only. The ONLY working write path is mounting the
  volume (`volumes={REMOTE_OUT: volume}`) and writing files to the mount.
  Reads work via `volume.read_file()` (returns a **generator** — must
  `b"".join(...)`), `listdir`, and the CLI `modal volume get/put/ls/rm`
  (put refuses overwrite — rm first).
- **Modal function calls:** use `modal run -m ncu_view.modal_app::<fn> --args`
  from the repo root — the CLI ships the module. Standalone scripts in
  `/tmp/opencode/` that `import ncu_view` crash-loop in the container
  (`ModuleNotFoundError`) because the module isn't packaged — keep ad-hoc
  debug functions self-contained, or put them in `modal_app.py`.
- **Modal image:** `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` +
  `cuda-nsight-compute-13-0` + `ncu-report`; `ncu --import` works on CPU (no GPU
  needed for re-extraction); the "NVIDIA Driver was not detected" banner is
  harmless. GPU profiling runs under `--set full --warp-sampling-interval auto
  --launch-skip 1 --launch-count 1` — the explicit warp-sampling flag is
  harmless (ncu's own default) and future-proofs captures for ncu versions
  that expose warpidsamp; on ncu 13.x it adds nothing to the exports (the
  "PM Sampling: Warp States" section CSV is legitimately 0 bytes — the
  section is timeline-only, and the raw dump's warp stall data is the pcsamp
  family either way).
- **Modal image also installs:** torch (cu130 index), `nvidia-cutlass-dsl`
  (the CuTe DSL — provides `cutlass.cute`; PyPI's plain `cutlass` is the
  legacy 0.9.0 bindings, WRONG for DSL), `nvidia-cutlass` (C++ headers under
  `cutlass_library/source/include` — partial tree, no `cutlass/util/*`), and a
  shallow clone of CUTLASS at `/opt/cutlass` (LATEST tag, v4.2.2; the `util`
  helpers live under `tools/util/include` in 3.x AND 4.x, so nvcc needs BOTH
  `-I/opt/cutlass/include` and `-I/opt/cutlass/tools/util/include`).
- **Modal GPU selection is real:** one module-level `@app.function` per GPU
  type (`_profile_source_<tag>`) delegating to `_profile_source_body` — modal
  serializes functions by module qualname, so closures/factories DO NOT
  hydrate (a factory produced the "serialized=True / functools.wraps" warning
  and an empty run). `PROFILE_SOURCES` covers the full Modal catalog: T4, L4,
  A10, L40S, A100, A100-40GB, A100-80GB, RTX-PRO-6000, H100, H200, B200,
  B200+, B300. `--modal-gpu <name>` picks the function; unknown names fall
  back to H100 with a note.
- **Profile include/arch handling:** the client's `_guess_run_cmd` embeds the
  LOCAL `-I` path (useless in the container). `_profile_source_body` rewrites
  it via `_rewrite_cutlass_include` (strips nonexistent `-I` flags from the
  compile segment, appends the container's roots) and replaces `-arch=native`
  with the arch for the ACTUAL device from `nvidia-smi compute_cap`
  (7.5→sm_75 … 9.0→sm_90a, 10.0→sm_100a, 10.3→sm_103a, 11.0→sm_110a,
  12.0→sm_120a; `-arch=native` → `sm_90` without `a` on H100, tripping cute's
  `CUTE_ARCH_MMA_SM90A_ENABLED` assert). Same detect sets `CUTE_DSL_ARCH` for
  DSL kernels (sm_80+). `_guess_run_cmd` single-file dispatch: `.py` →
  `python3 <name>.py` (a bare `.py` previously fell through to nvcc!), `.cu` →
  nvcc with the file NAME only (runs in the container's cwd, not the local
  path) plus `-lcublas` (cublas isn't auto-linked). Single-file uploads now
  also ship sibling `.cu/.cuh/.h/.hpp/.c/.cpp/.py` (e.g. `common.h`).
- **`_profile_source_body` returns `{"error": ...}` instead of raising** on
  failure (ncu run AND `ncu --import raw` — SystemExit/called-process errors
  in the container weren't serialized and the client died with empty output).
  The client prints the error dict and exits 1.
- **ncu launch selection:** default profiles ONE kernel (`--launch-skip 1
  --launch-count 1` — warmup skipped) so sweep benchmarks with hundreds of
  launches stay cheap; ncu replays the whole app once per counter pass
  regardless, so `launch_count` > 1 costs no extra replays. Explicit
  `--launch-skip N` lands on a specific launch. `ingest` dedupes by kernel
  name (last occurrence = steady state), drops runtime plumbing
  (tensor-pipe util < 5% — torch init/compare, device query, memcpy;
  `NOISE_TENSOR_PCT`/`NOISE_NAMES` in `ingest.py`), and the report
  JS stars the dominant kernel (max time_us).
- **`profile` command verified end-to-end on Modal** for all four kernel
  kinds — raw CUDA (`cuda/A100/matmul_v1.cu` + `benchmark.cu` via
  `--build-cmd`), CUTE C++ (`cute/H100/matmul_v1.cu`), CUTLASS C++
  (`cutlass/gemm.cu`), cute-dsl Python (`cute_dsl/H100/...py`) — each with 7
  NVIDIA detailed sections. GPU selection verified on A100-80GB, and **B200
  verified live** (`--modal-gpu B200` on `cute_dsl/B200/matmul_v1.py`:
  device NVIDIA B200, tcgen05 kernel, LATENCY-BOUND verdict).
- **CPU-only extraction entrypoint:** `extract_sections(rep_key, run_id, dest_dir)`
  (client wrapper) → `_extract_sections` (container, volume-mounted write).
  `--no-modal` local path: `profile._export_sections_locally`.

## Architecture decisions (user-approved)

- **Official NVIDIA data, plus derived metrics tagged ours** (user decisions):
  the report renders NVIDIA's own exported section tables and NVIDIA's own
  rule-engine results verbatim. Nothing is invented: our derived sections
  (`sections.py`) and our rules/decision-table verdict (`rules.py`) were
  DELETED; the per-kernel banner verdict IS NVIDIA's SOLBottleneck rule
  (`_ncu_verdict` in `report.py`, fallback: highest-severity NVIDIA rule).
  `kernels[].sections` = `ncu_sections`, `kernels[].rules` = `ncu_rules`
  (there are no `ncu_sections`/`ncu_rules` keys anymore). NEW (user asked for
  "things the profiler doesn't provide directly"): `kernels[].derived` from
  `ncu_view/derived.py` — 29 tiles computed ONLY from NVIDIA's own section
  rows (LaunchStats/SpeedOfLight/Occupancy/InstructionStats/SchedulerStats/
  WarpStateStats) and raw counters, each carrying its formula + source rows +
  `src="ours"`, rendered as the `#derived` card (`.src-tag.ours`) atop the
  analysis panel and a sidebar item under "NCU-VIEW CALCULATIONS". No assumed
  hardware specs; a metric whose inputs are missing is skipped; `Duration`
  row units vary per rep (`ms`/`us`/`µs`) — parse via `_time_s`, never assume
  ms. Tiny values (AI ~4.3e-5) render exponential via `fmtx` in `derivedHtml`,
  never plain `fmt` (would show 0); `fmtx(0)` renders plain `0`.
  Each tile also carries a one-line `desc` (plain-language definition) and a
  `group`; tiles render under 6 group headers in a fixed order
  (COMPUTE → MEMORY → ROOFLINE → OCCUPANCY & SCHEDULING → TIMING → PM
  SAMPLING) and the formula/sources/note fold into a `.ddetails` block hidden
  by default — clicking a `.dmetric` toggles `.open` to reveal it
  (`onclick="this.classList.toggle('open')"`, `.dtog` chevron rotates). The
  "Tensor FLOPS (NVIDIA ops-path)" tile is ALWAYS emitted (value 0 + honest
  note "No tensor-core work in this profile" when every ops-path counter is
  zero/absent, e.g. FMA-only CUDA kernels); "Tensor instructions (...)" is
  arch-aware — per-family candidate counters (utcmma tcgen05 / shared_gmma
  wgmma / mma / tma_ld / tma_st), first present non-zero wins.
- **Metric reference (NVIDIA verbatim)**: `kernels[].metric_ref` from
  `metric_ref()` in `metric_ref.py` — 9 families straight from the Profiling
  Guide §2.4 (2.4.2 Launch / 2.4.3 Occupancy / 2.4.6 Device Attributes /
  2.4.7 Warp Stall Reasons / 2.4.8 Warp Stall Reasons, Not Issued / 2.4.9
  Warp Stalls per Warp ID / 2.4.10 Warp Stalls per Warp ID, Not Issued /
  2.4.11 Source Metrics / 2.4.12 L2 Cache Eviction Policies) — NVIDIA metric
  names + descriptions verbatim, values from the profile, `present:false`
  rows render '—'. Rendered by `metricRefHtml` as 9 `.card`s
  (`#sec-mr-<sid>`, meta `§<guide> · N/M collected` + `.src-tag NVIDIA`)
  inside the `#metric-ref` wrapper card at the END of the analysis panel,
  sidebar group "METRIC REFERENCE" (between "NCU-VIEW CALCULATIONS" and
  "NSIGHT COMPUTE SECTIONS", items `data-scroll="sec-mr-<sid>"`), search
  palette: kind 's' rows (label = family title + " (metric reference)", sid
  `mr-<sid>`) + kind 'r' rows per metric. String metrics (e.g.
  `launch__cluster_scheduling_policy`, `device__attribute_display_name`)
  live in `kp.str_metrics` — NOT numeric `kp.metrics`; rows carry `"str"`
  and render `.mref-str`. Value formatting in the JS `v()` helper: tiny
  non-zero (<0.01) → `toExponential(2)` (plain `fmt` would render 0 — same
  trap as `fmtx` in derived), ≥1e4 → `smart(val,'')` (SI rescale, e.g.
  35,339,164 → `35.34M`, 191,503,138,816 → `191.5G`), else
  `fmt(val, 2)` + '%' when name ends `_pct`. Golden B200 counts: launch 55/58 (54 numeric + policy string;
  `execution_model`/`kernel_name`/`sub_launch_name` absent), occupancy 3/3,
  device 20/27 documented (profile exports 164 attrs — only the
  guide-documented subset listed), pcsamp + not-issued 17/18 (`imc_miss`
  not on B200; 18/18 on H100), warpidsamp + not-issued 0/18 (VERIFIED
  unavailable in every available ncu build — the LATEST packages
  `cuda-nsight-compute-13-1/13-2/13-3` (13.1.2/13.2.2/13.3.1 debs) all ship
  the same ncu build 2025.3.1.0 and list only `smsp__pcsamp_*` in
  `--query-metrics-collection warpsampling` on ga100/gh100/b100, so it is
  NOT a version gap; the warpidsamp families carry an honest note pointing
  at the pcsamp cards as the collected equivalent), source 15/26 (H100 15,
  A100 14;
  `inst_executed` 35,339,164 B200), evict 7/7 (B200 counter names
  `explicit_hitprop_evict_normal_demote`/`explicit_missprop_evict_*` with a
  note; values 0.0 on golden). `launch__waves_per_multiprocessor` is tiny on
  A100 (0.0046 → renders `4.63e-3`) — the audit mirrors JS formatting
  (toLocaleString strips trailing zeros: `12.5%` not `12.50%`).
- **SI units everywhere (user request — "big numbers are daunting")**: the JS
  `smart(v,unit)` helper humanizes big numbers report-wide. Time units
  `ns|us|µs|ms|s(/...)` auto-convert (3429 µs → `3.43 ms`); FLOP units get
  K/M/G/T/P prefixes (`1.100 TFLOP`); any unit ≥1e3 rescales via K/M/G/T/P
  (`4,493,960 cycle` → `4.49 M cycle`, `35,339,164` → `35.34M`) with a
  guard `e>=5 && |n|>=1000 → toExponential(3)` (e.g. 6.022e23 → `6.02e+23`);
  pct values and <1e3 values stay verbatim `fmt`. Applied in: NVIDIA section
  tables (`tableHtml` — all numeric non-pct rows; bar rows with `%` untouched),
  KPI strip + SOL chips (`metric`), rule focus evidence (`fv` — tiny <0.01 →
  exponential), metric reference `v()` (≥1e4), PM sampling config, and
  summary durations (`smart(time_us,'µs')` — series table Duration column +
  Best duration KPI, label just "Duration"/"Best duration"). The
  audit's `exp_cells` mirrors the ≥1e4 SI branch exactly (rounding `n>=100`
  → 1 decimal, else 2). Golden renders: `4.49 M cycle`, `4.37 M cycle`,
  `35.34M`, `1.13G`, `191.5G`, `143.9K`, `1.05 M inst`, `1.100 TFLOP`.
- **Computed roofline (user request, replacing fakes)**: `kernels[].roofline`
  from `roofline()` in `derived.py` — `achieved` counts the FULL FLOP stream:
  FMA from NVIDIA's own `derived__sm__sass_thread_inst_executed_op_*_x2/x4`
  counters (ffma/hfma/dfma, FLOP/inst 2/2/2) **plus tensor-core FLOPs from
  NVIDIA's own `sm__ops_path_tensor_*` FLOP-path counters** (per-precision
  `_src_<fmt>_dst_<fmt>`; the `_op_` per-opcode family is the same FLOPs —
  excluded to avoid double counting; one family per arch, sparsity-suffixed
  on A100, plain on B200). No MMA shape is ever assumed: golden B200 =
  1.0995e12 tensor FLOPs at 453.6 TFLOP/s (exactly NVIDIA's own
  `...per_second` counter), `achieved_bytes`
  (dram read/write), `envelope` (achieved DRAM BW ÷ NVIDIA DRAM Throughput%
  = derived peak ~7.67 TB/s B200 / 1.92 TB/s A100 — validates vs specs; FMA
  roof = Σ NVIDIA peak_sustained insts ×2 × SM freq ~138.6 TFLOP/s; tensor
  roof = ops-path FLOPs ÷ NVIDIA %-of-peak ~4.43 PFLOPS B200; `ridge` =
  compute roof ÷ dram peak ≈ 578 FLOP/B golden (tensor), 13-27 FLOP/B on
  FMA-only kernels), `levels` (L1 = xbar2l1tex_read +
  l1tex2xbar_write bytes, L2 = lts__t_sectors×32, DRAM; achieved BW, NVIDIA
  pct, derived peak = achieved÷pct, AI per level). Rendered: `rlChart()` in
  `html.py`
  draws the log-log SOL roofline SVG (`circle.rl-pt` labeled "Achieved",
  ridge dash, region labels, compute roof = `peak_compute_flops` with
  `peak_compute_source` tensor|fma; blank + honest note when counters
  absent) and `rlTables()` adds
  hierarchy (AI column = total FLOPs/level bytes) + FLOP accounting tables
  (FMA rows + tensor rows with real FLOPs) to `#derived`.
- **PM Sampling card shows real configuration** — the "Illustrative sampled
  behavior (not this profile's samples)" fake chart was DELETED. NVIDIA's
  timeline needs per-timestamp samples from the .ncu-rep; `timed_warp_samples()`
  returns [] on these exports, so the card renders the REAL config (PM
  sampler buffer/interval rows + `profiler__pmsampler_*` counters) and
  pcsamp aggregates (sample count/aggregated passes/dropped bytes; period =
  `smsp__pcsamp_interval_cycles` ÷ SM freq ≈ 2.24 µs) as ours tiles, plus an
  honest note that the timeline is not reproducible from this export.
- Summary chips are NVIDIA counters read verbatim: pct-of-peak values are
  ncu's own `pct_of_peak_sustained_*`; SM clock prefers the SOL "SM
  Frequency" row, else ncu's own formula (`sm__cycles_elapsed.avg /
  gpu__time_duration.avg` — `_sm_freq` in `report.py`); stall is NVIDIA's
  metric — the sum of its per-reason stall counters (`stall_total` in
  `ingest.py`).
- Summary "Prioritized recommendations (all kernels)" flattens EVERY kernel's
  rules (nothing is sliced/deduped — the count is whatever NVIDIA's rule
  engine exported for that capture; `--set full` captures export more rules
  than default ones). Rows sort by NVIDIA est. speedup DESC via `estSort()`
  (empty est last, severity tiebreak) so the biggest bottleneck is first;
  rule name is the headline (`span.rec-name`) — kernel names were removed
  from summary pages by user request (no rec-k sub-line; the series table's
  Kernel column renders only for multi-kernel reports via `multiSeries`,
  and the Best kernel KPI shows a name only when >1 kernels). `est`
  strings ALREADY carry their unit
  (`'57.6x'` from `_ncu_speedup` in ingest.py, or `'5700.0%'` from the old
  API) — render verbatim, NEVER append 'x' (that was the "est. 57.6xx" bug).
  Display kernel names strip a leading `matmul_`/`matmul-`/`matmul.` token
  via `kdisp()` (but keep bare `matmul`), and long names shorten to
  `first14…last3underscoreSegments` with click-to-expand full name
  (`knameHtml`, `.kname.full` toggle).
- **Memory chart + tables (`kernels[].memory` from `memory_model()` in
  `derived.py`, rendered by `memChartHtml`/`memTablesHtml` in `html.py`)**:
  the NVIDIA Memory Workload Analysis chart mirrored from NVIDIA's own
  l1tex__*/lts__*/dram__* counters — logical units (Kernel/Global/Local/
  Surface/Shared/…, green) generate traffic, physical units (L1/TEX, L2,
  Device/System/Peer Memory, blue) service it; links carry NVIDIA counters
  (inst = SASS global/local load+store insts, req = requests, sectors, B)
  with `pct` = NVIDIA's `pct_of_peak_sustained_elapsed` for color + a
  legend; grey = no pct counter. Table blocks for Shared (inst/req/wf/
  % peak/bank conflicts per ld/st/atom/red op), L1/TEX (per-op hits/misses,
  Sector Misses to L2, whole-cache Hit Rate = NVIDIA's
  `l1tex__t_sector_hit_rate.pct`), L2 (per-client sectors incl. TEX Op
  clients, + the ECC client row; GPU Total Hit Rate = NVIDIA's
  `lts__t_sector_hit_rate.pct`), L2 Eviction Policies (per-policy sectors
  incl. zero rows; totals sum ALL cells incl. None → 0), Device Memory
  (sectors/bytes/% peak/throughput; Total % Peak = sum of read+writepct —
  not a measured value). Table render order is HARDCODED
  (shared,l1,l2,texops,evict,dram) because report serialization
  sort_keys=true alphabetizes dict keys (would put dram first). Cells
  carry formula + sources + `src=ours`; missing counters render '—'.
  Units/links are aggregated by (from,to,kind); each link also `src=ours`.
  The Kernel unit is part of `units` but drawn from the chart's `kk` box —
  `memChartHtml` must filter `name!=='Kernel'` from rows/loop or it draws
  twice. On counters-only inputs only the tables the counters support
  exist (e.g. just Device Memory). Duplicate links (two formulas, one
  path, e.g. L1 global stores via sectors AND wavefronts reqs) are
  summed by kind; `_OP_LABEL2` maps atom/red suffixes for the L2 client
  table (Atomics (ALU)/(CAS), Reductions).
- **Search palette (⌘K)**: `buildSearchIndex`/`searchPopHtml`/`searchGo`/
  `bindSearch` in `html.py` — indexes sections (`k` kernels, `s` sections,
  `r` rules, `d` derived tiles, `c` concepts from `docMetrics`+
  `docCollection`+`docReference` — the doc panels, kernel-independent) +
  metric-reference rows (`s` = family title, `r` = individual metric).
  Concept rows navigate to the kernel page, find `.definition b/.unit h4/
  .faq h3` by label, switch to the CONTAINING `.tab-panel` via
  `showTab(panel.id.replace('panel-',''))`. Concept rows have key='' —
  `ensureKernel` must guard `if(key)pushHash('#k-'+key)` (a bare '#k-'
  hash crashes jsdom's history.replaceState and re-renders in browsers).
  Popup is a div inside `.global-search` (position:relative), arrow keys
  move `.active` + scroll the row into view, Enter/Space go, Escape
  closes; `searchMark` scrolls the ACTIVE row, not the first. Only one
  row is ever `.active` (searchMark(0) then ArrowDown). Audit (see
  `/tmp/opencode/demo/audit-nc.py`) asserts search via Playwright typing
  into `#global` (fill returns None — never chain fill/click in `and`
  conditions, and `page.wait_for_timeout` also returns None).
- Inputs with no NVIDIA sections (raw CSV, counters JSON) render chips + an
  honest "No NVIDIA sections for this input" note — nothing invented.
- Sections are rendered client-side from embedded JSON via JS template
  literals in `html.py` (`kernelPage`, `secHtml`, `rulesHtml`) — `data-sid`,
  `sec-<sid>` ids appear in the RENDERED DOM only after `document.ready()`;
  jsdom checks MUST wait (setTimeout ~400ms) or sections read as missing.
- NVIDIA rule messages get "Focus metrics: name: value (info)" evidence from
  `focus_metrics[].info` (`_ncu_rules_new` in `ingest.py`).
- **No device or shape values are assumed; no TFLOPS; no GPU catalog.** The
  report is bottleneck-focused (NVIDIA verdict + pipe/DRAM/occupancy/stall
  chips + NVIDIA sections); TFLOPS was removed by user decision (Sept
  session) — `--M` survives only as the `--compare-cublas` GEMM shape. The
  hardcoded device catalog (`gpus.py`, `--gpu`, `--config`,
  `CONFIG_DEFAULTS`) was deleted by user decision (Oct session): every
  "% of peak" (DRAM, pipe, occupancy) is NVIDIA's own
  `pct_of_peak_sustained` value from the profile — DRAM via
  `dram_pct_of_peak` in `ingest.py` (the raw counter when present, else the
  Speed Of Light "DRAM Throughput" row of the exported NVIDIA section). ncu
  computes peaks from the device at profile time, so nothing is divided by a
  hardcoded spec. Reference for all features: NVIDIA's Nsight Compute
  Profiling Guide,
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html.

## Commands

```bash
cd ncu-view
python3 -m pytest tests/ -q                # 36 tests (all pass)
python3 tests/test_against_ncu.py ../kernels/cute_dsl/B200/results/golden/matmul_v1.ncu-rep \
    ../kernels/cute_dsl/B200/results/golden/matmul_v1.raw.csv   # ingest-vs-ncu harness, expects "OK:"
```

Golden rebuild (from repo root):

```bash
cd ncu-view
python3 - <<'EOF'
import sys, json; sys.path.insert(0, '.')
from ncu_view.ingest import ingest
from ncu_view.report import build
from ncu_view.html import render_html
base = '../kernels/cute_dsl/B200/results/golden/matmul_v1'
r = build(base + '.ncu-rep')  # NVIDIA sections/rules come from the sibling sec-*.csv
open(base + '.html', 'w').write(render_html(r))
open(base + '.json', 'w').write(json.dumps(r, indent=1))
EOF
```

Re-extract NVIDIA sections from the golden rep on Modal:

```bash
cd ncu-view
modal run -m ncu_view.modal_app::_extract_sections \
    --rep-key ncu-golden/matmul_v1.ncu-rep --run-id matmul_v1
# then pull: modal volume get gpulab-cute-dsl-traces "ncu-golden/<name>" <destdir>
```

UI verification via jsdom: wait 400ms after load, then assert
`#sec-<NVIDIA sid>` present with `.src-tag` = NVIDIA, rows in
`.sec-body td.l`, no `#derived-toggle`/`.derived-sec`/derived-item in the
DOM, the banner `.verdict` shows NVIDIA's rule name, `circle.rl-pt` +
`Achieved (FMA)` in the SOL roofline SVG, "Sampling configuration" in the
PM Sampling card, and 2 `.rl-t-title` tables in `#derived`. Derived card:
every `.dmetric` has a `.ddesc`, `.dgroup` headers for the 6 groups,
`.ddetails` hidden by default and expanded by clicking the tile, and the
"Tensor FLOPS (NVIDIA ops-path)" `.dname` present on every report. Memory:
`#sec-MemoryWorkloadAnalysis` holds `.memchart` (11 unit boxes — one
`Kernel` — and 13 link labels on the golden B200; fewer on A100/H100) +
6 `.mem-table`s in the HARDCODED order (first = Shared Memory, first row
'Loads'), each in its own `.mem-table-block` with an uppercase `.mem-t-title`
(SHARED MEMORY, L1/TEX CACHE, L2 CACHE, TEXTURE OPERATIONS, L2 EVICTION
POLICIES, DEVICE MEMORY) and a top border — the titles live in the table
objects (`t.title`) and MUST be rendered (the pre-July layout dropped them,
leaving six unlabeled tables glued together). Search: type into `#global`, wait ~300ms, assert
`#search-pop .search-row` counts, click a `.search-row[data-kind=c]` and
assert the active `.tab-panel` becomes `panel-metrics`.

## Key files

- `ncu-view/ncu_view/ingest.py` — `NCU_SECTION_TITLES`, `_ncu_section`,
  `_parse_sec_csv` (long-format + fallback), `_sec_csv_files`,
  `_apply_sec_csvs` (overlay replaces one-liner by sid, appends unknown sids,
  marks `detailed`), `_ncu_rules_new` (focus_info), `dram_pct_of_peak`,
  `stall_total`.
- `ncu-view/ncu_view/derived.py` — `derive()` (29 tiles on the golden B200 rep; metric-only tiles on raw-CSV inputs; each tile carries `desc` + `group`; the "Tensor FLOPS (NVIDIA ops-path)" tile is always emitted, 0 + honest note when no tensor counters) + `roofline()`
  (achieved point/envelope/levels) + `memory_model()` (units/links/tables for
  the memory chart); helpers `_m` (metric with suffix fallback),
  `_sm_freq`, `_dur_s`, `_row`, `_num`, `GROUPS`, `FMA_OPS` (ffma/hfma/dfma —
  NVIDIA's complete SASS FMA FLOP-counter family)/`TENSOR_OPS` (arch-aware
  candidate counters per family), `_op_breakdown`, `_SPACE_LABEL`,
  `_OP_LABEL`, `_OP_LABEL2`, `_cell`.
- `ncu-view/ncu_view/metric_ref.py` — data module: `FAMILIES` (9 dicts, one
  per Profiling Guide §2.4 family: sid/title/guide/intro/metrics list with
  NVIDIA names + verbatim descriptions; optional `note` for arch-specific
  counter names) + `metric_ref(kp)` builder (rows carry `value`/`str` or
  `present:false`; present counts come from `kp.metrics`/`kp.str_metrics`).
- `ncu-view/ncu_view/modal_app.py` — image, volume, `_run_ncu_sections`,
  `_profile_source` (GPU), `_extract_sections`/`extract_sections` (CPU),
  `profile_on_modal`. NOTE: `_profile_source` still returns bytes via `.remote()`
  — port it to the volume-mount write path when touched.
- `ncu-view/ncu_view/profile.py` — `NCU_DETAIL_SECTIONS`, `NCU_SID_ALIAS`,
  `_sec_filename`, `_export_sections_locally`, run-cmd guessing.
- `ncu-view/ncu_view/html.py` — `kernelPage` (NVIDIA-only: chips + banner +
  NVIDIA sections/rules), `secHtml`, `focusEvidence`/`rulesHtml`, sidebar
  NVIDIA section items; `rlChart`/`rlNote` (SOL card roofline), `samplingBody`
  (real PM config block), `rlTables` (hierarchy + precision tables);
  `memChartHtml`/`memTablesHtml` (memory chart + 6 tables, appended to the
  main Memory section's `memBody`); `buildSearchIndex`/`searchPopHtml`/
  `searchMark`/`searchMove`/`searchGo`/`flash`/`bindSearch` (⌘K palette,
  binds `#global` input + `#search-pop` keynav + row clicks).
- `ncu-view/ncu_view/report.py` — `_ncu_verdict` (SOLBottleneck rule,
  fallback highest severity), `_sm_freq` (SOL "SM Frequency", else ncu's
  formula), dict serialization (`roofline` embedded per kernel).
  `sections.py`/`rules.py` were DELETED.
- `ncu-view/tests/test_ncu_sections.py` — parser/overlay tests (fixture is the
  REAL ncu long-format CSV); `tests/test_render_html.py` — jsdom golden checks
  (incl. memory chart/tables + search palette wiring; note the golden JS
  payload now exceeds the 128 KiB single-exec-arg limit, so `_run_node`
  pipes the program via `node -` stdin); `tests/test_derived.py` — tile values
  + `roofline()` golden values + `memory_model()` golden units/links/tables +
  the raw-CSV contract (metric-only tiles, no section-derived tiles).

## Style

- No comments unless asked. Match existing style (single quotes, 4-space indent,
  JS template literals in `html.py`).
- Never commit unless explicitly asked. Write results as JSON to `$GMN_RESULT_PATH`
  on givemeanode nodes; on Modal, declare verdicts in the function's return.