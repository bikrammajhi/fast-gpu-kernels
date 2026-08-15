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
  harmless. GPU profiling runs under `--set full --launch-skip 1 --launch-count 1`.
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
- **ncu launch selection:** default `launch_skip=None` profiles EVERY
  launch in one pass (`--launch-count 100000 --launch-skip 0` — same replay
  count as count 1, since ncu replays once per counter pass). `ingest`
  dedupes by kernel name (last occurrence = steady state), drops runtime
  plumbing (tensor-pipe util < 5% — torch init/compare, device query,
  memcpy; `NOISE_TENSOR_PCT`/`NOISE_NAMES` in `ingest.py`), and the report
  JS stars the dominant kernel (max time_us). Explicit `--launch-skip N`
  still forces a single launch.
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

- **NVIDIA-first rendering**: when NVIDIA's detailed section table exists
  (`Section.detailed`), it is the primary section; our derived section for that
  topic is hidden behind the "show derived (ours)" checkbox
  (`#derived-toggle` toggles `.derived-sec`, sidebar `.derived-item`). Derived
  data stays in the report JSON. Our derivations are used only where NVIDIA has
  nothing. `NVIDIA_COVER` in `html.py` maps NVIDIA sid → our covered sid.
- Sections are rendered client-side from embedded JSON via JS template
  literals in `html.py` (`kernelPage`, `secHtml`, `rulesHtml`) — `data-sid`,
  `sec-<sid>` ids appear in the RENDERED DOM only after `document.ready()`;
  jsdom checks MUST wait (setTimeout ~400ms) or sections read as missing.
- NVIDIA rule messages get "Focus metrics: name: value (info)" evidence from
  `focus_metrics[].info` (`_ncu_rules_new` in `ingest.py`).
- Verdict: our "LATENCY-BOUND" == NVIDIA "Latency Issue" mapping (B200 device
  auto-detect: 2250 TFLOPS / 8000 GB/s).

## Commands

```bash
cd ncu-view
python3 -m pytest tests/ -q                # 41 tests (all pass)
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
r = build(base + '.ncu-rep')
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
`.sec-body td.l`, `.derived-sec` display none → block after
`#derived-toggle` click, sidebar `.nav-item.derived-item` hidden.

## Key files

- `ncu-view/ncu_view/ingest.py` — `NCU_SECTION_TITLES`, `_ncu_section`,
  `_parse_sec_csv` (long-format + fallback), `_sec_csv_files`,
  `_apply_sec_csvs` (overlay replaces one-liner by sid, appends unknown sids,
  marks `detailed`), `_ncu_rules_new` (focus_info).
- `ncu-view/ncu_view/modal_app.py` — image, volume, `_run_ncu_sections`,
  `_profile_source` (GPU), `_extract_sections`/`extract_sections` (CPU),
  `profile_on_modal`. NOTE: `_profile_source` still returns bytes via `.remote()`
  — port it to the volume-mount write path when touched.
- `ncu-view/ncu_view/profile.py` — `NCU_DETAIL_SECTIONS`, `NCU_SID_ALIAS`,
  `_sec_filename`, `_export_sections_locally`, run-cmd guessing.
- `ncu-view/ncu_view/html.py` — `NVIDIA_COVER`, `kernelPage` (NVIDIA-first),
  `bindDerived`, `focusEvidence`/`rulesHtml`, sidebar derived-item.
- `ncu-view/ncu_view/{model,report}.py` — `Section.detailed`,
  `RuleResult.focus_info`, dict serialization.
- `ncu-view/tests/test_ncu_sections.py` — parser/overlay tests (fixture is the
  REAL ncu long-format CSV); `tests/test_render_html.py` — jsdom golden checks.

## Style

- No comments unless asked. Match existing style (single quotes, 4-space indent,
  JS template literals in `html.py`).
- Never commit unless explicitly asked. Write results as JSON to `$GMN_RESULT_PATH`
  on givemeanode nodes; on Modal, declare verdicts in the function's return.