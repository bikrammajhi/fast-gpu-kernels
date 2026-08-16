# ncu-view

Nsight-Compute-style kernel profile reports as a self-contained web page —
from any profile input, for any NVIDIA GPU, for any kind of CUDA kernel.

Point it at an ncu profile and read the result like Nsight Compute's GUI: a
**Summary** page with prioritized recommendations and a per-kernel
**Details** page with the sections NVIDIA itself exports — Speed Of Light,
Occupancy, Scheduler, Warp State, Compute/Memory Workload, Instruction,
PM Sampling and NvLink — and NVIDIA's rule-engine findings on top.
All big numbers are shown with smart units (1.100 TFLOP, 453.62 TFLOP/s,
7.67 TB/s) and the UI wears a terminal theme (monospace, phosphor green,
☼ toggles light).

Or point it at *source code* and it runs the kernel on your Modal account,
profiles it with ncu, and renders the report — one command, no local ncu
installation needed.

## Features

- **Any input, one report.** `.ncu-rep` (NVIDIA's own format, read via their
  `ncu_report` module), `ncu --page raw --csv` dumps, or our JSON — all
  produce the same HTML + JSON report.
- **Official NVIDIA data, plus derived metrics marked ours.** When the
  profile is a `.ncu-rep`, the report embeds NVIDIA's *own* detailed section
  tables (exported via `ncu --import <rep> --section <X> --csv`), labeled
  with a `NVIDIA` source tag, and NVIDIA's rule-engine recommendations
  verbatim — including their severity, message and focus metrics. On top of
  that, a **Derived metrics** card (tagged `ours`) computes things the
  profiler doesn't give directly — FLOPs (FMA-pipe and tensor-ops-path),
  achieved FLOP/s, arithmetic intensity, CTAs launched, instructions per
  thread, kernel-wide IPC, duration per CTA, occupancy utilization,
  warp-slot stall share, implied SM clock, DRAM/FMA/tensor bandwidths,
  PM-sampler configuration and more — each with a one-line definition and
  computed from NVIDIA's own exported rows and raw counters with its
  formula and source rows shown; nothing is assumed, and a metric whose
  inputs are missing is skipped. Tiles are grouped by what they measure
  (Compute → Memory → Roofline → Occupancy & Scheduling → Timing → PM
  Sampling) and clicking a tile expands its formula/sources. The Tensor
  FLOPS tile is always present — even on FMA-only kernels, where it
  honestly reads 0.
- **The roofline is computed, not sketched.** The Speed Of Light card draws
  the log-log roofline with the real achieved point (FMA FLOPs from NVIDIA's
  SASS counters **plus tensor-core FLOPs from NVIDIA's own
  `sm__ops_path_tensor_*` FLOP-path counters** ÷ DRAM bytes), the
  memory-roof slope (NVIDIA achieved bandwidth ÷ NVIDIA %-of-peak), the
  compute roof (the tensor ops-path peak when the kernel uses tensor cores,
  else the FMA-pipe peak) and the ridge point — plus a memory-hierarchy
  table (L1/L2/DRAM achieved vs derived peak, AI per level) and a
  per-precision FLOP accounting (FMA rows + tensor rows with real FLOPs).
  No MMA shape is ever assumed. The per-kernel banner *is* NVIDIA's Speed
  Of Light
  bottleneck rule. Inputs without exported sections (raw CSVs, counters
  JSON) get the summary chips and an honest note that NVIDIA sections are
  missing — never a fabricated section.
- **NVIDIA-accurate everywhere.** Every "% of peak" (DRAM Throughput,
  Tensor pipe, occupancy) is NVIDIA's own `pct_of_peak_sustained` value
  from the profile — ncu computes peaks from the device itself at profile
  time, so numbers are right on any NVIDIA GPU: T4 → B300. No device
  specs are hardcoded or assumed; the device name is reported as the
  profile states it.
- **Profiles *any* kernel kind that ncu can profile:** raw CUDA (`.cu`),
  CUTE C++ (cute/tensor.hpp), CUTLASS C++, and the CuTe DSL Python drivers
  (`cutlass.cute`). Single files, directories, Makefile projects.
- **Every Modal accelerator, selected per run:** `--modal-gpu` accepts the
  full Modal catalog — `T4 L4 A10 L40S A100 A100-40GB A100-80GB
  RTX-PRO-6000 H100 H200 B200 B200+ B300`. The compile arch
  (`-arch=sm_XXa`) and the CuTe DSL arch (`CUTE_DSL_ARCH`) are detected from
  the actual GPU automatically.
- **Honest numbers.** Nothing is fabricated: rows read from ncu counters
  verbatim; counters never collected show `n/a (not collected)` — never a
  zero. The stall figure is NVIDIA's own metric — the sum of its per-reason
  stall counters; the SM clock is NVIDIA's "SM Frequency" when exported.
- **Self-contained output.** The HTML embeds the report JSON — open it from
  disk, email it, or serve it with `ncu-view serve`.

## Install

```bash
pip install ncu-view            # everything except .ncu-rep reading
pip install "ncu-view[ncu]"     # + NVIDIA's ncu_report reader for .ncu-rep
pip install "ncu-view[modal]"   # + the `profile` command (Modal client)
pip install "ncu-view[all]"
```

The `profile` command also needs a [Modal](https://modal.com) account —
`pip install "ncu-view[modal]"` then `modal token new`. The container image
(which installs ncu, torch, the cutlass headers and the CuTe DSL) is built
and cached automatically on first use.

## Quickstart

Build a report from any profile input:

```bash
ncu-view report  results/ncu_counters.json
# → creates results/ncu_counters-ncu-report/ncu_counters.html + .json
#   (a "<kernel>-ncu-report/" folder next to the input)

ncu-view report  report.ncu-rep --open
# → creates report-ncu-report/report.html + .json
#   (.ncu-rep also carries NVIDIA's own sections)

ncu-view report  report.ncu-rep -o out/             # -o overrides the folder
ncu-view summary report.ncu-rep                    # terminal verdict table
ncu-view serve   report.ncu-rep --port 8000        # live view: /report.html, /report.json
```

`report` is the default command, so the word is optional:
`ncu-view foo.ncu-rep` is the same as `ncu-view report foo.ncu-rep`.

## Profile anything with one command

Point `profile` at a CUDA source and get a full ncu report back, profiled on
your Modal account:

```bash
# raw CUDA / CUTE / CUTLASS .cu files — compile + profile + render
ncu-view profile matmul.cu -o out/
ncu-view profile kernels/cute/H100/matmul_v1.cu -o out/

# CuTe DSL Python drivers — same command, same pipeline
ncu-view profile kernels/cute_dsl/H100/kernels/run.py -o out/

# pick the GPU (default H100); the report renders against the actual device
ncu-view profile matmul.cu --modal-gpu B200 -o out/
ncu-view profile kernels/cute_dsl/B200/matmul_v1.py --modal-gpu B200 -o out/

# directories: the tool guesses how to build/run
ncu-view profile kernels/cutlass -o out/
ncu-view profile . --build-cmd 'make && ./run' -o out/   # Makefile projects

# run ncu locally instead (needs ncu on PATH)
ncu-view profile src -o out/ --no-modal
```

Omitting `-o` uses the same convention as `report`: everything lands in a
`<source-name>-ncu-report/` folder next to the source.

How the source is run when you don't pass `--build-cmd`:

| source | default run command |
|---|---|
| single `.py` (CuTe DSL driver) | `python3 <file>.py` |
| single `.cu` | `nvcc -arch=<detected> -O3 -lcublas <cutlass -I flags> -o /tmp/run <file> && /tmp/run` |
| directory with `.cu` files | compile all of them together, run the binary |
| directory with a single `.py` | `python3 <file>.py` |
| anything else | ask for `--build-cmd` |

`-arch` is pinned to the *actual* device (e.g. `sm_90a` on H100, `sm_100a`
on B200) — plain `-arch=native` yields `sm_90` without the `a` and trips
cute's `CUTE_ARCH_MMA_SM90A_ENABLED` assert. The same detection sets
`CUTE_DSL_ARCH` for CuTe DSL kernels. CUTLASS C++ headers come from a pinned
latest-tag clone (`/opt/cutlass`), with both `-I .../include` and
`-I .../tools/util/include` (the `util` helpers — `print_error.hpp`,
`GPU_Clock.hpp` — live under `tools/util` in 3.x and 4.x alike).

Profiling captures **one kernel** by default (warmup launch skipped): ncu
replays the app once per counter pass, so a sweep benchmark with hundreds of
launches stays cheap. `--launch-skip N` lands on a specific launch, and
`--launch-count >1` averages `gpu__time_duration` over several
steady-state launches. The run shows live progress: every step
(container start, upload, ncu capture, raw CSV, NVIDIA sections) prints
with elapsed time and a completion bar, so nothing runs dark — and if a
step fails, the error is printed instead of dying silently.

**Clock fairness.** `ncu-view profile` defaults to `--clock-control base` —
ncu's own default, so a plain run reproduces vanilla ncu exactly. Note that
base locks the GPU to its base clock, which understates steady-state
throughput vs boost (e.g. a kernel that does 1790 TF/s at boost reads
~1215 TF/s at base). Pass `--clock-control boost` for the reproducible peak
number reviewers expect (where the host allows it; if the host refuses
`boost`, the run retries at `none` and the console notes it), or
`--clock-control none` to let the app's own warm-up drive clocks. Every
kernel's measured SM clock
(`sm__cycles_elapsed / gpu__time_duration`) is shown on its stat strip, so
the capture clock is self-documenting.

**Fair cuBLAS baseline.** `--compare-cublas` additionally profiles a cuBLAS
`cublasGemmEx` GEMM in the same run, under identical `--clock-control`,
`--launch-skip`/`--launch-count` and shape (`--M` — required, no assumed
default) — both land
in the same report series, so your kernel and cuBLAS are
compared at the same clock. `--bench-precision fp16|bf16` matches the
baseline's io dtype to your kernel (default fp16).

## Cross-GPU and cross-ecosystem compatibility

`--compare-cublas` is orthogonal to the kernel kind being profiled and to the
GPU — it is a standalone `cublasGemmEx` program recompiled per-arch on each
node, so it works across **raw CUDA, CUTE C++, CUTLASS and CuTe DSL** sources,
and across the **whole Modal catalog** (T4, L4, A10, L40S, A100, A100-40GB,
A100-80GB, RTX-PRO-6000, H100, H200, B200, B200+, B300). Two caveats:

- **bf16 baseline needs sm_80+** (A100 and newer). Only T4 (sm_75) in the
  catalog lacks bf16 tensor GEMM, so `--bench-precision bf16` fails there;
  fp16 works on every GPU in the catalog (sm_75+).
- **Match the dtype.** The comparison is only as fair as the precision match.
  `--bench-precision` should equal your kernel's `io_dtype` (e.g. fp16 for
  `matmul_v6.py`).

## Benchmark methodology (what reviewers accept)

For a "reaching cuBLAS-like performance" claim the community expects numbers
that are clock-fair, precision-matched and reproducible:

- **Same, disclosed clock** — the same `--clock-control` mode for every
  kernel compared (default `base`); the
  report's per-kernel SM-clock chip makes the clock explicit so a base-clock
  capture can never be mistaken for a boost-clock one.
- **Same precision** — fp16/bf16 have identical dense rate on Blackwell tensor
  cores, but the baseline's `--bench-precision` must match the kernel's dtype.
- **Same shape & layout** — 8192³ (the size you pass with `--M`), fp32
  accumulate, row-major, no sparsity.
- **Steady state** — one kernel is captured by default (warmup launch
  skipped); use `--launch-skip N` to land on a specific launch or
  `--launch-count >1` and report min and mean±std, not a single snapshot.
- **% of peak** — the report's bottleneck rows (DRAM Throughput, tensor
  pipe, occupancy) are NVIDIA's own `pct_of_peak_sustained` values from the
  profile (ncu computes peaks from the device at profile time; when the raw
  counter is absent, NVIDIA's Speed Of Light "DRAM Throughput" row is used).
  Nothing is divided by a hardcoded spec; absent data degrades to an honest
  `n/a` instead of a fabricated number. Reference: NVIDIA's Nsight Compute
  Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- **Pinned toolchain** — the report records the device; pin CUTLASS/cute-dsl,
  CUDA, ncu and driver versions in the write-up.
- **A same-condition cuBLAS baseline** — `--compare-cublas` benchmarks cuBLAS
  in the same run, same clock, rather than quoting a vendor figure.

## Inputs

| input | how it's produced | fidelity |
|---|---|---|
| `results/ncu_counters.json` | `ncu_profile_all.py` probe set | tier 1/2 |
| `ncu --page raw --csv` dump | `ncu --set full --page raw --csv` | tier 1/2 |
| `report.ncu-rep` | `ncu -o report.ncu-rep --set full …` | tier 1/2 + NVIDIA's own sections and rule engine (`ncu_report`) |

### How the NVIDIA sections get in

`.ncu-rep` is NVIDIA's proprietary container — the full section tables are
re-exported with the ncu CLI itself:

```bash
ncu --import report.ncu-rep --section WarpStateStats --page details --csv
```

`ncu-view` runs this for the detailed sections (SpeedOfLight, Occupancy,
WarpStateStats, SchedulerStats, ComputeWorkloadAnalysis, SourceCounters,
PM Sampling) whenever a `.ncu-rep` is ingested, and overlays the results
under each section's `NVIDIA` source tag. On Modal this happens in the
container before the artifacts return; a CPU-only helper
(`extract_sections(rep_key, run_id, dest)`) can re-export a report already
on the Modal volume.

## Accuracy model

Everything in the report is NVIDIA's own data, straight from the profile:

- **Sections** — NVIDIA's exported section tables, verbatim, under the
  `NVIDIA` source tag. A section that has no table rows (ncu printed
  "No metrics to show") keeps NVIDIA's one-line description.
- **Rules** — NVIDIA's rule-engine results from the `.ncu-rep` itself,
  shown with their severity, message and "Focus metrics" evidence. Counters
  that were never collected show `n/a (not collected)` — never a zero.
- **Summary chips** — NVIDIA counters read verbatim: each "% of peak" is
  ncu's own `pct_of_peak_sustained` value, the SM clock is NVIDIA's
  "SM Frequency" (same formula from its own counters when not exported),
  and the stall figure is NVIDIA's metric — the sum of its per-reason
  stall counters.
- **No tier 2 / tier 3 of our own.** No derived rows, no heuristics, no
  decision table. Inputs that carry no NVIDIA sections (raw CSV dumps,
  counters JSON) render the chips and an honest note — nothing invented.

## Verdict

The per-kernel banner is NVIDIA's Speed Of Light bottleneck rule
("Bottleneck: Latency Issue" etc.), verbatim — the same verdict Nsight
Compute's GUI shows. When the rule engine reported no SOL bottleneck, the
banner shows NVIDIA's most severe rule for the kernel. There is no verdict
of our own.

## CLI reference

```
ncu-view [command] [options] INPUT...

commands: report (default) | summary | serve | view | profile

common options
  -o, --outdir DIR        where to write the report (default: a
                          <kernel>-ncu-report/ folder next to the input)
  --kernel-regex REGEX    only analyze kernels matching REGEX
  --M N                   profile: square GEMM size for the --compare-cublas
                          reference run (required with --compare-cublas)
  --open                  open the report in the browser (report/view)
  --port N                serve port (default 8000)

profile options
  --build-cmd CMD         command that builds+runs the kernel
                          (default: guessed from the source)
  --modal-gpu ACCEL       Modal accelerator: T4 L4 A10 L40S A100 A100-40GB
                          A100-80GB RTX-PRO-6000 H100 H200 B200 B200+ B300
                          (default H100)
  --timeout N             Modal run timeout in seconds (default 1200)
  --launch-skip N         ncu launch skip (default 1: skip the warmup
                          launch and profile ONE kernel)
  --launch-count N        ncu launches to capture (default 1 — one kernel
                          at a time); >1 averages steady-state
  --clock-control MODE    ncu clock control: base | boost | none (default
                          boost; ncu's own base understates throughput ~30-45%)
  --compare-cublas        also profile a cuBLAS GEMM under identical flags,
                          same report series (fair, same-clock baseline)
  --bench-precision P     baseline io dtype: fp16 | bf16 (default fp16; bf16
                          needs sm_80+)
  --no-modal              run ncu locally instead of on Modal
```

## Development

```bash
cd ncu-view
python3 -m pytest tests/ -q          # 36 tests
python3 tests/test_against_ncu.py ../kernels/cute_dsl/B200/results/golden/matmul_v1.ncu-rep \
    ../kernels/cute_dsl/B200/results/golden/matmul_v1.raw.csv
# the ingest-vs-ncu harness: both ingests must agree, or ncu-view has a bug
```

The golden artifacts (`kernels/cute_dsl/B200/results/golden/`) are real ncu
profiles of the `Beating cuBLAS on B200` cute-dsl matmul, with NVIDIA's
section exports; they pin the rendering and NVIDIA's rule results.

## License

MIT