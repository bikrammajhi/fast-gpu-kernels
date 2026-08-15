# ncu-view

Nsight-Compute-style kernel profile reports as a self-contained web page —
from any profile input, for any NVIDIA GPU, for any kind of CUDA kernel.

Point it at an ncu profile and read the result like Nsight Compute's GUI: a
**Summary** page with prioritized recommendations and a per-kernel
**Details** page with Speed Of Light, Occupancy, Scheduler, Warp State,
Compute/Memory Workload, Instruction, PM Sampling and NvLink sections —
each with the kernel's bottleneck verdict.

Or point it at *source code* and it runs the kernel on your Modal account,
profiles it with ncu, and renders the report — one command, no local ncu
installation needed.

## Features

- **Any input, one report.** `.ncu-rep` (NVIDIA's own format, read via their
  `ncu_report` module), `ncu --page raw --csv` dumps, or our JSON — all
  produce the same HTML + JSON report.
- **NVIDIA-first fidelity.** When the profile is a `.ncu-rep`, the report
  embeds NVIDIA's *own* detailed section tables (exported via
  `ncu --import <rep> --section <X> --csv`), labeled with a `NVIDIA` source
  tag. Our derived analysis for the same topic stays in the report, hidden
  behind a **"show derived (ours)"** toggle — NVIDIA's tables primary, ours
  one click away. Our derivations are used only where NVIDIA has nothing.
- **Device-accurate everywhere.** Peak specs (TFLOPS, DRAM bandwidth,
  per-SM limits) come from the profile's own device attributes, so numbers
  are right on any NVIDIA GPU: T4 → B300. You can also force a spec with
  `--gpu` or override peaks with `--config tensor_peak=...`.
- **Profiles *any* kernel kind that ncu can profile:** raw CUDA (`.cu`),
  CUTE C++ (cute/tensor.hpp), CUTLASS C++, and the CuTe DSL Python drivers
  (`cutlass.cute`). Single files, directories, Makefile projects.
- **Every Modal accelerator, selected per run:** `--modal-gpu` accepts the
  full Modal catalog — `T4 L4 A10 L40S A100 A100-40GB A100-80GB
  RTX-PRO-6000 H100 H200 B200 B200+ B300`. The compile arch
  (`-arch=sm_XXa`) and the CuTe DSL arch (`CUTE_DSL_ARCH`) are detected from
  the actual GPU automatically.
- **Honest numbers.** Nothing is fabricated: rows read from ncu counters
  verbatim; derived rows carry their formula in the report (tooltip);
  counters never collected show `n/a (not collected)` — never a zero.
- **Verdicts.** Each kernel gets a decision-table verdict: `LATENCY-BOUND`,
  `PIPE-BOUND`, `ISSUE-SERIALIZATION`, `LOAD-PATH`, `COMPILER-OPACITY`,
  `CONVERGED`/`CONVERGING`, or `REFERENCE` (cuBLAS) — computed from measured
  counters and their deltas vs the previous kernel in the series.
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
ncu-view report  results/ncu_counters.json            # writes .html + .json next to the input
ncu-view report  report.ncu-rep -o out/ --open        # .ncu-rep also carries NVIDIA's own sections
ncu-view summary report.ncu-rep                      # terminal verdict table
ncu-view serve   report.ncu-rep --port 8000          # live view: /report.html, /report.json
```

`report` is the default command, so `ncu-view --path foo.ncu-rep -o out/`
does the same thing.

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

Profiling auto-captures every launch in one pass (same replay count as
profiling one — ncu replays once per counter pass) and the report stars
the dominant kernel (max total time, the steady-state benchmark loop),
so any app profiles correctly with no launch-order knowledge. Runtime
plumbing (torch init/compare, device query, memcpy — anything that never
drives the tensor pipe, <5% tensor-pipe utilization) is dropped
(`noise_dropped` in the report meta). Pass `--launch-skip N` to force a
specific launch; apps with fewer launches still profile (launch 0 is
always captured). Tune with `--launch-skip N` /
`--launch-count N` (`>1` averages
`gpu__time_duration` over several steady-state launches).

**Clock fairness.** ncu's own default is `--clock-control base`, which locks
the GPU to its base clock during profiling — that understates steady-state
throughput by ~30-45% (e.g. a kernel that does 1790 TF/s at boost reads
~1215 TF/s at base). `ncu-view profile` therefore defaults to
`--clock-control boost` (the reproducible peak number reviewers expect); use
`--clock-control none` to let the app's own warm-up drive clocks, or `base`
to reproduce vanilla ncu. If the host refuses `boost`, the run retries at
`none` and the report's console notes it. Every kernel's measured SM clock
(`sm__cycles_elapsed / gpu__time_duration`) is shown on its stat strip, so
each TFLOPS is self-documenting.

**Fair cuBLAS baseline.** `--compare-cublas` additionally profiles a cuBLAS
`cublasGemmEx` GEMM in the same run, under identical `--clock-control`,
`--launch-skip`/`--launch-count` and shape (`--M`, default 8192) — both land
in the same report series and TFLOPS chart, so your kernel and cuBLAS are
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

- **Same, disclosed clock** — `--clock-control boost` (or locked `none`); the
  report's per-kernel SM-clock chip makes the clock explicit so a base-clock
  capture can never be mistaken for a boost-clock one.
- **Same precision** — fp16/bf16 have identical dense rate on Blackwell tensor
  cores, but the baseline's `--bench-precision` must match the kernel's dtype.
- **Same shape & layout** — 8192³, fp32 accumulate, row-major, no sparsity.
- **Steady state** — auto mode profiles every launch and the report stars
  the dominant one; use `--launch-skip N` to skip warm-up or
  `--launch-count >1` and report min and mean±std, not a single snapshot.
- **% of theoretical peak** — the report computes TFLOPS against the detected
  device's dense peak (e.g. B200 = 2250 TF/s, so ~1770 TF/s ≈ 79%).
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

- **tier 1** — rows read from ncu counters, verbatim.
- **tier 2** — derived rows (TFLOPS, TMA %, DRAM %, IPC…) carry their
  formula in the report (`derived` marker + tooltip) and are computed from
  measured counters only, against the detected device's peak specs.
- **tier 3** — rules. `our-rule` rows are documented heuristics; `NVIDIA rule`
  rows are the rule engine from the `.ncu-rep` itself, shown alongside with
  their "Focus metrics" evidence. Counters that were never collected show
  `n/a (not collected)` — never a zero.

## Verdicts

The per-kernel verdict is a decision table over the measured counters and
their deltas vs the previous kernel in the series:

| verdict | reads as |
|---|---|
| LATENCY-BOUND | total stall > 80 cyc/issue, tensor pipe < 40% — waiting, not working |
| COMPILER-OPACITY | more instructions while getting faster — layout facts hidden from the compiler |
| PIPE-BOUND | pipe busy < 92%, nothing else saturated — per-tile overhead |
| ISSUE-SERIALIZATION | stall total frozen while the pipe moved — issue-side serialization |
| LOAD-PATH | stall 45–80, long-scoreboard ≥ 5%, pipe < 50% — the LSU path throttles |
| CONVERGED / CONVERGING | pipe ≥ 90% and time flat — the plateau; caveat when the stall denominator moved |
| REFERENCE | cuBLAS: the bar to compare the series against |

## CLI reference

```
ncu-view [command] [options] INPUT...

commands: report (default) | summary | serve | view | profile

common options
  -o, --outdir DIR        where to write the report (default: next to input)
  --kernel-regex REGEX    only analyze kernels matching REGEX
  --config k=v            override a config key, e.g. --config M=4096
                          --config tensor_peak=1800 (repeatable)
  --gpu NAME              force a device spec, e.g. --gpu 'H100 SXM'
  --M N                   matrix size for TFLOPS (default 8192)
  --open                  open the report in the browser (report/view)
  --port N                serve port (default 8000)

profile options
  --build-cmd CMD         command that builds+runs the kernel
                          (default: guessed from the source)
  --modal-gpu ACCEL       Modal accelerator: T4 L4 A10 L40S A100 A100-40GB
                          A100-80GB RTX-PRO-6000 H100 H200 B200 B200+ B300
                          (default H100)
  --timeout N             Modal run timeout in seconds (default 1200)
  --launch-skip N         ncu launch skip (default: auto — profile every
                          launch; the report stars the dominant kernel)
  --launch-count N        ncu launches to capture; >1 averages steady-state
                          (default 1)
  --clock-control MODE    ncu clock control: boost | none | base (default
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
python3 -m pytest tests/ -q          # 41 tests
python3 tests/test_against_ncu.py ../kernels/cute_dsl/B200/results/golden/matmul_v1.ncu-rep \
    ../kernels/cute_dsl/B200/results/golden/matmul_v1.raw.csv
# the ingest-vs-ncu harness: both ingests must agree, or ncu-view has a bug
```

The golden artifacts (`kernels/cute_dsl/B200/results/golden/`) are real ncu
profiles of the `Beating cuBLAS on B200` cute-dsl matmul, with NVIDIA's
section exports; they pin the rendering and the verdicts.

## License

MIT