"""Derived metrics: ncu-view calculations over NVIDIA's own exported rows.

Every metric here is computed from section rows and counters that NVIDIA
itself exported from the profile (LaunchStats, SpeedOfLight, Occupancy,
InstructionStats, SchedulerStats, WarpStateStats, dram__*, smsp__*,
l1tex__*, lts__*, profiler__pmsampler_*). No hardware specifications are
assumed and no values are invented: a metric whose inputs are missing is
skipped. Each metric carries its formula, its source rows, and is tagged
``src=ours`` so renderers can label it as a ncu-view calculation rather
than NVIDIA data.
"""

from __future__ import annotations

from contextlib import suppress

from .model import KernelProfile

OURS_SRC = "ours"

# Display groups, in render order. Tiles are emitted group by group.
GROUPS = ["COMPUTE", "MEMORY", "ROOFLINE", "OCCUPANCY & SCHEDULING",
          "TIMING", "PM SAMPLING"]


def _row(kp: KernelProfile, sid: str, prefix: str):
    for sec in kp.ncu_sections:
        if sec.sid != sid:
            continue
        for r in sec.rows:
            if r.label.lower().startswith(prefix.lower()):
                return r
    return None


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


_TIME_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}


def _time_s(row) -> float | None:
    """Duration row value in seconds, honoring the row's own unit."""
    if row is None:
        return None
    scale = _TIME_SCALE.get(str(row.unit or "").strip().lower())
    n = _num(row.value)
    return n * scale if (n is not None and scale is not None) else None


def _add(out: list[dict], name, value, unit, formula, sources, note=None,
         desc=None, group=None):
    if value is None:
        return
    d = {"name": name, "value": value, "unit": unit, "formula": formula,
         "sources": sources, "src": OURS_SRC}
    if note:
        d["note"] = note
    if desc:
        d["desc"] = desc
    if group:
        d["group"] = group
    out.append(d)


def _m(kp: KernelProfile, name: str) -> float | None:
    v = kp.metrics.get(name)
    if v is None:
        suffix = "." + name
        for k, val in kp.metrics.items():
            if k.endswith(suffix):
                v = val
                break
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sm_freq(kp: KernelProfile) -> float | None:
    r = _row(kp, "SpeedOfLight", "SM Frequency")
    if r is not None and _num(r.value) is not None:
        return _num(r.value)
    cycles = _m(kp, "sm__cycles_elapsed.avg")
    dur_ns = _m(kp, "gpu__time_duration.avg")
    if cycles and dur_ns:
        return cycles / (dur_ns * 1e-9) / 1e9
    return None


def _dur_s(kp: KernelProfile) -> float | None:
    dur_ns = _m(kp, "gpu__time_duration.avg")
    if dur_ns:
        return dur_ns * 1e-9
    return _time_s(_row(kp, "SpeedOfLight", "Duration"))


FMA_OPS = [
    ("FP32 FMA (ffma)", "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
     "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"),
    ("FP16 FMA (hfma)", "derived__sm__sass_thread_inst_executed_op_hfma_pred_on_x4",
     "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained"),
    ("FP64 FMA (dfma)", "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2",
     "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"),
]

# Tensor-core SASS instruction counters, per architecture (the counter names
# differ between generations: utcmma on Blackwell, shared_gmma/wgmma on
# Hopper, mma on Ampere). The first present, non-zero counter per family
# wins; families whose counters are absent or zero are skipped.
TENSOR_OPS = [
    ("utcmma (tcgen05)", ["smsp__sass_inst_executed_op_utcmma.sum"]),
    ("shared_gmma (wgmma)", ["smsp__sass_inst_executed_op_shared_gmma.sum",
                             "smsp__sass_inst_executed_op_wgmma.sum"]),
    ("mma (mma.sync)", ["smsp__sass_inst_executed_op_mma.sum",
                        "smsp__sass_inst_executed_op_hmma.sum"]),
    ("tma_ld", ["smsp__sass_inst_executed_op_tma_ld.sum"]),
    ("tma_st", ["smsp__sass_inst_executed_op_tma_st.sum"]),
]


def _fma_counts(kp: KernelProfile) -> list[dict]:
    out = []
    for name, counter, _peak in FMA_OPS:
        v = _m(kp, counter)
        if v is not None:
            out.append({"name": name, "flops": v, "counter": counter})
    return out


def _fma_peaks(kp: KernelProfile) -> list[dict]:
    """Peak FMA-pipe rates from NVIDIA's own peak_sustained database.

    peak_sustained is NVIDIA's per-SM peak rate for the counter; for these
    SASS op counters it is the peak FMA instruction rate across all SMs, so
    peak FLOP/s = peak_insts × 2 (each FMA is two FLOPs) × SM frequency.
    """
    freq = _sm_freq(kp)
    if not freq:
        return []
    out = []
    for name, _counter, peak in FMA_OPS:
        p = _m(kp, peak)
        if p is not None:
            out.append({"name": name, "peak_insts": p,
                        "flops_s": p * 2.0 * freq * 1e9,
                        "formula": f"peak_sustained {p:g} × 2 FLOP/inst × "
                                   f"SM freq {freq:g} GHz"})
    return out


def _big(x: float) -> str:
    """Render a number compactly for formula strings: exponential above 1e6."""
    if x is None:
        return "—"
    return f"{x:.4g}" if abs(x) >= 1e6 else f"{x:g}"


def _tensor_paths(kp: KernelProfile, dur: float | None) -> list[dict]:
    """NVIDIA's own tensor FLOP counters: sm__ops_path_tensor_<fmt>_dst_<fmt>.

    Per-precision FLOP paths executed on the tensor cores (the '_op_' family
    is the same FLOPs split per SASS opcode — exclude it to avoid counting
    twice). Each entry carries NVIDIA's %-of-peak, from which the peak rate
    is backed out (or read directly when the rep exports it).
    """
    hits = sorted(
        n for n in kp.metrics
        if n.startswith("sm__ops_path_tensor_src_")
        and "_op_" not in n and n.endswith(".sum")
    )
    rows = []
    for counter in hits:
        flops = _m(kp, counter)
        if flops in (None, 0.0):
            continue
        label = counter[len("sm__ops_path_tensor_"):-len(".sum")]
        pct = _m(kp, counter + ".pct_of_peak_sustained_elapsed")
        peak = _m(kp, counter + ".peak_sustained_elapsed.per_second")
        if peak is None and pct and dur:
            peak = (flops / dur) / (pct / 100.0)
        row = {"name": label, "flops": flops, "counter": counter}
        if dur:
            row["flops_s"] = flops / dur
        if pct is not None:
            row["pct"] = pct
        if peak is not None:
            row["peak_flops_s"] = peak
        rows.append(row)
    return rows


def roofline(kp: KernelProfile) -> dict:
    """The computed roofline model: achieved point + envelope + memory levels.

    Every number comes from NVIDIA's own counters/rows (formulas are shown
    in the rendered card). Tensor-core FLOPs come from NVIDIA's own
    sm__ops_path_tensor_* FLOP-path counters, so the achieved point covers
    the whole FMA + tensor stream without any assumed MMA shape.
    """
    out: dict = {}
    dur = _dur_s(kp)
    read = _m(kp, "dram__bytes_read.sum")
    write = _m(kp, "dram__bytes_write.sum")
    dram_bw = _m(kp, "dram__bytes.sum.per_second")
    dram_pct = None
    r = _row(kp, "SpeedOfLight", "DRAM Throughput")
    if r is not None and _num(r.value) is not None:
        dram_pct = _num(r.value)
    if dram_pct is None:
        v = _m(kp, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        if v is not None:
            dram_pct = v
    fma = _fma_counts(kp)
    peaks = _fma_peaks(kp)
    fma_flops = sum(f["flops"] for f in fma) if fma else None
    tensor = _tensor_paths(kp, dur)
    tensor_flops = sum(t["flops"] for t in tensor) if tensor else None
    total_flops = None
    if fma_flops is not None and tensor_flops is not None:
        total_flops = fma_flops + tensor_flops
    elif fma_flops is not None:
        total_flops = fma_flops

    if total_flops is not None and dur:
        out["achieved"] = {
            "flops": total_flops,
            "flops_s": total_flops / dur,
            "fma": [{**f, "flops_s": f["flops"] / dur} for f in fma],
            "tensor": [{**t, "flops_s": t["flops"] / dur}
                       for t in tensor if t.get("flops_s")],
        }
    if read is not None and write is not None:
        out["achieved_bytes"] = {"read": read, "write": write,
                                 "total": read + write}
    if dram_bw is not None and dram_pct:
        out["envelope"] = {
            "dram_bw": dram_bw,
            "dram_pct": dram_pct,
            "peak_dram_bw": dram_bw / (dram_pct / 100.0),
            "formula": "NVIDIA dram__bytes.sum.per_second ÷ (NVIDIA "
                       f"DRAM Throughput {dram_pct:g}% / 100)",
        }
    if peaks:
        env = out.setdefault("envelope", {})
        env["fma_peaks"] = peaks
        env["peak_fma_flops"] = sum(p["flops_s"] for p in peaks)
        env["fma_peak_formula"] = "Σ (NVIDIA peak_sustained insts × 2 FLOP/inst × SM frequency)"
    if tensor:
        env = out.setdefault("envelope", {})
        env["tensor_peaks"] = [t for t in tensor if t.get("peak_flops_s")]
        env["peak_tensor_flops"] = sum(t["peak_flops_s"] for t in tensor
                                       if t.get("peak_flops_s"))
        env["tensor_peak_formula"] = ("Σ NVIDIA ops-path FLOPs ÷ "
                                      "(NVIDIA %-of-peak / 100)")
    env = out.get("envelope")
    ach = out.get("achieved")
    bts = out.get("achieved_bytes")
    if env and ach and "peak_dram_bw" in env:
        peak_compute = (env.get("peak_tensor_flops")
                        or env.get("peak_fma_flops"))
        if peak_compute:
            env["peak_compute_flops"] = peak_compute
            env["peak_compute_source"] = ("tensor" if env.get("peak_tensor_flops")
                                          else "fma")
            env["ridge"] = peak_compute / env["peak_dram_bw"]
    if ach and bts and bts.get("total"):
        ai = ach["flops"] / bts["total"]
        ach["ai"] = ai
        if env and "peak_dram_bw" in env:
            ach["roof_flops_s_at_ai"] = env["peak_dram_bw"] * ai
    if dur and bts and bts.get("total"):
        out["levels"] = []
        l1_read = _m(kp, "l1tex__m_xbar2l1tex_read_bytes.sum")
        l1_write = _m(kp, "l1tex__m_l1tex2xbar_write_bytes.sum")
        l1_pct = _m(kp, "l1tex__m_xbar2l1tex_read_bytes.sum.pct_of_peak_sustained_elapsed")
        l2 = _m(kp, "lts__t_sectors.sum")
        l2_pct = _m(kp, "lts__t_sectors.sum.pct_of_peak_sustained_elapsed")
        levels = []
        if l1_read is not None and l1_write is not None:
            levels.append(("L1 (SM↔L2)", l1_read + l1_write, l1_pct,
                           "xbar2l1tex_read + l1tex2xbar_write bytes"))
        if l2 is not None:
            levels.append(("L2", l2 * 32.0, l2_pct,
                           "lts__t_sectors × 32 B/sector"))
        levels.append(("DRAM", bts["total"], dram_pct,
                       "dram__bytes_read + dram__bytes_write"))
        res = []
        for name, bytes_, pct, formula in levels:
            level = {"level": name, "bytes": bytes_, "bw": bytes_ / dur,
                     "formula": formula}
            if pct is not None:
                level["pct"] = pct
                level["peak_bw"] = (bytes_ / dur) / (pct / 100.0)
            if total_flops:
                level["ai"] = total_flops / bytes_
            res.append(level)
        if res:
            out["levels"] = res
    return out


def derive(kp: KernelProfile) -> list[dict]:
    out: list[dict] = []

    def v(sid, prefix):
        r = _row(kp, sid, prefix)
        return _num(r.value) if r else None

    grid = v("LaunchStats", "Grid Size")
    block = v("LaunchStats", "Block Size")
    elapsed = v("SpeedOfLight", "Elapsed Cycles")
    dur_s = _time_s(_row(kp, "SpeedOfLight", "Duration"))
    inst = v("InstructionStats", "Executed Instructions")
    ach = v("Occupancy", "Achieved Occupancy")
    theo = v("Occupancy", "Theoretical Occupancy")
    stall_cyc = v("WarpStateStats", "Warp Cycles Per Issued Instruction")
    issued = v("SchedulerStats", "Issued Warp Per Scheduler")

    threads = grid * block if (grid is not None and block is not None) else None

    COMPUTE = "COMPUTE"
    MEMORY = "MEMORY"
    ROOFLINE = "ROOFLINE"
    OCC = "OCCUPANCY & SCHEDULING"
    TIMING = "TIMING"
    PMSAMP = "PM SAMPLING"

    rl = roofline(kp)
    bts = rl.get("achieved_bytes") or {}
    rl_ach = rl.get("achieved") or {}
    env = rl.get("envelope") or {}

    # ---- COMPUTE ----
    if rl_ach.get("flops") is not None:
        _add(out, "FMA FLOPS (SASS)",
             sum(f["flops"] for f in rl_ach["fma"]), "FLOP",
             " + ".join(f"{f['counter']} ({_big(f['flops'])})"
                        for f in rl_ach["fma"]),
             [f["counter"] for f in rl_ach["fma"]],
             note="ffma (FP32), hfma (FP16) and dfma (FP64) are the only "
                  "SASS FMA opcode families NVIDIA's database provides FLOP "
                  "counters for — FP8/FP4/TF32 arithmetic runs on the tensor "
                  "cores (see Tensor FLOPS).",
             desc="FLOPs executed by the CUDA-core FMA pipe, summed from "
                  "NVIDIA's per-opcode SASS counters.",
             group=COMPUTE)
    tpaths = rl_ach.get("tensor", [])
    if tpaths:
        _add(out, "Tensor FLOPS (NVIDIA ops-path)",
             sum(t["flops"] for t in tpaths), "FLOP",
             " + ".join(f"{t['counter']} ({_big(t['flops'])})"
                        for t in tpaths),
             [t["counter"] for t in tpaths],
             note="NVIDIA's own sm__ops_path_tensor_* FLOP-path counters — "
                  "the MMA FLOPs ncu itself counts, no shape assumed.",
             desc="FLOPs executed on the tensor cores — counted by NVIDIA's "
                  "own sm__ops_path_tensor_* counters.",
             group=COMPUTE)
    else:
        _add(out, "Tensor FLOPS (NVIDIA ops-path)", 0.0, "FLOP",
             "no NVIDIA sm__ops_path_tensor_* counters in this profile",
             ["sm__ops_path_tensor_src_* .sum"],
             note="No tensor-core work in this profile — every NVIDIA "
                  "sm__ops_path_tensor_* FLOP-path counter is zero or absent.",
             desc="FLOPs executed on the tensor cores — counted by NVIDIA's "
                  "own sm__ops_path_tensor_* counters.",
             group=COMPUTE)
    if rl_ach.get("flops") is not None:
        _add(out, "Total FLOPS (FMA + tensor)",
             rl_ach["flops"], "FLOP",
             "FMA FLOPS (SASS) + Tensor FLOPS (NVIDIA ops-path)",
             ["FMA FLOPS (SASS)", "Tensor FLOPS (NVIDIA ops-path)"],
             desc="The kernel's full floating-point work: FMA pipe + tensor "
                  "cores.",
             group=COMPUTE)
    if rl_ach.get("flops_s") is not None:
        _add(out, "FLOP/s (achieved)",
             rl_ach["flops_s"], "FLOP/s",
             f"Total FLOPS ({_big(rl_ach['flops'])}) ÷ Duration",
             ["gpu__time_duration.avg"],
             desc="Actual arithmetic throughput — total FLOPS divided by "
                  "kernel duration.",
             group=COMPUTE)
    for tname, candidates in TENSOR_OPS:
        for counter in candidates:
            cnt = _m(kp, counter)
            if cnt in (None, 0.0):
                continue
            _add(out, f"Tensor instructions ({tname})",
                 cnt, "inst",
                 counter, [counter],
                 note="SASS instruction count — the FLOPs of these "
                      "instructions are NVIDIA's sm__ops_path_tensor_* "
                      "counters above.",
                 desc="SASS tensor-core instruction count; the FLOPs live "
                      "in the ops-path tile.",
                 group=COMPUTE)
            break
    if env.get("peak_fma_flops") is not None:
        _add(out, "FMA-pipe peak (derived)",
             env["peak_fma_flops"], "FLOP/s",
             env.get("fma_peak_formula", ""),
             [p["formula"] for p in env.get("fma_peaks", [])],
             note="Computed from NVIDIA's own peak_sustained rates × 2 "
                  "FLOP/inst × SM frequency.",
             desc="CUDA-core FMA ceiling — NVIDIA peak_sustained rates × 2 "
                  "FLOP/inst × SM clock.",
             group=COMPUTE)
    if env.get("peak_tensor_flops") is not None:
        _add(out, "Tensor-pipe peak (NVIDIA)",
             env["peak_tensor_flops"], "FLOP/s",
             env.get("tensor_peak_formula", ""),
             [t["counter"] for t in env.get("tensor_peaks", [])],
             note="Backed out of NVIDIA's own ops-path FLOP counter and its "
                  "%-of-peak row.",
             desc="Tensor-core ceiling, backed out of NVIDIA's ops-path "
                  "FLOPs ÷ %-of-peak.",
             group=COMPUTE)
    if env.get("peak_compute_flops") is not None:
        _add(out, "Compute peak (NVIDIA)",
             env["peak_compute_flops"], "FLOP/s",
             "Tensor-pipe peak (NVIDIA) — kernel uses tensor cores"
             if env.get("peak_compute_source") == "tensor"
             else "FMA-pipe peak (derived) — no tensor FLOPs",
             [env["peak_compute_source"] == "tensor"
              and "Tensor-pipe peak (NVIDIA)" or "FMA-pipe peak (derived)"],
             note="The roof drawn in the roofline chart.",
             desc="The compute roof on the roofline chart — tensor pipe if "
                  "the kernel uses it, else the FMA pipe.",
             group=COMPUTE)

    # ---- MEMORY ----
    if bts:
        _add(out, "DRAM bytes moved",
             bts["total"] / 1e9, "GB",
             "dram__bytes_read.sum + dram__bytes_write.sum",
             ["dram__bytes_read.sum", "dram__bytes_write.sum"],
             note=f"read {bts['read'] / 1e9:g} GB + write "
                  f"{bts['write'] / 1e9:g} GB",
             desc="Total bytes read from and written to DRAM.",
             group=MEMORY)
    if env.get("dram_bw") is not None:
        _add(out, "DRAM bandwidth achieved",
             env["dram_bw"] / 1e9, "GB/s",
             "NVIDIA dram__bytes.sum.per_second", ["dram__bytes.sum.per_second"],
             desc="Whole-kernel DRAM transfer rate (NVIDIA's own counter).",
             group=MEMORY)
    if env.get("peak_dram_bw") is not None:
        _add(out, "DRAM peak bandwidth (derived)",
             env["peak_dram_bw"] / 1e9, "GB/s",
             env.get("formula", ""),
             ["dram__bytes.sum.per_second", "SpeedOfLight 'DRAM Throughput'"],
             note="NVIDIA's own %-of-peak row used to back out the peak "
                  "rate; nothing hardcoded.",
             desc="The device DRAM ceiling, derived from NVIDIA's %-of-peak "
                  "row.",
             group=MEMORY)

    # ---- ROOFLINE ----
    if rl_ach.get("ai") is not None:
        _add(out, "Arithmetic intensity",
             rl_ach["ai"], "FLOP/B",
             "Total FLOPS ÷ DRAM bytes moved",
             ["Total FLOPS (FMA + tensor)", "DRAM bytes moved"],
             desc="FLOPs per byte moved from DRAM — the roofline "
                  "x-coordinate.",
             group=ROOFLINE)
    if env.get("ridge") is not None:
        _add(out, "Roofline ridge point",
             env["ridge"], "FLOP/B",
             "compute peak FLOP/s ÷ peak DRAM bandwidth",
             ["Compute peak (NVIDIA)", "DRAM peak bandwidth (derived)"],
             desc="Compute roof ÷ DRAM peak — the AI where the kernel turns "
                  "memory-bound into compute-bound.",
             group=ROOFLINE)
    if rl_ach.get("roof_flops_s_at_ai") is not None:
        _add(out, "Memory-roof FLOP/s at this AI",
             rl_ach["roof_flops_s_at_ai"], "FLOP/s",
             "peak DRAM bandwidth × arithmetic intensity",
             ["DRAM peak bandwidth (derived)", "Arithmetic intensity"],
             note="The achieved point's height on the memory-roof slope — "
                  "what the kernel would need to hit the memory roof at "
                  "this arithmetic intensity.",
             desc="What the kernel would sustain if it sat exactly on the "
                  "memory roof at this AI.",
             group=ROOFLINE)

    # ---- OCCUPANCY & SCHEDULING ----
    _add(out, "CTAs launched", grid, "CTA",
         "NVIDIA Launch Statistics 'Grid Size' row (grid X × Y × Z)",
         ["LaunchStats"],
         desc="Thread blocks launched for this kernel (grid X × Y × Z).",
         group=OCC)
    _add(out, "Total threads", threads, "thread",
         "Grid Size × Block Size", ["LaunchStats"],
         desc="Grid size × block size — all threads this kernel launched.",
         group=OCC)
    _add(out, "Occupancy utilization",
         ach / theo * 100 if (ach is not None and theo) else None,
         "%", "Achieved Occupancy ÷ Theoretical Occupancy", ["Occupancy"],
         note="May exceed 100% because achieved occupancy is a measured "
              "average that can round above the theoretical bound.",
         desc="Achieved occupancy as a share of the theoretical maximum.",
         group=OCC)
    _add(out, "Warp-slot stall share",
         stall_cyc * issued
         if (stall_cyc is not None and issued is not None) else None,
         "%", "Warp Cycles Per Issued Instruction × Issued Warp Per Scheduler",
         ["WarpStateStats", "SchedulerStats"],
         note="Share of scheduler warp slots lost to stalls. 'Issued Warp "
              "Per Scheduler' is a low-precision row (2 decimals), so this "
              "is an estimate.",
         desc="Scheduler warp slots lost to stalls — an estimate from "
              "ncu's rows.",
         group=OCC)
    _add(out, "Instructions per thread",
         inst / threads if (inst is not None and threads) else None,
         "inst/thread", "Executed Instructions ÷ (Grid Size × Block Size)",
         ["InstructionStats", "LaunchStats"],
         desc="SASS instructions executed per thread, whole kernel.",
         group=OCC)
    _add(out, "Kernel-wide IPC",
         inst / elapsed if (inst is not None and elapsed) else None,
         "inst/cycle", "Executed Instructions ÷ Elapsed Cycles",
         ["InstructionStats", "SpeedOfLight"],
         note="ncu's own IPC rows are per-scheduler averages; this is the "
              "whole-kernel instruction throughput.",
         desc="Whole-kernel instruction throughput (instructions per SM "
              "cycle).",
         group=OCC)

    # ---- TIMING ----
    _add(out, "Duration per CTA",
         dur_s * 1e6 / grid if (dur_s is not None and grid) else None,
         "µs", "Duration ÷ Grid Size", ["SpeedOfLight", "LaunchStats"],
         desc="Kernel duration ÷ grid size — the average time per thread "
              "block.",
         group=TIMING)
    _add(out, "Implied SM clock",
         elapsed / dur_s / 1e9 if (elapsed is not None and dur_s) else None,
         "GHz", "Elapsed Cycles ÷ Duration", ["SpeedOfLight"],
         note="Cross-check against NVIDIA's 'SM Frequency' row in the same "
              "section.",
         desc="Elapsed cycles ÷ duration — the clock the cycle counts "
              "imply.",
         group=TIMING)

    # ---- PM SAMPLING ----
    r = _row(kp, "PM Sampling", "Maximum Buffer Size")
    if r is not None and _num(r.value) is not None:
        _add(out, "PM sampler buffer",
             _num(r.value), str(r.unit or ""),
             "NVIDIA 'Maximum Buffer Size' row", ["PM Sampling"],
             desc="PM counter buffer size the sampler was configured with.",
             group=PMSAMP)
    r = _row(kp, "PM Sampling", "Maximum Sampling Interval")
    if r is not None and _num(r.value) is not None:
        _add(out, "PM sampler interval",
             _num(r.value), str(r.unit or ""),
             "NVIDIA 'Maximum Sampling Interval' row", ["PM Sampling"],
             desc="Maximum PM sampling interval configured.",
             group=PMSAMP)
    interval_cyc = _m(kp, "smsp__pcsamp_interval_cycles")
    freq = _sm_freq(kp)
    if interval_cyc and freq:
        _add(out, "Warp-sampling period",
             interval_cyc / freq / 1e3, "µs",
             "smsp__pcsamp_interval_cycles ÷ SM frequency",
             ["smsp__pcsamp_interval_cycles", "SpeedOfLight 'SM Frequency'"],
             desc="Wall-clock time between warp-sampling ticks.",
             group=PMSAMP)
    for name, counter in (("Warp samples collected", "smsp__pcsamp_sample_count"),
                          ("Warp sampling aggregated passes", "smsp__pcsamp_aggregated_passes"),
                          ("Warp sampling dropped bytes", "smsp__pcsamp_dropped_bytes")):
        v = _m(kp, counter)
        if v is not None:
            _add(out, name, v,
                 "samples" if "samples" in name else ("pass" if "passes" in name else "B"),
                 f"NVIDIA {counter}", [counter],
                 desc={"Warp samples collected":
                           "Warp samples the PC-sampler captured.",
                       "Warp sampling aggregated passes":
                           "Sampler passes aggregated into the profile.",
                       "Warp sampling dropped bytes":
                           "Sampler bytes dropped (buffer overflow)."}[name],
                 group=PMSAMP)
    return out


def _op_breakdown(kp: KernelProfile, base: str) -> dict:
    """(space, op) → {sum, hit, miss} from NVIDIA counters.

    Discovers the exported counter families ``base + <space>_op_<op>.sum``
    plus their ``_lookup_hit``/``_lookup_miss`` sector variants. Everything
    present in the profile's metric set is included; nothing is assumed.
    """
    out: dict = {}
    suf = ".sum"
    for n, v in kp.metrics.items():
        if not (n.startswith(base) and n.endswith(suf)):
            continue
        rest = n[len(base):-len(suf)]
        if "_op_" not in rest or "_lookup_" in rest:
            continue
        space, _, op = rest.partition("_op_")
        try:
            out.setdefault((space, op), {})["sum"] = float(v)
        except (TypeError, ValueError):
            continue
    for tag, field in (("_lookup_hit", "hit"), ("_lookup_miss", "miss")):
        msuf = tag + suf
        for n, v in kp.metrics.items():
            if not (n.startswith(base) and n.endswith(msuf)):
                continue
            rest = n[len(base):-len(msuf)]
            if "_op_" not in rest:
                continue
            space, _, _op = rest.partition("_op_")
            d = out.get((space, _op))
            if d:
                with suppress(TypeError, ValueError):
                    d[field] = float(v)
    return out


_OP_ORDER = {"ld": 0, "st": 1, "atom": 2, "red": 3, "redas": 3}
_OP_LABEL = {"ld": "Loads", "st": "Stores", "atom": "Atomics",
             "red": "Reductions", "redas": "Reductions"}
_SPACE_LABEL = {"global": "Global", "local": "Local", "shared": "Shared",
                "dshared": "Load Global Store Shared", "surface": "Surface",
                "texture": "Texture", "const": "Constant"}
_INST_SPACES = {"global": "global", "local": "local", "shared": "shared"}


def _cell(v, unit, formula, sources):
    if v is None:
        return None
    return {"value": v, "unit": unit, "formula": formula,
            "sources": sources, "src": OURS_SRC}


def _pct(kp: KernelProfile, counter: str) -> float | None:
    return _m(kp, counter + ".pct_of_peak_sustained_elapsed")


def _inst_sum(kp: KernelProfile, space: str) -> float | None:
    """Executed SASS load+store instructions for a memory space."""
    hits = []
    for op in ("loads", "stores"):
        v = _m(kp, f"sass__inst_executed_{space}_{op}")
        if v is not None:
            hits.append(v)
    return sum(hits) if hits else None


def _sass_inst(kp: KernelProfile, space: str, op: str) -> float | None:
    name = {"ld": "loads", "st": "stores"}.get(op)
    if name is None:
        return None
    return _m(kp, f"sass__inst_executed_{space}_{name}")


def memory_model(kp: KernelProfile) -> dict:
    """NVIDIA Memory Chart + Memory Tables semantics, computed from NVIDIA's
    own counters (l1tex__*, lts__*, syslts__*, dram__*, sass__inst_executed_*).

    Mirrors the Profiling Guide's Memory Workload Analysis chart (logical
    green units, physical blue units, links colored by % of NVIDIA peak
    sustained, grey when the counter or pct is absent) and the five memory
    tables (Shared, L1/TEX, L2, L2 eviction policies, Device Memory). Every
    cell carries its formula + source counters and is tagged ours.
    """
    out: dict = {"units": [], "links": [], "tables": {}}

    reqs = _op_breakdown(kp, "l1tex__t_requests_pipe_lsu_mem_")
    wfs = _op_breakdown(kp, "l1tex__data_pipe_lsu_wavefronts_mem_")
    sectors = _op_breakdown(kp, "l1tex__t_sectors_pipe_lsu_mem_")
    tex_sec = _op_breakdown(kp, "l1tex__t_sectors_pipe_tex_mem_")

    def req_sum(space, op):
        d = reqs.get((space, op))
        if d is not None:
            return d.get("sum")
        d = wfs.get((space, op))
        return d.get("sum") if d is not None else None

    def req_pct(space, op):
        if (space, op) in reqs:
            p = _pct(kp, f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum")
            if p is not None:
                return p
        if (space, op) in wfs:
            return _pct(kp, f"l1tex__data_pipe_lsu_wavefronts_mem_{space}_op_{op}.sum")
        return None

    # ---- units: which are present in this profile ----
    present = set()
    for space, _op in list(reqs) + list(wfs) + list(sectors) + list(tex_sec):
        present.add(_SPACE_LABEL.get(space, space))
    active: dict = {}

    def mark_active(link_name, value):
        active[link_name] = (active.get(link_name, False) or bool(value))

    # ---- chart links ----
    links = []
    for space in ("global", "local", "shared"):
        v = _inst_sum(kp, space)
        label = _SPACE_LABEL[space]
        if v is not None:
            links.append({"from": "Kernel", "to": label, "kind": "inst",
                          "value": v, "unit": "inst", "pct": None,
                          "formula": "Σ sass__inst_executed_"
                                     f"{space}_loads/stores",
                          "sources": [f"sass__inst_executed_{space}_loads",
                                      f"sass__inst_executed_{space}_stores"]})
            mark_active(label, v)
    dshared = reqs.get(("dshared", "ld")) or wfs.get(("dshared", "ld")) \
        or ({"sum": 0.0} if ("dshared", "ld") in sectors else None)
    ldgsts = _m(kp, "smsp__sass_inst_executed_op_ldgsts.sum")
    if ldgsts is not None or dshared is not None:
        v = ldgsts if ldgsts is not None else dshared.get("sum")
        links.append({"from": "Kernel", "to": "Load Global Store Shared",
                      "kind": "inst" if ldgsts is not None else "req",
                      "value": v, "unit": "inst" if ldgsts is not None else "req",
                      "pct": None, "formula": "smsp__sass_inst_executed_op_ldgsts.sum"
                      if ldgsts is not None
                      else "Σ l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum",
                      "sources": ["smsp__sass_inst_executed_op_ldgsts.sum"]
                      if ldgsts is not None
                      else ["l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum"]})
        mark_active("Load Global Store Shared", v)
    for (space, op), d in sorted(reqs.items()):
        v = d.get("sum")
        links.append({"from": _SPACE_LABEL.get(space, space), "to": "L1/TEX Cache",
                      "kind": "req", "value": v, "unit": "req",
                      "pct": req_pct(space, op),
                      "formula": f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum",
                      "sources": [f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum"]})
        mark_active(_SPACE_LABEL.get(space, space), v)
    for (space, op), d in sorted(wfs.items()):
        if (space, op) in reqs:
            continue
        v = d.get("sum")
        links.append({"from": _SPACE_LABEL.get(space, space), "to": "L1/TEX Cache",
                      "kind": "req", "value": v, "unit": "req",
                      "pct": req_pct(space, op),
                      "formula": f"l1tex__data_pipe_lsu_wavefronts_mem_{space}_op_{op}.sum",
                      "sources": [f"l1tex__data_pipe_lsu_wavefronts_mem_{space}_op_{op}.sum"]})
        mark_active(_SPACE_LABEL.get(space, space), v)
    for (space, op), d in sorted(tex_sec.items()):
        v = d.get("sum")
        links.append({"from": _SPACE_LABEL.get(space, space), "to": "L1/TEX Cache",
                      "kind": "req", "value": v, "unit": "req", "pct": None,
                      "formula": f"l1tex__t_sectors_pipe_tex_mem_{space}_op_{op}.sum",
                      "sources": [f"l1tex__t_sectors_pipe_tex_mem_{space}_op_{op}.sum"]})
        mark_active(_SPACE_LABEL.get(space, space), v)

    l1_miss = sum(d.get("miss", 0) or 0 for d in sectors.values())
    l1_sec_sum = sum(d.get("sum") or 0 for d in sectors.values())
    if l1_sec_sum > 0 or l1_miss > 0 or sectors:
        l1_pct = None
        for c in ("tex", "gcc"):
            p = _pct(kp, f"lts__t_sectors_srcunit_{c}.sum")
            if p is not None:
                l1_pct = max(l1_pct or 0, p)
        links.append({"from": "L1/TEX Cache", "to": "L2 Cache", "kind": "sectors",
                      "value": l1_miss, "unit": "sectors", "pct": l1_pct,
                      "formula": "Σ l1tex__t_sectors_pipe_lsu_mem_*_op_*"
                                 "_lookup_miss.sum",
                      "sources": [f"{n}.sum" for n in
                                  [f"l1tex__t_sectors_pipe_lsu_mem_{s}_op_{o}"
                                   for (s, o) in sectors]]})
        mark_active("L1/TEX Cache", l1_miss)
        mark_active("L2 Cache", l1_miss)
    for tag, unit_name in (("read", "Device Memory"), ("write", "Device Memory")):
        v = _m(kp, f"dram__bytes_{tag}.sum")
        if v is not None:
            links.append({"from": "L2 Cache", "to": unit_name, "kind": "bytes",
                          "value": v, "unit": "B", "pct": _pct(kp, f"dram__bytes_{tag}.sum"),
                          "formula": f"dram__bytes_{tag}.sum",
                          "sources": [f"dram__bytes_{tag}.sum"]})
            mark_active(unit_name, v)
    for tag, unit_name in (("sysmem", "System Memory"), ("peer", "Peer Memory")):
        v = _m(kp, f"syslts__t_sectors_aperture_{tag}_lookup_miss.sum")
        if v is not None:
            links.append({"from": "L2 Cache", "to": unit_name, "kind": "sectors",
                          "value": v, "unit": "sectors", "pct": None,
                          "formula": f"syslts__t_sectors_aperture_{tag}_lookup_miss.sum",
                          "sources": [f"syslts__t_sectors_aperture_{tag}_lookup_miss.sum"]})
            mark_active(unit_name, v)

    active["Kernel"] = True
    out["units"].append({"name": "Kernel", "kind": "logical", "active": True})
    for name in ("Global", "Local", "Texture", "Surface", "Shared",
                 "Load Global Store Shared"):
        if name not in present and not active.get(name):
            continue
        if name in ("Texture", "Surface") and name not in present:
            continue
        out["units"].append({"name": name, "kind": "logical",
                             "active": bool(active.get(name))})
    for name in ("L1/TEX Cache", "L2 Cache", "Device Memory", "System Memory",
                 "Peer Memory"):
        if name not in active:
            continue
        out["units"].append({"name": name, "kind": "physical",
                             "active": bool(active.get(name))})
    agg: dict = {}
    for ln in links:
        key = (ln["from"], ln["to"], ln["kind"])
        if key not in agg:
            agg[key] = dict(ln)
            agg[key]["value"] = ln["value"] or 0.0
            agg[key]["sources"] = list(ln["sources"])
        else:
            agg[key]["value"] += ln["value"] or 0.0
            if ln["pct"] is not None:
                agg[key]["pct"] = max(agg[key]["pct"] or 0, ln["pct"])
            agg[key]["sources"] += ln["sources"]
    for ln in agg.values():
        ln["src"] = OURS_SRC
    out["links"] = list(agg.values())

    # ---- tables ----
    tables: dict = {}

    shared_ops = sorted(
        set(op for (space, op) in list(wfs) + list(reqs) if space == "shared"),
        key=lambda o: (_OP_ORDER.get(o, 9), o))
    if shared_ops:
        rows = []
        headers = ["Instructions", "Requests", "Wavefronts", "% Peak",
                   "Bank Conflicts"]
        for op in shared_ops:
            row = {}
            row["Instructions"] = _cell(
                _sass_inst(kp, "shared", op), "inst",
                f"sass__inst_executed_shared_{'loads' if op == 'ld' else 'stores'}"
                if op in ("ld", "st") else "no SASS counter for this op",
                [f"sass__inst_executed_shared_{'loads' if op == 'ld' else 'stores'}"]
                if op in ("ld", "st") else [])
            v = req_sum("shared", op)
            row["Requests"] = _cell(
                v, "req",
                f"l1tex__t_requests_pipe_lsu_mem_shared_op_{op}.sum"
                if (("shared", op) in reqs) else
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_"
                f"{op}.sum (one request per shared instruction)",
                [f"l1tex__t_requests_pipe_lsu_mem_shared_op_{op}.sum"]
                if (("shared", op) in reqs) else
                [f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum"])
            d = wfs.get(("shared", op))
            row["Wavefronts"] = _cell(
                d.get("sum") if d else None, "wf",
                f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum",
                [f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum"])
            row["% Peak"] = _cell(
                req_pct("shared", op), "%",
                "NVIDIA pct_of_peak_sustained_elapsed of the shared "
                f"{_OP_LABEL.get(op, op).lower()} counter",
                [f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum.pct_of_peak_sustained_elapsed"])
            bc = _m(kp, f"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum")
            row["Bank Conflicts"] = _cell(
                bc, "conflicts",
                f"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum",
                [f"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum"])
            rows.append({"label": _OP_LABEL.get(op, op), "cells": row})
        totals = {}
        for h in headers:
            vals = [r["cells"][h].get("value") for r in rows
                    if r["cells"][h] is not None and r["cells"][h].get("value") is not None]
            totals[h] = _cell(sum(vals), "inst" if h == "Instructions" else
                              ("req" if h == "Requests" else
                               ("wf" if h == "Wavefronts" else
                                ("%" if h == "% Peak" else "conflicts"))),
                              "Σ " + " + ".join(c["sources"][0] for r in rows
                              for c in [r["cells"][h]] if c) if vals else "",
                              [s for r in rows for c in [r["cells"][h]] if c for s in c["sources"]])
        rows.append({"label": "Total", "cells": totals})
        tables["shared"] = {"title": "Shared Memory", "cols": headers, "rows": rows}

    l1_ops = sorted(sectors,
                    key=lambda so: (_SPACE_LABEL.get(so[0], so[0]),
                                    _OP_ORDER.get(so[1], 9), so[1]))
    if l1_ops:
        headers = ["Instructions", "Requests", "Wavefronts", "Sectors",
                   "Sectors/Req", "Hit Rate", "Bytes", "Sector Misses to L2"]
        rows = []
        for (space, op) in l1_ops:
            d = sectors[(space, op)]
            r = {}
            inst_name = f"sass__inst_executed_{space}_" \
                + {"ld": "loads", "st": "stores"}[op] \
                if space in _INST_SPACES and op in ("ld", "st") else None
            r["Instructions"] = _cell(
                _sass_inst(kp, space, op), "inst",
                inst_name if inst_name else "no SASS counter for this access type",
                [inst_name] if inst_name else [])
            r["Requests"] = _cell(
                req_sum(space, op), "req",
                f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum"
                if (space, op) in reqs else "not exported for this access type",
                [f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum"]
                if (space, op) in reqs else [])
            wd = wfs.get((space, op))
            r["Wavefronts"] = _cell(
                wd.get("sum") if wd else None, "wf",
                f"l1tex__data_pipe_lsu_wavefronts_mem_{space}_op_{op}.sum",
                [f"l1tex__data_pipe_lsu_wavefronts_mem_{space}_op_{op}.sum"])
            r["Sectors"] = _cell(d.get("sum"), "sectors",
                                 f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum",
                                 [f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum"])
            v = req_sum(space, op)
            r["Sectors/Req"] = _cell(
                d.get("sum") / v if (v and d.get("sum") is not None) else None,
                "sectors/req",
                f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum ÷ "
                f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum",
                [f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum",
                 f"l1tex__t_requests_pipe_lsu_mem_{space}_op_{op}.sum"])
            hit, miss = d.get("hit"), d.get("miss")
            r["Hit Rate"] = _cell(
                hit / (hit + miss) * 100 if (hit is not None and miss is not None
                                             and hit + miss > 0) else None,
                "%",
                f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}_lookup_hit.sum ÷ "
                f"(…_lookup_hit.sum + …_lookup_miss.sum)",
                [f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}_lookup_hit.sum",
                 f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}_lookup_miss.sum"])
            r["Bytes"] = _cell(
                d.get("sum") * 32 if d.get("sum") is not None else None, "B",
                f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum × 32 B/sector",
                [f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}.sum"])
            r["Sector Misses to L2"] = _cell(
                miss, "sectors",
                f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}_lookup_miss.sum",
                [f"l1tex__t_sectors_pipe_lsu_mem_{space}_op_{op}_lookup_miss.sum"])
            rows.append({"label": f"{_SPACE_LABEL.get(space, space)} {_OP_LABEL.get(op, op)}",
                         "cells": r})
        t = {}
        for h in headers:
            vals = [r["cells"][h].get("value") for r in rows
                    if r["cells"][h] and r["cells"][h].get("value") is not None]
            t[h] = _cell(
                sum(vals), "sectors" if h == "Sectors" else
                ("B" if h == "Bytes" else "sectors"),
                "Σ " + " + ".join(r["cells"][h]["sources"][0] for r in rows
                                  if r["cells"][h] and r["cells"][h]["sources"]) if vals else "",
                [s for r in rows for c in [r["cells"][h]] if c for s in c["sources"]])
        hr = _m(kp, "l1tex__t_sector_hit_rate.pct")
        t["Hit Rate"] = _cell(hr, "%",
                              "NVIDIA l1tex__t_sector_hit_rate.pct (whole cache)",
                              ["l1tex__t_sector_hit_rate.pct"])
        rows.append({"label": "Total", "cells": t})
        tables["l1"] = {"title": "L1/TEX Cache", "cols": headers, "rows": rows}

    l2_clients = {}
    l2_tex_ops = {}
    for n, v in kp.metrics.items():
        if n.startswith("lts__t_sectors_srcunit_") and n.endswith(".sum"):
            client = n[len("lts__t_sectors_srcunit_"):-len(".sum")]
            if "_lookup_" in client or "." in client:
                continue
            if client.startswith("tex_op_"):
                l2_tex_ops[client[len("tex_op_"):]] = v
            else:
                l2_clients[client] = v
    if l2_clients or l2_tex_ops:
        headers = ["Requests", "Sectors", "Sectors/Req", "% Peak", "Hit Rate",
                   "Bytes", "Throughput", "Sector Misses to System",
                   "Sector Misses to Peer"]
        _CLIENT_LABEL = {"tex": "L1/TEX Total", "gcc": "GCC",
                         "ltcfabric": "L2 Fabric Total", "l1": "L1/TEX Total"}
        _OP_LABEL2 = {"read": "Reads", "write": "Writes", "atom": "Atomics",
                      "red": "Reductions", "cas": "Compare-And-Swap",
                      "atom_dot_alu": "Atomics (ALU)",
                      "atom_dot_cas": "Atomics (CAS)",
                      "red_dot_alu": "Reductions (ALU)",
                      "red_dot_cas": "Reductions (CAS)",
                      "red_dot_minmax": "Reductions (min/max)"}

        def _l2_row(label, sectors_v, client=None, op=None):
            r = {}
            if client is not None:
                base = f"lts__t_sectors_srcunit_{client}"
                rv = _m(kp, f"lts__t_requests_srcunit_{client}.sum")
                r["Requests"] = _cell(rv, "req",
                                      f"lts__t_requests_srcunit_{client}.sum",
                                      [f"lts__t_requests_srcunit_{client}.sum"])
                r["% Peak"] = _cell(_pct(kp, base + ".sum"), "%",
                                    "NVIDIA pct_of_peak_sustained_elapsed of "
                                    f"{base}.sum",
                                    [base + ".sum.pct_of_peak_sustained_elapsed"])
                hit = _m(kp, f"lts__t_sectors_srcunit_{client}_lookup_hit.sum")
                miss = _m(kp, f"lts__t_sectors_srcunit_{client}_lookup_miss.sum")
                thr = _m(kp, base + ".sum.per_second")
                src = f"lts__t_sectors_srcunit_{client}.sum"
            elif op is not None:
                base = f"lts__t_sectors_srcunit_tex_op_{op}"
                rv = _m(kp, f"lts__t_requests_srcunit_tex_op_{op}.sum")
                r["Requests"] = _cell(
                    rv, "req", f"lts__t_requests_srcunit_tex_op_{op}.sum",
                    [f"lts__t_requests_srcunit_tex_op_{op}.sum"])
                r["% Peak"] = _cell(_pct(kp, base + ".sum"), "%",
                                    "NVIDIA pct_of_peak_sustained_elapsed of "
                                    f"{base}.sum",
                                    [base + ".sum.pct_of_peak_sustained_elapsed"])
                hit = _m(kp, f"lts__t_sectors_srcunit_tex_op_{op}_lookup_hit.sum")
                miss = _m(kp, f"lts__t_sectors_srcunit_tex_op_{op}_lookup_miss.sum")
                thr = _m(kp, base + ".sum.per_second")
                src = f"lts__t_sectors_srcunit_tex_op_{op}.sum"
            else:
                rv, hit, miss, thr, src = None, None, None, None, ""
            r["Sectors"] = _cell(sectors_v, "sectors", src, [src])
            r["Sectors/Req"] = _cell(
                sectors_v / rv if (rv and sectors_v is not None) else None,
                "sectors/req",
                src + " ÷ " + src.replace("lts__t_sectors", "lts__t_requests"),
                [src, src.replace("lts__t_sectors", "lts__t_requests")])
            r["Hit Rate"] = _cell(
                hit / (hit + miss) * 100 if (hit is not None and miss is not None
                                             and hit + miss > 0) else None,
                "%", src + " (lookup_hit) ÷ (lookup_hit + lookup_miss)",
                [src.replace(".sum", "_lookup_hit.sum"),
                 src.replace(".sum", "_lookup_miss.sum")])
            r["Bytes"] = _cell(sectors_v * 32 if sectors_v is not None else None,
                               "B", src + " × 32 B/sector", [src])
            r["Throughput"] = _cell(thr, "sectors/s", src + ".per_second",
                                    [src + ".per_second"])
            for ap, tag in (("sysmem", "System"), ("peer", "Peer")):
                apv = _m(kp, f"syslts__t_sectors_srcunit_{client}_aperture_{ap}"
                             f"_lookup_miss.sum" if client is not None else
                             f"syslts__t_sectors_srcunit_tex_op_{op}_aperture_{ap}"
                             f"_lookup_miss.sum")
                r[f"Sector Misses to {tag}"] = _cell(
                    apv, "sectors",
                    f"syslts__t_sectors_srcunit_{client}_aperture_{ap}"
                    f"_lookup_miss.sum" if client is not None else
                    f"syslts__t_sectors_srcunit_tex_op_{op}_aperture_{ap}"
                    f"_lookup_miss.sum",
                    [f"syslts__t_sectors_srcunit_{client}_aperture_{ap}"
                     f"_lookup_miss.sum" if client is not None else
                     f"syslts__t_sectors_srcunit_tex_op_{op}_aperture_{ap}"
                     f"_lookup_miss.sum"])
            for h in headers:
                r.setdefault(h, None)
            return {"label": label, "cells": r}

        rows = []
        for client, v in sorted(l2_clients.items()):
            rows.append(_l2_row(_CLIENT_LABEL.get(client, client.upper()), v,
                                client=client))
        ecc = _m(kp, "lts__t_sectors_data_ecc.sum")
        if ecc:
            rows.append(_l2_row("ECC Total", ecc))
        if rows:
            t = {}
            for h in headers:
                vals = [r["cells"][h].get("value") for r in rows
                        if r["cells"][h] and r["cells"][h].get("value") is not None]
                t[h] = _cell(
                    sum(vals), "req" if h == "Requests" else
                    ("sectors" if h in ("Sectors", "Sector Misses to System",
                                        "Sector Misses to Peer") else
                     ("B" if h == "Bytes" else "sectors/s")),
                    "Σ " + " + ".join(r["cells"][h]["sources"][0] for r in rows
                                      if r["cells"][h] and r["cells"][h]["sources"])
                    if vals else "",
                    [s for r in rows for c in [r["cells"][h]] if c
                     for s in c["sources"]])
            hit = _m(kp, "lts__t_sectors_lookup_hit.sum")
            miss = _m(kp, "lts__t_sectors_lookup_miss.sum")
            t["Sectors"] = _cell(
                hit + miss if (hit is not None and miss is not None) else None,
                "sectors",
                "lts__t_sectors_lookup_hit.sum + lts__t_sectors_lookup_miss.sum",
                ["lts__t_sectors_lookup_hit.sum", "lts__t_sectors_lookup_miss.sum"])
            t["Bytes"] = _cell(
                (hit + miss) * 32 if (hit is not None and miss is not None)
                else None, "B",
                "(lts__t_sectors_lookup_hit.sum + lts__t_sectors_lookup_miss.sum)"
                " × 32 B/sector",
                ["lts__t_sectors_lookup_hit.sum", "lts__t_sectors_lookup_miss.sum"])
            t["Hit Rate"] = _cell(
                hit / (hit + miss) * 100 if (hit is not None and miss is not None
                                             and hit + miss > 0) else None,
                "%",
                "lts__t_sectors_lookup_hit.sum ÷ (…_lookup_hit.sum + …_lookup_miss.sum)",
                ["lts__t_sectors_lookup_hit.sum", "lts__t_sectors_lookup_miss.sum"])
            t["% Peak"] = None
            t["Sectors/Req"] = None
            t["Requests"] = None
            t["Throughput"] = None
            rows.append({"label": "GPU Total", "cells": t})
            tables["l2"] = {"title": "L2 Cache", "cols": headers, "rows": rows}

        if l2_tex_ops:
            tro = []
            for op, v in sorted(l2_tex_ops.items(),
                                key=lambda kv: _OP_ORDER.get(kv[0], 9)):
                tro.append(_l2_row("TEX Op " + _OP_LABEL2.get(op, op.upper()),
                                   v, op=op))
            tables["texops"] = {"title": "Texture Operations",
                                "cols": headers, "rows": tro}
    evict = []
    for policy in ("first", "last", "normal", "normal_demote"):
        hit = _m(kp, f"lts__t_sectors_evict_{policy}_lookup_hit.sum")
        miss = _m(kp, f"lts__t_sectors_evict_{policy}_lookup_miss.sum")
        if hit is None and miss is None:
            continue
        r = {
            "label": "First" if policy == "first" else
                     "Last" if policy == "last" else
                     "Normal" if policy == "normal" else "Normal Demote",
            "cells": {
                "Sectors": _cell(
                    (hit or 0) + (miss or 0), "sectors",
                    f"lts__t_sectors_evict_{policy}_lookup_hit.sum + "
                    f"…_lookup_miss.sum",
                    [f"lts__t_sectors_evict_{policy}_lookup_hit.sum",
                     f"lts__t_sectors_evict_{policy}_lookup_miss.sum"]),
                "Hit Rate": _cell(
                    hit / (hit + miss) * 100 if (hit is not None and miss is not None
                                                 and hit + miss > 0) else None,
                    "%",
                    f"lts__t_sectors_evict_{policy}_lookup_hit.sum ÷ "
                    f"(…_lookup_hit.sum + …_lookup_miss.sum)",
                    [f"lts__t_sectors_evict_{policy}_lookup_hit.sum",
                     f"lts__t_sectors_evict_{policy}_lookup_miss.sum"]),
            },
        }
        evict.append(r)
    if evict:
        t = {}
        for h in ("Sectors", "Hit Rate"):
            vals = [r["cells"][h].get("value") for r in evict
                    if r["cells"][h] and r["cells"][h].get("value") is not None]
            t[h] = _cell(sum(vals), "sectors" if h == "Sectors" else "%",
                         "Σ " + " + ".join(r["cells"][h]["sources"][0] for r in evict
                                           if r["cells"][h] and r["cells"][h]["sources"]) if vals else "",
                         [s for r in evict for c in [r["cells"][h]] if c for s in c["sources"]])
        evict.append({"label": "Total", "cells": t})
        tables["evict"] = {"title": "L2 Eviction Policies",
                           "cols": ["Sectors", "Hit Rate"], "rows": evict}

    dram = {}
    for tag in ("read", "write"):
        sec = _m(kp, f"dram__sectors_{tag}.sum")
        byt = _m(kp, f"dram__bytes_{tag}.sum")
        if sec is None and byt is None:
            continue
        dram[tag] = {
            "Sectors": _cell(sec, "sectors", f"dram__sectors_{tag}.sum",
                             [f"dram__sectors_{tag}.sum"]),
            "% Peak": _cell(_pct(kp, f"dram__bytes_{tag}.sum"), "%",
                            "NVIDIA pct_of_peak_sustained_elapsed of "
                            f"dram__bytes_{tag}.sum",
                            [f"dram__bytes_{tag}.sum.pct_of_peak_sustained_elapsed"]),
            "Bytes": _cell(byt, "B", f"dram__bytes_{tag}.sum",
                           [f"dram__bytes_{tag}.sum"]),
            "Throughput": _cell(_m(kp, f"dram__bytes_{tag}.sum.per_second"),
                                "B/s", f"dram__bytes_{tag}.sum.per_second",
                                [f"dram__bytes_{tag}.sum.per_second"]),
        }
    if dram:
        rows = [{"label": "Reads" if tag == "read" else "Writes", "cells": d}
                for tag, d in sorted(dram.items())]
        t = {}
        for h in ("Sectors", "% Peak", "Bytes", "Throughput"):
            vals = [r["cells"][h].get("value") for r in rows
                    if r["cells"][h] and r["cells"][h].get("value") is not None]
            t[h] = _cell(sum(vals), "sectors" if h == "Sectors" else
                         ("%" if h == "% Peak" else ("B" if h == "Bytes" else "B/s")),
                         "Σ " + " + ".join(r["cells"][h]["sources"][0] for r in rows
                                           if r["cells"][h]["sources"]) if vals else "",
                         [s for r in rows for c in [r["cells"][h]] if c for s in c["sources"]])
        rows.append({"label": "Total", "cells": t})
        tables["dram"] = {"title": "Device Memory", "cols": ["Sectors", "% Peak",
                                                             "Bytes", "Throughput"],
                          "rows": rows}

    out["tables"] = tables
    return out
