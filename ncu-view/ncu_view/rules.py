"""Rule engine: per-kernel rules plus the per-kernel bottleneck verdict.

Two rule tiers (see report.py for assembly):

* ncu-rule   -- NVIDIA's own rule results, read verbatim from a .ncu-rep
               report (Phase 2; nothing re-derived).
* our-rule   -- the rules below. Thresholds are documented at each rule and
               mirror NVIDIA's published ones where they exist (the SOL
               bottleneck thresholds 10/60/80 come from NVIDIA's own rule
               script), plus the lessons of the "Beating cuBLAS on B200"
               write-up (per-active-cycle efficiency, the stall-counter
               denominator guard, the "signal is the row that moved" rule).

A rule that cannot evaluate (its counters are absent) returns None and is
simply not reported -- never a fabricated verdict.
"""

from __future__ import annotations

from .ingest import STALL_BASES, STALL_SHORT
from .model import KernelProfile, RuleResult

SOL_BALANCED_THRESHOLD = 10.0   # |compute - memory| <= 10 -> balanced
SOL_LATENCY_THRESHOLD = 60.0    # max utilization < 60 -> latency-bound
SOL_NO_BOUND_THRESHOLD = 80.0   # min utilization >= 80 -> both saturated
OCCUPANCY_LOW = 50.0            # achieved warps % below this -> warning
STALL_TOTAL_HIGH = 60.0         # stall cycles per issue-active above this
STALL_SHARE_HIGH = 50.0         # top stall reason share above this
GRID_LOW = 25.0                 # active-CTA % below this -> grid too small
TENSOR_SHARE_LOW = 20.0         # tensor-instruction share below this
PIPE_BOUND_MAX = 92.0           # pipe below this with nothing else saturated
LATENCY_STALL = 80.0            # verdict: stall total above this
LATENCY_PIPE = 40.0             # verdict: pipe below this
LOADPATH_STALL_MIN, LOADPATH_STALL_MAX = 45.0, 80.0
LOADPATH_LG_SHARE = 5.0         # LSU-throttle share in the stall total
LOADPATH_PIPE = 50.0
FREEZE_TOL = 3.0                # % change in stall total that counts as frozen
CONVERGED_PIPE = 90.0
CONVERGED_TIME_TOL = 1.0        # % time change vs previous kernel
CONVERGING_PIPE = 88.0
DENOM_WARP_DELTA = 2.0          # warps-active jump that breaks stall comparability

STALL_FIX = {
    "long scoreboard": "warps wait on memory results - overlap loads with the MMA (prefetch, vectorize, cp.async)",
    "short scoreboard": "warps wait on a dependent instruction - unroll and reorder to lengthen the dependency chains",
    "LSU throttle": "the load path is the tax - check sector efficiency and layout (static shapes, TMA)",
    "barrier": "warps wait at a barrier - reduce sync frequency or split the work",
    "math pipe throttle": "the math pipe is congested - reduce per-thread math or spread work",
    "MIO throttle": "the MIO (memory input/output) queues are full - reduce pressure on the LSU/AGU path",
    "fixed-latency wait": "fixed-latency dependency - more instruction-level parallelism needed",
    "no instruction": "the scheduler pool is empty - raise occupancy",
    "not selected": "warps eligible but not picked - many eligible warps, check for issue-slot contention",
    "sleeping": "warps sleeping on __nanosleep - implicit; expected when co-operative groups sleep",
}

VERDICT_TEXT = {
    "LATENCY-BOUND": "nothing saturated at once - every resource idles while warps wait",
    "COMPILER-OPACITY": "the compiler could not see the layout facts - more instructions while getting faster",
    "LOAD-PATH": "loads now run, and the LSU path throttles - the tax of the memory layout",
    "ISSUE-SERIALIZATION": "stall total frozen while everything else moved - the serialization is on the issue side",
    "CONVERGED": "counters stopped recommending anything - the kernel is at its plateau",
    "CONVERGING": "still improving, but the stall counter has stopped carrying signal",
    "PIPE-BOUND": "the tensor pipe idles below saturation while nothing else is busy - per-tile overhead",
    "NO-SINGLE-STORY": "no single bottleneck dominates - the remaining gap is the product of many small inefficiencies",
    "REFERENCE": "the reference implementation - compare the series against it, not each other",
}


def _v(kp: KernelProfile, metric: str) -> float | None:
    return kp.metrics.get(metric)


def _stall_total(kp: KernelProfile) -> float | None:
    vals = [_v(kp, b) for b in STALL_BASES]
    vals = [x for x in vals if x is not None]
    return sum(vals) if vals else None


def _stall_share(kp: KernelProfile, base: str) -> float | None:
    v = _v(kp, base)
    total = _stall_total(kp)
    if v is None or not total:
        return None
    return v / total * 100.0


def _tflops(kp: KernelProfile, cfg: dict) -> float | None:
    t = _v(kp, "gpu__time_duration.avg")
    if t is None or not cfg["M"]:
        return None
    return 2.0 * cfg["M"] ** 3 / (t * 1e-9) / 1e12


def _pipe_pct(kp: KernelProfile) -> float | None:
    return _v(kp, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")


def _dram_pct(kp: KernelProfile, cfg: dict) -> float | None:
    bps = _v(kp, "dram__bytes.sum.per_second")
    if bps is None:
        return None
    return bps / cfg["dram_peak"] * 100.0


def _tma_pct(kp: KernelProfile) -> float | None:
    from .sections import tma_pct as _tma

    return _tma(kp)[0]


def _movers(kp: KernelProfile, prev: KernelProfile | None) -> str:
    if prev is None:
        return ""
    rows = [
        ("time", "gpu__time_duration.avg", lambda v: f"{v / 1e3:.0f} µs"),
        ("pipe", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", lambda v: f"{v:+.1f}pt"),
        ("DRAM", "dram__bytes.sum.per_second", lambda v: f"{v / 1e9:+.1f} GB/s"),
        ("TMA", None, lambda v: f"{v:+.1f}pt"),
        ("stall", None, lambda v: f"{v:+.2f} cyc"),
        ("instructions", "smsp__inst_executed.sum", lambda v: f"{v / 1e6:+.2f}M"),
    ]
    moves = []
    for label, metric, fmt in rows:
        if metric is None:
            if label == "TMA":
                a, b = _tma_pct(prev), _tma_pct(kp)
            elif label == "stall":
                a, b = _stall_total(prev), _stall_total(kp)
            else:
                continue
        else:
            a, b = _v(prev, metric), _v(kp, metric)
        if a is None or b is None or a == 0:
            continue
        moves.append((abs(b - a) / abs(a) * 100.0, label, fmt(b - a)))
    if not moves:
        return ""
    pct, label, delta = max(moves)
    return f"largest mover vs previous: {label} {delta} ({pct:.0f}% change)"


# ---------------------------------------------------------------------------
# per-kernel rules
# ---------------------------------------------------------------------------

def rule_sol_bottleneck(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    sm = _pipe_pct(kp)
    mem = _dram_pct(kp, cfg)
    if sm is None or mem is None:
        return None
    mx = max(sm, mem)
    if mx < SOL_LATENCY_THRESHOLD:
        kind, sev, why = "latency-bound", "warning", (
            f"neither compute ({sm:.1f}%) nor memory ({mem:.1f}%) is busy - "
            "the kernel is waiting, not working")
    elif min(sm, mem) >= SOL_NO_BOUND_THRESHOLD:
        kind, sev, why = "saturated", "info", (
            f"both compute ({sm:.1f}%) and memory ({mem:.1f}%) are near peak")
    elif abs(sm - mem) <= SOL_BALANCED_THRESHOLD:
        kind, sev, why = "balanced", "info", (
            f"compute ({sm:.1f}%) and memory ({mem:.1f}%) are similarly busy")
    else:
        kind, sev, why = ("compute-bound" if sm > mem else "memory-bound"), "warning", (
            f"{'compute' if sm > mem else 'memory'} ({max(sm, mem):.1f}%) "
            f"outruns {'memory' if sm > mem else 'compute'} ({min(sm, mem):.1f}%)")
    return RuleResult("sol-bottleneck", "Speed Of Light bottleneck", sev,
                      f"{kind}: {why}.",
                      kernel=kp.key, source="our",
                      focus={"compute_sol_pct": sm, "memory_sol_pct": mem})


def rule_occupancy_low(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    warps = _v(kp, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if warps is None:
        return None
    if warps >= OCCUPANCY_LOW:
        return None
    return RuleResult("occupancy-low", "Low achieved occupancy", "warning",
                      f"achieved occupancy is {warps:.1f}% of the SM's warp "
                      "capacity - too few resident warps to hide memory "
                      "latency. Raise occupancy (smaller tiles, fewer "
                      "registers, more CTAs).",
                      kernel=kp.key, focus={"achieved_occupancy_pct": warps})


def rule_dominant_stall(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    total = _stall_total(kp)
    if total is None or total < STALL_TOTAL_HIGH:
        return None
    ranked = sorted(
        ((b, _stall_share(kp, b)) for b in STALL_BASES),
        key=lambda kv: kv[1] or 0.0, reverse=True)
    top_base, top_share = ranked[0]
    if top_share is None or top_share < STALL_SHARE_HIGH:
        return None
    label = STALL_SHORT[top_base]
    fix = STALL_FIX.get(label, "address the dominant stall reason")
    return RuleResult("dominant-stall", "Dominant warp stall", "warning",
                      f"{total:.1f} cycles of stall per issued instruction, "
                      f"{top_share:.1f}% of it '{label}' - {fix}.",
                      kernel=kp.key, source="our",
                      focus={"stall_total_cycles": total, "top_stall_share_pct": top_share})


def rule_grid_small(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    ctas = _v(kp, "sm__ctas_active.avg.pct_of_peak_sustained_active")
    if ctas is None or ctas >= GRID_LOW:
        return None
    return RuleResult("grid-small", "Grid smaller than the machine", "suggestion",
                      f"active CTAs are {ctas:.1f}% of what the GPU can hold - "
                      "the grid may be too small to fill the device.",
                      kernel=kp.key, focus={"active_ctas_pct": ctas})


def rule_instruction_mix(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    inst = _v(kp, "smsp__inst_executed.sum")
    tens = _v(kp, "sm__inst_executed_pipe_tensor.sum")
    if inst is None or tens is None or inst <= 0:
        return None
    share = tens / inst * 100.0
    pipe = _pipe_pct(kp) or 0.0
    if share >= TENSOR_SHARE_LOW or pipe >= PIPE_BOUND_MAX:
        return None
    return RuleResult("instruction-mix", "Non-tensor instruction mix", "info",
                      f"tensor instructions are {share:.1f}% of the {inst / 1e6:.2f}M "
                      "executed - the mix is dominated by index math and "
                      "address arithmetic. Declare layout facts statically "
                      "so the compiler stops paying per-tile.",
                      kernel=kp.key, focus={"tensor_share_pct": share})


def rule_tma_idle(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    tma = _tma_pct(kp)
    dram = _dram_pct(kp, cfg)
    if tma is None or dram is None:
        return None
    if tma < 30.0 and dram > 30.0:
        return RuleResult("tma-idle", "TMA underused for copy traffic", "suggestion",
                          f"DRAM moves {dram:.1f}% of peak but TMA is active "
                          f"only {tma:.1f}% of cycles - bulk copies via "
                          "cp.async.bulk/TMA would overlap them with compute.",
                          kernel=kp.key, focus={"tma_active_pct": tma, "dram_pct": dram})
    return None


def rule_dram_saturated(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    dram = _dram_pct(kp, cfg)
    if dram is None or dram < 90.0:
        return None
    return RuleResult("dram-saturated", "DRAM bandwidth saturated", "warning",
                      f"DRAM runs at {dram:.1f}% of peak - the kernel is "
                      "bandwidth-bound; reduce bytes moved (data reuse, "
                      "smaller tiles, compression).",
                      kernel=kp.key, focus={"dram_pct": dram})


def rule_cache_behavior(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    l2 = _v(kp, "lts__t_sector_hit_rate.pct")
    dram = _dram_pct(kp, cfg)
    if l2 is None or dram is None:
        return None
    if l2 < 60.0 and dram > 50.0:
        return RuleResult("cache-behavior", "Low L2 hit rate under DRAM load", "suggestion",
                          f"L2 hit rate is {l2:.1f}% while DRAM runs at "
                          f"{dram:.1f}% of peak - improve data reuse and "
                          "locality (tiling, L2 persistence).",
                          kernel=kp.key, focus={"l2_hit_rate_pct": l2, "dram_pct": dram})
    return None


def rule_bank_conflicts(kp: KernelProfile, cfg: dict) -> RuleResult | None:
    conflicts = _v(kp, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    inst = _v(kp, "smsp__inst_executed.sum")
    if conflicts is None:
        return None
    if conflicts <= 0:
        return None
    sev = "warning" if inst and conflicts / inst > 0.05 else "suggestion"
    return RuleResult("bank-conflicts", "Shared-memory bank conflicts", sev,
                      f"{conflicts:.0f} shared-memory bank conflicts detected "
                      "- reorder shared arrays or pad strides to spread "
                      "accesses across banks.",
                      kernel=kp.key, focus={"bank_conflicts": conflicts})


# ---------------------------------------------------------------------------
# bottleneck verdict (the decision table)
# ---------------------------------------------------------------------------

def verdict(kp: KernelProfile, prev: KernelProfile | None, cfg: dict) -> RuleResult:
    """One-line per-kernel bottleneck verdict (first matching rule wins)."""
    stall = _stall_total(kp)
    pipe = _pipe_pct(kp) or 0.0
    dram = _dram_pct(kp, cfg) or 0.0
    tma = _tma_pct(kp) or 0.0
    time_ = _v(kp, "gpu__time_duration.avg")
    insts = _v(kp, "smsp__inst_executed.sum")
    warps = _v(kp, "sm__warps_active.avg.pct_of_peak_sustained_active")
    p_time = _v(prev, "gpu__time_duration.avg") if prev else None
    p_pipe = _pipe_pct(prev) if prev else None
    p_stall = _stall_total(prev) if prev else None
    p_insts = _v(prev, "smsp__inst_executed.sum") if prev else None
    p_warps = _v(prev, "sm__warps_active.avg.pct_of_peak_sustained_active") if prev else None

    if time_ is None:
        return RuleResult("verdict", "NO-DATA", "info",
                          "no duration recorded for this kernel — profile it "
                          "with `ncu --set full` (or a full probe set).",
                          kernel=kp.key, source="our", focus={})

    label, sev, msg, caveat = "NO-SINGLE-STORY", "info", "", ""

    if kp.key == "cublas" or "cublas" in kp.name.lower():
        label, sev = "REFERENCE", "info"
        tfl = _tflops(kp, cfg)
        eff = tfl / (pipe / 100.0) if tfl and pipe else None
        msg = (f"the reference implementation: pipe busy {pipe:.1f}% at "
               f"{tfl:.0f} TFLOPS" + (f" ({eff:.0f} per 100% pipe)" if eff else "") +
               " - compare the series against it, not each other.")
    elif stall is not None and stall > LATENCY_STALL and pipe < LATENCY_PIPE:
        label, sev = "LATENCY-BOUND", "warning"
        msg = f"{stall:.1f} cycles of stall per issued instruction, pipe {pipe:.1f}% - {VERDICT_TEXT[label]}. Overlap the loads."
    elif (p_insts is not None and insts is not None and p_time is not None
          and time_ is not None and insts > p_insts and time_ < p_time):
        label, sev = "COMPILER-OPACITY", "warning"
        msg = (f"{p_insts / 1e6:.2f}M -> {insts / 1e6:.2f}M instructions while "
               f"{p_time / 1e3:.0f} -> {time_ / 1e3:.0f} µs - {VERDICT_TEXT[label]}. Declare the layout statically.")
    elif (stall is not None and LOADPATH_STALL_MIN < stall <= LOADPATH_STALL_MAX
          and (_stall_share(kp, "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active") or 0.0) >= LOADPATH_LG_SHARE
          and pipe < LOADPATH_PIPE):
        label, sev = "LOAD-PATH", "warning"
        msg = f"{VERDICT_TEXT[label]} (stall {stall:.1f}, LSU-throttle {_stall_share(kp, 'smsp__average_warps_issue_stalled_lg_throttle_per_issue_active') or 0.0:.1f}%). Check layout and sector efficiency."
    elif (stall is not None and p_stall is not None and p_stall > 0
          and abs(stall - p_stall) / p_stall * 100.0 <= FREEZE_TOL
          and p_pipe is not None and abs(pipe - p_pipe) >= 5.0):
        label, sev = "ISSUE-SERIALIZATION", "warning"
        msg = f"stall total frozen ({p_stall:.1f} -> {stall:.1f}) while pipe moved {p_pipe:.1f} -> {pipe:.1f}% - {VERDICT_TEXT[label]}. Warp-specialize."
    elif pipe >= CONVERGED_PIPE and p_time and time_ and abs(time_ - p_time) / p_time * 100.0 <= CONVERGED_TIME_TOL:
        label, sev = "CONVERGED", "info"
        msg = f"pipe {pipe:.1f}%, time {time_ / 1e3:.0f} -> {p_time / 1e3:.0f} µs - {VERDICT_TEXT[label]}."
    elif pipe >= CONVERGING_PIPE and p_insts is not None and insts is not None and p_time and time_ and insts < p_insts and time_ < p_time:
        label, sev = "CONVERGING", "info"
        msg = f"{VERDICT_TEXT[label]} (pipe {pipe:.1f}%, instructions down, time down)."
    elif pipe < PIPE_BOUND_MAX and dram < 50.0 and tma < 60.0:
        label, sev = "PIPE-BOUND", "warning"
        msg = f"pipe {pipe:.1f}% with DRAM {dram:.1f}% and TMA {tma:.1f}% - {VERDICT_TEXT[label]}. Bigger tiles, cluster-level work."
    else:
        msg = VERDICT_TEXT[label] + "."

    if (warps is not None and p_warps is not None and warps - p_warps >= DENOM_WARP_DELTA):
        caveat = (f" caveat: achieved occupancy jumped {p_warps:.1f} -> {warps:.1f}%, "
                  "so the stall counter's denominator changed and its totals are not comparable - trust pipe + clock.")

    mover = _movers(kp, prev)
    msg = msg + (f" {mover}." if mover else "") + caveat
    return RuleResult("verdict", label, sev, msg,
                      kernel=kp.key, source="our",
                      focus={"stall_total_cycles": stall or 0.0, "pipe_pct": pipe})


def rules_for(kp: KernelProfile, prev: KernelProfile | None, cfg: dict) -> list[RuleResult]:
    """All rules that evaluate for one kernel (verdict last)."""
    from .sections import CONFIG_DEFAULTS

    cfg = {**CONFIG_DEFAULTS, **(cfg or {})}
    out = [
        rule_sol_bottleneck(kp, cfg),
        rule_occupancy_low(kp, cfg),
        rule_grid_small(kp, cfg),
        rule_dominant_stall(kp, cfg),
        rule_instruction_mix(kp, cfg),
        rule_tma_idle(kp, cfg),
        rule_dram_saturated(kp, cfg),
        rule_cache_behavior(kp, cfg),
        rule_bank_conflicts(kp, cfg),
        verdict(kp, prev, cfg),
    ]
    return [r for r in out if r is not None]
