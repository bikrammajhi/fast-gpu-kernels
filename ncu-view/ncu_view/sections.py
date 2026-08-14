"""Build the Nsight-Compute-style detail sections from a KernelProfile.

Counter-driven: a row exists only when the counter (or every counter in its
formula) is present in the profile. Derived rows carry `derived=True` and a
`note` stating the formula. What was never collected shows a single honest
"not collected" row instead of a fabricated zero.
"""

from __future__ import annotations

from .ingest import FALLBACK_MISSING, STALL_BASES, STALL_SHORT
from .model import KernelProfile, Row, Section

CONFIG_DEFAULTS = {
    "M": 8192,          # matrix size for TFLOPS (2*M^3 / time); None = unknown
    "dram_peak": 8.0e12,  # B200 HBM3e spec, bytes/s
    "tensor_peak": 2250.0,  # B200 dense FP16 peak, TFLOPS
    "smsp_per_sm": 4,    # SM sub-partitions per SM (Blackwell)
    "max_warps_per_sm": 64,  # Blackwell
    "max_smem_per_sm": 227.0 * 2 ** 20,  # B200 shared mem per SM, bytes
}

SOL_DESCRIPTION = (
    "Achieved utilization of compute and memory resources vs their "
    "theoretical peaks. The busiest resource bounds the kernel."
)
WARP_DESCRIPTION = (
    "Warp cycles per issued instruction (per issue-active cycle) and where "
    "they went. Only focus on stalls if the schedulers fail to issue every "
    "cycle."
)
COMPUTE_DESCRIPTION = (
    "Executed instructions and per-pipe utilization. A pipe near 100% may "
    "limit overall performance."
)
MEMORY_DESCRIPTION = (
    "Memory throughput, cache hit rates and shared-memory traffic. Memory "
    "can limit the kernel when units are fully busy or bandwidth is maxed."
)
OCCUPANCY_DESCRIPTION = (
    "Ratio of active warps per SM to the hardware maximum. Low achieved "
    "occupancy reduces the ability to hide latencies."
)
LAUNCH_DESCRIPTION = (
    "Grid and block configuration plus per-thread resources."
)
SCHED_DESCRIPTION = (
    "Warps in each scheduler's pool (theoretical, active, eligible) and how "
    "many issue slots were used. Skipped slots indicate poor latency hiding."
)
INST_DESCRIPTION = (
    "Instruction mix: which pipes the executed instructions were issued to."
)


def _val(kp: KernelProfile, metric: str) -> float | None:
    return kp.metrics.get(metric)


def _not_collected(what: str) -> Row:
    return Row(what, FALLBACK_MISSING, note="profile with `ncu --set full`")


def _fmt_ns_us(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:.1f} ms"
    return f"{ns / 1e3:.1f} µs"


def _fmt_bytes_gb(b: float) -> str:
    return f"{b / 1e9:.2f} GB"


def _fmt_per_second_gbs(bps: float) -> str:
    return f"{bps / 1e9:.1f} GB/s"


def _fmt_million(n: float) -> str:
    return f"{n / 1e6:.2f}M"


def section_sol(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("speedoflight", "GPU Speed Of Light Throughput", SOL_DESCRIPTION)
    t = _val(kp, "gpu__time_duration.avg")
    if t is not None:
        sec.rows.append(Row("Duration", _fmt_ns_us(t)))
    pipe = _val(kp, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    if pipe is not None:
        sec.rows.append(Row("Tensor pipe (compute proxy)", f"{pipe:.1f}", "%", bar=pipe,
                            derived=True, note="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"))
    dram_bps = _val(kp, "dram__bytes.sum.per_second")
    if dram_bps is not None:
        dram_pct = dram_bps / cfg["dram_peak"] * 100.0
        sec.rows.append(Row("DRAM (memory proxy)", f"{dram_pct:.1f}", "%", bar=dram_pct,
                            derived=True, note=f"dram__bytes.sum.per_second / {cfg['dram_peak']:.0e} B/s"))
        sec.rows.append(Row("DRAM throughput", _fmt_per_second_gbs(dram_bps)))
    warps = _val(kp, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if warps is not None:
        sec.rows.append(Row("Achieved occupancy (warps)", f"{warps:.1f}", "%", bar=warps))
    ctas = _val(kp, "sm__ctas_active.avg.pct_of_peak_sustained_active")
    if ctas is not None:
        sec.rows.append(Row("Active CTAs", f"{ctas:.1f}", "%", bar=ctas))
    cyc = _val(kp, "sm__cycles_elapsed.avg")
    if cyc is not None:
        sec.rows.append(Row("Elapsed cycles", f"{cyc:.0f}"))
    return sec


def _fmt_bytes(b: float) -> str:
    if b >= 2 ** 20:
        return f"{b / 2 ** 20:.1f} MiB"
    if b >= 2 ** 10:
        return f"{b / 2 ** 10:.1f} KiB"
    return f"{b:.0f} B"


def section_launch(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("launchstats", "Launch Statistics", LAUNCH_DESCRIPTION)
    m = kp.metrics

    def dims(prefix: str) -> str | None:
        xs = [m.get(f"launch__{prefix}_dim_{c}") for c in "xyz"]
        if not any(x is not None for x in xs):
            return None
        vals = [int(x) if x is not None else 1 for x in xs]
        return f"{vals[0]}x{vals[1]}x{vals[2]}"

    grid = dims("grid")
    block = dims("block")
    if grid:
        sec.rows.append(Row("Grid size", grid, derived=True,
                            note="launch__grid_dim_{x,y,z}"))
    grid_n = m.get("launch__grid_size")
    if grid_n is not None:
        sec.rows.append(Row("Total CTAs launched", f"{grid_n:.0f}"))
    if block:
        sec.rows.append(Row("Block size", block, derived=True,
                            note="launch__block_dim_{x,y,z}"))
    threads = m.get("launch__thread_count")
    if threads is not None:
        sec.rows.append(Row("Threads", f"{threads:.0f}"))
    regs = m.get("launch__registers_per_thread")
    if regs is not None:
        sec.rows.append(Row("Registers per thread", f"{regs:.0f}"))
    for key, label in [
        ("launch__shared_mem_per_block_static", "Static shared mem / block"),
        ("launch__shared_mem_per_block_dynamic", "Dynamic shared mem / block"),
        ("launch__shared_mem_per_block", "Shared mem / block (total)"),
        ("launch__shared_mem_per_block_allocated", "Shared mem / block (allocated)"),
    ]:
        v = m.get(key)
        if v is not None:
            sec.rows.append(Row(label, _fmt_bytes(v)))
    waves = m.get("launch__waves_per_multiprocessor")
    if waves is not None:
        sec.rows.append(Row("Waves per SM", f"{waves:.2f}",
                            derived=True, note="launch__waves_per_multiprocessor"))
    cluster = dims("cluster")
    if cluster and cluster != "1x1x1":
        sec.rows.append(Row("Cluster size", cluster, derived=True,
                            note="launch__cluster_dim_{x,y,z}"))
    # Theoretical occupancy: the binding occupancy limit, expressed in warps.
    limits = [
        ("launch__occupancy_limit_warps", 1),
        ("launch__occupancy_limit_blocks", None),
        ("launch__occupancy_limit_registers", None),
        ("launch__occupancy_limit_shared_mem", None),
    ]
    block_size = m.get("launch__block_size") or 32.0
    warp_limits = []
    for key, scale in limits:
        v = m.get(key)
        if v is None:
            continue
        warps = v * (scale or (block_size / 32.0))
        warp_limits.append(warps)
    if warp_limits and cfg["max_warps_per_sm"]:
        theo = min(warp_limits) / cfg["max_warps_per_sm"] * 100.0
        sec.rows.append(Row("Theoretical occupancy", f"{theo:.1f}", "%", bar=theo,
                            derived=True,
                            note="min(occupancy limits in warps) / max warps per SM"))
        for key, label in [
            ("launch__occupancy_limit_warps", "Occupancy limit: warps"),
            ("launch__occupancy_limit_blocks", "Occupancy limit: blocks"),
            ("launch__occupancy_limit_registers", "Occupancy limit: registers"),
            ("launch__occupancy_limit_shared_mem", "Occupancy limit: shared mem"),
        ]:
            v = m.get(key)
            if v is not None:
                sec.rows.append(Row(label, f"{v:.0f}"))
    if not sec.rows:
        sec.rows.append(_not_collected("launch statistics"))
    return sec


def section_occupancy(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("occupancy", "Occupancy", OCCUPANCY_DESCRIPTION)
    warps = _val(kp, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if warps is not None:
        sec.rows.append(Row("Achieved occupancy", f"{warps:.1f}", "%", bar=warps))
    ctas = _val(kp, "sm__ctas_active.avg.pct_of_peak_sustained_active")
    if ctas is not None:
        sec.rows.append(Row("Active CTAs", f"{ctas:.1f}", "%", bar=ctas))
    if not sec.rows:
        sec.rows.append(_not_collected("occupancy counters"))
    return sec


def section_scheduler(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("schedulerstats", "Scheduler Statistics", SCHED_DESCRIPTION)
    have = any(_val(kp, m) is not None for m in (
        "smsp__average_warps_issue_stalled_selected_per_issue_active",
        "smsp__average_warps_issue_stalled_not_selected_per_issue_active",
    ))
    if not have:
        sec.rows.append(_not_collected("scheduler pool / issue counters"))
        return sec
    total = section_stall_total(kp)
    selected = _val(kp, "smsp__average_warps_issue_stalled_selected_per_issue_active")
    not_sel = _val(kp, "smsp__average_warps_issue_stalled_not_selected_per_issue_active")
    if selected is not None:
        sec.rows.append(Row("Cycles issued (selected)", f"{selected:.2f}"))
    if not_sel is not None:
        sec.rows.append(Row("Cycles eligible but not selected", f"{not_sel:.2f}"))
    if total is not None:
        issued = selected or 0.0
        sec.rows.append(Row("Issue slots used per active cycle", f"{issued / total * 100:.1f}", "%",
                            derived=True, note="selected / stall_total per issue-active cycle"))
    return sec


def section_stall_total(kp: KernelProfile) -> float | None:
    vals = [_val(kp, b) for b in STALL_BASES]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def section_warpstate(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("warpstate", "Warp State Statistics", WARP_DESCRIPTION)
    total = section_stall_total(kp)
    if total is None:
        sec.rows.append(_not_collected("warp stall counters"))
        return sec
    sec.rows.append(Row("Warp cycles per issued instruction", f"{total:.2f}",
                        derived=True, note="sum of the 17 smsp__average_warps_issue_stalled_* counters"))
    ranked = sorted(
        ((b, _val(kp, b)) for b in STALL_BASES),
        key=lambda kv: kv[1] if kv[1] is not None else 0.0,
        reverse=True,
    )
    ranked = [(b, v) for b, v in ranked if v is not None and v > 0]
    for base, v in ranked[:8]:
        sec.rows.append(Row(STALL_SHORT[base], f"{v:.2f}", "cyc", bar=v / total * 100.0))
    other = sum(v for _, v in ranked[8:])
    if other > 0:
        sec.rows.append(Row("other (9 unlisted reasons)", f"{other:.2f}", "cyc", bar=other / total * 100.0))
    return sec


def section_compute(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("computeworkload", "Compute Workload Analysis", COMPUTE_DESCRIPTION)
    inst = _val(kp, "smsp__inst_executed.sum")
    if inst is not None:
        sec.rows.append(Row("Instructions executed", _fmt_million(inst)))
    pipe = _val(kp, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    if pipe is not None:
        sec.rows.append(Row("Tensor pipe utilization", f"{pipe:.1f}", "%", bar=pipe))
    cycles = _val(kp, "sm__cycles_elapsed.avg")
    if inst is not None and cycles is not None and cycles > 0:
        ipc = inst / (cycles * cfg["smsp_per_sm"])
        sec.rows.append(Row("IPC (per SM sub-partition)", f"{ipc:.3f}",
                            derived=True, note=f"smsp__inst_executed.sum / (sm__cycles_elapsed.avg * {cfg['smsp_per_sm']})"))
    t = _val(kp, "gpu__time_duration.avg")
    if t is not None and cfg["M"]:
        tflops = 2.0 * cfg["M"] ** 3 / (t * 1e-9) / 1e12
        sec.rows.append(Row("Achieved TFLOPS", f"{tflops:.1f}",
                            derived=True, note=f"2*{cfg['M']}**3 / duration"))
        if pipe is not None and pipe > 0:
            sec.rows.append(Row("TFLOPS per 100% pipe (per active cycle)",
                                f"{tflops / (pipe / 100.0):.1f}",
                                derived=True, note="Achieved TFLOPS / (tensor pipe % / 100)"))
    if not sec.rows:
        sec.rows.append(_not_collected("instruction counters"))
    return sec


def tma_pct(kp: KernelProfile) -> tuple[float | None, str]:
    """TMA (Tensormap) utilization as a percent, across metric generations.

    Returns (percent, provenance). Two spellings of the same fact:
    * probe path: raw cycles with TMA active on the xbar, normalized by
      elapsed cycles (l1tex__m_l1tex2xbar_req_cycles_active_op_tma).
    * ncu 2025 report exports: the counter is gone; ncu itself publishes
      xbar cycles as a percent of peak sustained (elapsed).
    """
    tma_cyc = _val(kp, "l1tex__m_l1tex2xbar_req_cycles_active_op_tma")
    cyc = _val(kp, "sm__cycles_elapsed.avg")
    if tma_cyc is not None and cyc and cyc > 0:
        return tma_cyc / cyc * 100.0, "l1tex__m_l1tex2xbar_req_cycles_active_op_tma / sm__cycles_elapsed.avg"
    for key in ("l1tex__m_l1tex2xbar_req_cycles_active.sum.pct_of_peak_sustained_elapsed",
                "l1tex__m_l1tex2xbar_req_cycles_active.avg.pct_of_peak_sustained_elapsed"):
        v = _val(kp, key)
        if v is not None:
            return v, f"{key} (ncu-normalized)"
    return None, ""


def _tma_row(kp: KernelProfile) -> Row | None:
    pct, note = tma_pct(kp)
    if pct is None:
        return None
    return Row("TMA active", f"{pct:.1f}", "%", bar=pct, derived=True, note=note)


def section_memory(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("memoryworkload", "Memory Workload Analysis", MEMORY_DESCRIPTION)
    dram_bps = _val(kp, "dram__bytes.sum.per_second")
    if dram_bps is not None:
        dram_pct = dram_bps / cfg["dram_peak"] * 100.0
        sec.rows.append(Row("DRAM throughput", f"{dram_pct:.1f}", "%", bar=dram_pct))
        sec.rows.append(Row("DRAM bandwidth", _fmt_per_second_gbs(dram_bps)))
        r = _val(kp, "dram__bytes_read.sum")
        w = _val(kp, "dram__bytes_write.sum")
        if r is not None and w is not None and r + w > 0:
            sec.rows.append(Row("DRAM read / write", f"{_fmt_bytes_gb(r)} / {_fmt_bytes_gb(w)}",
                                derived=True, note="dram__bytes_read.sum / dram__bytes_write.sum"))
            sec.rows.append(Row("Read share", f"{r / (r + w) * 100.0:.1f}", "%",
                                derived=True, note="read / (read + write)"))
    for metric, label in [
        ("l1tex__t_sector_hit_rate.pct", "L1/TEX hit rate"),
        ("lts__t_sector_hit_rate.pct", "L2 hit rate"),
    ]:
        v = _val(kp, metric)
        if v is not None:
            sec.rows.append(Row(label, f"{v:.1f}", "%", bar=v))
    tma_row = _tma_row(kp)
    if tma_row is not None:
        sec.rows.append(tma_row)
    conflicts = _val(kp, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    if conflicts is not None:
        sec.rows.append(Row("SMEM bank conflicts", f"{conflicts:.0f}"))
    waves = _val(kp, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    if waves is not None:
        sec.rows.append(Row("SMEM LSU wavefronts", f"{waves:.0f}"))
    if not sec.rows:
        sec.rows.append(_not_collected("memory counters"))
    return sec


def section_instruction(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("instructionstats", "Instruction Statistics", INST_DESCRIPTION)
    inst = _val(kp, "smsp__inst_executed.sum")
    if inst is None:
        sec.rows.append(_not_collected("instruction counters"))
        return sec
    sec.rows.append(Row("Instructions executed", _fmt_million(inst)))
    tens = _val(kp, "sm__inst_executed_pipe_tensor.sum")
    lsu = _val(kp, "sm__inst_executed_pipe_lsu.sum")
    if tens is not None:
        sec.rows.append(Row("Tensor pipe instructions", _fmt_million(tens),
                            bar=tens / inst * 100.0))
        sec.rows.append(Row("Tensor share", f"{tens / inst * 100.0:.1f}", "%",
                            derived=True, note="pipe_tensor / inst_executed"))
    if lsu is not None:
        sec.rows.append(Row("LSU pipe instructions", _fmt_million(lsu),
                            bar=lsu / inst * 100.0))
        sec.rows.append(Row("LSU share", f"{lsu / inst * 100.0:.1f}", "%",
                            derived=True, note="pipe_lsu / inst_executed"))
    known = (tens or 0.0) + (lsu or 0.0)
    other = inst - known
    if other > 0:
        sec.rows.append(Row("Other pipes (ALU/FMA/…) share", f"{other / inst * 100.0:.1f}", "%",
                            derived=True, note="1 - tensor_share - lsu_share"))
    return sec


def section_pm_sampling(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("pmsampling", "PM Sampling (PC sampling)",
                  "Sampled warps' PCs: where the hardware actually spends time. "
                  "Collected with --sampling-max-passes; absent if the profile "
                  "was taken without it.")
    for ns in kp.ncu_sections:
        if ns.sid == "PM Sampling":
            sec.rows = list(ns.rows)
            sec.table = ns.table
            sec.src = "ncu"
            return sec
    sec.rows.append(_not_collected("PC sampling table"))
    return sec


def section_nvlink(kp: KernelProfile, cfg: dict) -> Section:
    sec = Section("nvlink", "NvLink",
                  "NvLink topology and traffic between the GPU and its peers.")
    m = kp.metrics
    max_count = m.get("nvlink__max_count")
    enabled = m.get("nvlink__enabled_mask")
    peers = m.get("nvlink__count_physical") or 0
    logical = m.get("nvlink__count_logical") or 0
    peer_access = m.get("nvlink__peer_access") or 0
    direct = m.get("nvlink__is_direct_link") or 0
    switch = m.get("nvlink__is_nvswitch_connected") or 0
    bandwidth = m.get("nvlink__bandwidth")
    if max_count is not None:
        lanes = f"{max_count:.0f}"
        if enabled:
            lanes += f" enabled (mask 0x{int(enabled):x})"
        sec.rows.append(Row("NvLink lanes (device)", lanes,
                            derived=True, note="nvlink__max_count / nvlink__enabled_mask"))
        if peers:
            sec.rows.append(Row("Peers (physical links)", f"{peers:.0f}"))
            sec.rows.append(Row("Peers (logical links)", f"{logical:.0f}"))
            if bandwidth is not None and bandwidth > 0:
                sec.rows.append(Row("Link bandwidth", f"{bandwidth / 1e9:.0f} GB/s"))
        else:
            sec.rows.append(Row("Peers", "0",
                                note="single-GPU profile: no NvLink traffic to collect"))
        sec.rows.append(Row("Peer access", "yes" if peer_access else "no"))
        sec.rows.append(Row("Direct link", "yes" if direct else "no"))
        sec.rows.append(Row("NVSwitch connected", "yes" if switch else "no"))
    if not sec.rows:
        sec.rows.append(_not_collected("NvLink counters"))
    return sec


def sections_for(kp: KernelProfile, cfg: dict | None = None) -> list[Section]:
    cfg = {**CONFIG_DEFAULTS, **(cfg or {})}
    return [
        section_sol(kp, cfg),
        section_launch(kp, cfg),
        section_occupancy(kp, cfg),
        section_scheduler(kp, cfg),
        section_warpstate(kp, cfg),
        section_compute(kp, cfg),
        section_memory(kp, cfg),
        section_instruction(kp, cfg),
        section_pm_sampling(kp, cfg),
        section_nvlink(kp, cfg),
    ]
