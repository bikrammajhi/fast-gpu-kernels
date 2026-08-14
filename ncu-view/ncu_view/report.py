"""Assemble kernels, sections and rules into one analysis artifact.

`build` returns a plain-JSON-safe dict that every renderer (HTML report,
terminal summary, live server) consumes. Rules keep their `source` tag
("our" today; "ncu" once .ncu-rep ingest lands in Phase 2).
"""

from __future__ import annotations

from .ingest import ingest
from .model import KernelProfile, RuleResult, Section
from .rules import rules_for
from .sections import section_stall_total, sections_for

STALL_TOP = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active"


def _rule_dict(r: RuleResult) -> dict:
    return {
        "rid": r.rid,
        "name": r.name,
        "severity": r.severity,
        "message": r.message,
        "kernel": r.kernel,
        "est": r.est,
        "source": r.source,
        "focus": r.focus,
        "focus_info": r.focus_info,
    }


def _section_dict(s: Section) -> dict:
    return {
        "sid": s.sid,
        "title": s.title,
        "description": s.description,
        "src": s.src,
        "detailed": s.detailed,
        "rows": [
            {"label": r.label, "value": r.value, "unit": r.unit,
             "bar": r.bar, "derived": r.derived, "note": r.note}
            for r in s.rows
        ],
        "table": s.table,
    }


def _kernel_dict(kp: KernelProfile, rules: list[RuleResult],
                 sections: list[Section], cfg: dict) -> dict:
    verdict = next((r for r in rules if r.rid == "verdict"), None)
    m = kp.metrics
    t = m.get("gpu__time_duration.avg")
    pipe = m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    dram_bps = m.get("dram__bytes.sum.per_second")
    occ = m.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    cycles = m.get("sm__cycles_elapsed.avg")
    clock_ghz = cycles / t if (cycles and t) else None
    tflops = 2.0 * cfg["M"] ** 3 / (t * 1e-9) / 1e12 if (t and cfg["M"]) else None
    stall = section_stall_total(kp)
    top = m.get(STALL_TOP)
    stats = {
        "time_us": t / 1e3 if t else None,
        "tflops": tflops,
        "clock_ghz": clock_ghz,
        "pipe_pct": pipe,
        "dram_pct": dram_bps / cfg["dram_peak"] * 100.0 if dram_bps else None,
        "occupancy_pct": occ,
        "stall_cycles": stall,
        "top_stall_pct": top,
    }
    return {
        "key": kp.key,
        "name": kp.name,
        "provenance": kp.provenance,
        "stats": stats,
        "verdict": _rule_dict(verdict) if verdict else None,
        "rules": [_rule_dict(r) for r in rules],
        "sections": [_section_dict(s) for s in sections],
        "ncu_sections": [_section_dict(s) for s in kp.ncu_sections],
        "ncu_rules": [_rule_dict(r) for r in kp.ncu_rules],
    }


def _series_row(kp: KernelProfile, cfg: dict) -> dict:
    t = kp.metrics.get("gpu__time_duration.avg")
    pipe = kp.metrics.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    dram_bps = kp.metrics.get("dram__bytes.sum.per_second")
    occ = kp.metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    cycles = kp.metrics.get("sm__cycles_elapsed.avg")
    stall = section_stall_total(kp)
    top_stall = kp.metrics.get(STALL_TOP)
    tflops = 2.0 * cfg["M"] ** 3 / (t * 1e-9) / 1e12 if (t and cfg["M"]) else None
    row = {
        "key": kp.key,
        "name": kp.name,
        "time_us": t / 1e3 if t else None,
        "tflops": tflops,
        "clock_ghz": cycles / t if (cycles and t) else None,
        "pipe_pct": pipe,
        "dram_pct": dram_bps / cfg["dram_peak"] * 100.0 if dram_bps else None,
        "occupancy_pct": occ,
        "stall_cycles": stall,
        "top_stall": top_stall,
    }
    return row


def build(path: str, cfg: dict | None = None, kernel: str | None = None,
          gpu: str | None = None) -> dict:
    from .gpus import cfg_for_device, detect_device

    profs = ingest(path, kernel=kernel)
    device = profs[0].device
    if gpu:
        device = detect_device(name=gpu)
    cfg = cfg_for_device(device, cfg)
    kernels = []
    series = []
    prev: KernelProfile | None = None
    for kp in profs:
        rules = rules_for(kp, prev, cfg)
        sections = sections_for(kp, cfg)
        kernels.append(_kernel_dict(kp, rules, sections, cfg))
        series.append(_series_row(kp, cfg))
        prev = kp
    return {
        "meta": {
            "input": (", ".join(str(p) for p in path) if isinstance(path, (list, tuple))
                      else str(path)),
            "source": profs[0].provenance["source"] if profs else None,
            "kernels": len(profs),
            "device": {
                "name": device.name if device else None,
                "detected": device.detected if device else False,
                "matched": device.matched if device else False,
                "note": device.note if device else "",
                "tensor_peak_tflops": cfg["tensor_peak"],
                "dram_peak_gbps": cfg["dram_peak"] / 1e9,
                "max_warps_per_sm": cfg["max_warps_per_sm"],
                "smsp_per_sm": cfg["smsp_per_sm"],
            },
        },
        "kernels": kernels,
        "series": series,
    }
