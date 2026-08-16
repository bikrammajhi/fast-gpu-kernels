"""Assemble kernels and NVIDIA's own sections/rules into one report artifact.

`build` returns a plain-JSON-safe dict that every renderer (HTML report,
terminal summary, live server) consumes. Everything in it is NVIDIA's own
data: the sections and rule results exported from the .ncu-rep, plus raw
official counters for the summary chips. No analysis of our own is
invented on top.
"""

from __future__ import annotations

from . import ingest as _ingest
from .ingest import ingest
from .model import KernelProfile, RuleResult, Section

STALL_TOP = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active"

SEV_RANK = {"critical": 0, "warning": 1, "suggestion": 2, "info": 3, "hint": 4}


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


def _ncu_verdict(kp: KernelProfile) -> RuleResult | None:
    """The banner rule: NVIDIA's Speed Of Light bottleneck rule when the
    rule engine reported one, else its most severe rule."""
    for r in kp.ncu_rules:
        if r.rid == "SOLBottleneck":
            return r
    ranked = sorted(kp.ncu_rules,
                    key=lambda r: SEV_RANK.get(r.severity, 9))
    return ranked[0] if ranked else None


def _sm_freq(kp: KernelProfile) -> float | None:
    """SM clock: NVIDIA's Speed Of Light "SM Frequency" row when exported,
    else the same formula from ncu's own counters."""
    for sec in kp.ncu_sections:
        if sec.sid != "SpeedOfLight":
            continue
        for r in sec.rows:
            if r.label == "SM Frequency":
                try:
                    return float(r.value)
                except (TypeError, ValueError):
                    return None
    m = kp.metrics
    cycles, t = m.get("sm__cycles_elapsed.avg"), m.get("gpu__time_duration.avg")
    return cycles / t if (cycles and t) else None


def _kernel_dict(kp: KernelProfile) -> dict:
    verdict = _ncu_verdict(kp)
    m = kp.metrics
    stats = {
        "time_us": m.get("gpu__time_duration.avg") / 1e3
        if m.get("gpu__time_duration.avg") else None,
        "clock_ghz": _sm_freq(kp),
        "pipe_pct": m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
        "dram_pct": _ingest.dram_pct_of_peak(kp),
        "occupancy_pct": m.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "stall_cycles": _ingest.stall_total(kp),
        "top_stall_pct": m.get(STALL_TOP),
    }
    return {
        "key": kp.key,
        "name": kp.name,
        "provenance": kp.provenance,
        "stats": stats,
        "verdict": _rule_dict(verdict) if verdict else None,
        "rules": [_rule_dict(r) for r in kp.ncu_rules],
        "sections": [_section_dict(s) for s in kp.ncu_sections],
    }


def _series_row(kp: KernelProfile) -> dict:
    m = kp.metrics
    verdict = _ncu_verdict(kp)
    return {
        "key": kp.key,
        "name": kp.name,
        "time_us": m.get("gpu__time_duration.avg") / 1e3
        if m.get("gpu__time_duration.avg") else None,
        "clock_ghz": _sm_freq(kp),
        "pipe_pct": m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
        "dram_pct": _ingest.dram_pct_of_peak(kp),
        "occupancy_pct": m.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "stall_cycles": _ingest.stall_total(kp),
        "top_stall": m.get(STALL_TOP),
        "verdict": verdict.name if verdict else None,
    }


def build(path: str, kernel: str | None = None) -> dict:
    profs = ingest(path, kernel=kernel)
    kernels = [_kernel_dict(kp) for kp in profs]
    series = [_series_row(kp) for kp in profs]
    return {
        "meta": {
            "input": (", ".join(str(p) for p in path) if isinstance(path, (list, tuple))
                      else str(path)),
            "source": profs[0].provenance["source"] if profs else None,
            "kernels": len(profs),
            "noise_dropped": _ingest.noise_dropped,
            "device": {
                "name": profs[0].device_name if profs else None,
            },
        },
        "kernels": kernels,
        "series": series,
    }
