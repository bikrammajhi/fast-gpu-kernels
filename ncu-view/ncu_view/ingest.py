"""Ingest kernel profiles from the supported inputs.

1. results/ncu_counters.json -- the probe set written by ncu_profile_all.py.
2. raw ncu CSV dumps (`ncu --page raw --csv`).
3. .ncu-rep report files (via NVIDIA's own ncu_report; importable when the
   module is available).

Every input is normalized to the same canonical form: stall counters carry
their cycles-per-issue-active value, and nothing is derived here -- derived
rows live in sections.py, rules in rules.py.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import suppress
from pathlib import Path

from .model import KernelProfile, Row, RuleResult, Section

NOISE_TENSOR_PCT = 5.0
NOISE_MAX_DURATION_NS = 20e6  # plumbing is short-lived; a real scalar kernel is not
NOISE_NAMES = ("__nvcc_device_query", "at::", "distribution_",
               "elementwise", "reduce_", "memcpy_", "memset_",
               "memcpy", "cudaMemcpy")
noise_dropped = 0


def _is_noise(kp: KernelProfile) -> bool:
    """Runtime plumbing (torch init/compare, device query, memcpy) never
    drives the tensor pipe, so the GEMM report drops it. A low tensor-pipe
    % alone is NOT plumbing — scalar kernels legitimately never touch the
    tensor pipe — so the metric rule also requires a plumbing-shaped
    duration. The name fallback only fires when the metric is absent
    (old ncu / curated inputs)."""
    if any(n in kp.name for n in NOISE_NAMES):
        return True
    t = kp.metrics.get(
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    if t is None:
        return False
    if t >= NOISE_TENSOR_PCT:
        return False
    d = kp.metrics.get("gpu__time_duration.avg")
    return d is not None and d < NOISE_MAX_DURATION_NS


STALL_BASES = [
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active",
    "smsp__average_warps_issue_stalled_wait_per_issue_active",
    "smsp__average_warps_issue_stalled_membar_per_issue_active",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_selected_per_issue_active",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active",
    "smsp__average_warps_issue_stalled_drain_per_issue_active",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active",
    "smsp__average_warps_issue_stalled_misc_per_issue_active",
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active",
    "smsp__average_warps_issue_stalled_warpgroup_arrive_per_issue_active",
]

STALL_SHORT = {
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active": "long scoreboard",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active": "short scoreboard",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active": "barrier",
    "smsp__average_warps_issue_stalled_wait_per_issue_active": "wait",
    "smsp__average_warps_issue_stalled_membar_per_issue_active": "membar",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active": "MIO throttle",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active": "math pipe throttle",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active": "dispatch stall",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active": "not selected",
    "smsp__average_warps_issue_stalled_selected_per_issue_active": "selected",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active": "tex throttle",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active": "sleeping",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active": "LSU throttle",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active": "branch resolving",
    "smsp__average_warps_issue_stalled_drain_per_issue_active": "drain",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active": "no instructions",
    "smsp__average_warps_issue_stalled_misc_per_issue_active": "misc",
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active": "imc miss",
    "smsp__average_warps_issue_stalled_warpgroup_arrive_per_issue_active": "warpgroup arrive",
}

FALLBACK_MISSING = "n/a (not collected)"


def _display_name(key: str) -> str:
    if key == "cublas":
        return "cuBLAS"
    if key.startswith("matmul_"):
        return key[len("matmul_"):]
    return key


def _canonical(metrics: dict[str, float]) -> dict[str, float]:
    """Map the many spellings of one counter onto one canonical key.

    * probe JSON: `base` (raw value) and `base + ".raw"` / `base + ".share"`.
    * raw CSV / report metrics: `base + ".ratio"` (report-export spelling).
    * ncu-report 2026 API: domain-prefixed names, already aliased to base.
    The canonical value is the raw cycles-per-issue-active figure; `.share`
    variants are preserved for the probe path.
    """
    out = dict(metrics)
    for base in STALL_BASES:
        raw = metrics.get(base + ".raw")
        share = metrics.get(base)
        ratio = metrics.get(base + ".ratio")
        if raw is None and share is None and ratio is None:
            continue
        out[base] = raw if raw is not None else (ratio if ratio is not None else share)
        if share is not None:
            out[base + ".share"] = share
    return out


def ingest_json(path: str | Path, kernel: str | None = None) -> list[KernelProfile]:
    """Read an ncu_counters.json probe file (see ncu_profile_all.py)."""
    data = json.loads(Path(path).read_text())
    keys = [kernel] if kernel else [k for k in data if k != "cublas"] + (["cublas"] if "cublas" in data else [])
    profs = []
    for key in keys:
        if key not in data:
            raise KeyError(f"kernel {key!r} not found in {path}")
        profs.append(
            KernelProfile(
                key=key,
                name=_display_name(key),
                metrics=_canonical(data[key]),
                provenance={"source": "json", "path": str(path)},
            )
        )
    return profs


# Unit scales for the wide `ncu --import --page raw --csv` dump: that CSV
# prints display units (ms, Gbyte/s) where the report API and the probe JSON
# give SI base units (ns, B/s). The wide path scales by unit; the long path
# is left untouched (the probe reads it verbatim).
_WIDE_UNIT_SCALE = {
    "ms": 1e6, "us": 1e3, "s": 1.0,
    "Gbyte/s": 1e9, "Mbyte/s": 1e6, "Kbyte/s": 1e3, "byte/s": 1.0,
    "Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1.0,
    "Gbit/s": 1e9 / 8, "Mbit/s": 1e6 / 8,
}


def ingest_csv(path: str | Path, kernel: str | None = None) -> list[KernelProfile]:
    """Read a raw ncu CSV dump.

    Handles both layouts ncu emits:
    * long (`ncu --metrics ... --csv`): one row per (kernel, metric).
    * wide (`ncu --import rep --page raw --csv` in recent ncu): one row per
      kernel, metric names in the header columns and a units row beneath.
    """
    text = Path(path).read_text()
    rows = list(csv.reader(io.StringIO(text)))
    groups: dict[str, dict[str, float]] = {}

    def sniff_wide(rows) -> bool:
        if not rows:
            return False
        header = rows[0]
        if len(header) < 10:
            return False
        if "Kernel Name" in header and "Metric Name" not in header:
            return True
        metricy = sum(1 for h in header[5:15] if "__" in h)
        return metricy >= 8

    if sniff_wide(rows):
        header = rows[0]
        idx = header.index("Kernel Name") if "Kernel Name" in header else 4
        metric_cols = header[idx + 1:]
        # The units row is the first row whose metric cells are non-numeric.
        scale = [1.0] * len(metric_cols)
        for row in rows[1:]:
            if len(row) <= idx:
                continue
            if all(not cell.strip().replace(".", "").replace("-", "").isdigit()
                   for cell in row[idx + 1:] if cell.strip()):
                for i, _ in enumerate(metric_cols):
                    if i + idx + 1 >= len(row):
                        break
                    unit = row[i + idx + 1].strip()
                    scale[i] = _WIDE_UNIT_SCALE.get(unit, 1.0)
                break
        for row in rows[1:]:
            if len(row) <= idx:
                continue
            kern = row[idx].strip()
            if not kern or kern.startswith("==") or kern == "Kernel Name":
                continue
            g = groups.setdefault(kern, {})
            for i, name in enumerate(metric_cols):
                if i + idx + 1 >= len(row):
                    break
                value = row[i + idx + 1].strip()
                if not value or value in ("N/A", "n/a", "-"):
                    continue
                try:
                    g[name] = float(value) * scale[i]
                except ValueError:
                    if name.startswith("device__attribute_"):
                        g.setdefault("__str__:" + name, value)
                    continue
    else:
        for row in rows:
            if len(row) < 4:
                continue
            kern, metric, _, value = row[0], row[1], row[2], row[3]
            if not metric or metric.startswith("==") or kern == "Kernel Name":
                continue
            try:
                groups.setdefault(kern, {})[metric] = float(value)
            except ValueError:
                continue
    keys = [kernel] if kernel else list(groups)
    profs = []
    for key in keys:
        if key not in groups:
            raise KeyError(f"kernel {key!r} not found in {path}")
        raw = groups[key]
        str_metrics = {k[len("__str__:"):]: v for k, v in raw.items()
                       if k.startswith("__str__:")}
        profs.append(
            KernelProfile(
                key=key,
                name=_display_name(key),
                metrics=_canonical(raw),
                str_metrics=str_metrics,
                provenance={"source": "csv", "path": str(path)},
            )
        )
    return profs


def _ncu_kernels(ctx) -> list:
    """All kernels in an ncu_report context, regardless of API version."""
    if hasattr(ctx, "kernels"):
        return list(ctx.kernels())
    out = []
    ranges = ctx.ranges() if hasattr(ctx, "ranges") else list(ctx)
    for rng in ranges:
        kern = rng.kernels() if hasattr(rng, "kernels") else list(rng)
        out.extend(kern)
    return out


def _ncu_rows(section) -> list[Row]:
    rows = []
    try:
        raw = section.rows()
    except Exception:
        return rows
    for r in raw:
        try:
            rows.append(
                Row(
                    label=r.name(),
                    value=str(r.value()),
                    unit=r.unit() if hasattr(r, "unit") else "",
                    bar=_ncu_value(r.bar()) if hasattr(r, "bar") else None,
                    derived=bool(r.is_derived()) if hasattr(r, "is_derived") else False,
                    note=r.info() if hasattr(r, "info") else "",
                )
            )
        except Exception:
            continue
    return rows


def _ncu_value(v):
    """The float behind an ncu_report MetricValue (or a raw number)."""
    if hasattr(v, "value"):
        try:
            return float(v.value())
        except Exception:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ncu_severity(sev) -> str:
    s = str(sev).split(".")[-1].lower()
    for known in ("critical", "warning", "suggestion", "info"):
        if known in s:
            return known
    try:
        return {0: "info", 1: "suggestion", 2: "warning", 3: "critical"}[int(sev)]
    except (TypeError, ValueError):
        return "info"


def _ncu_speedup(value) -> str:
    """'est' text from the new API's speedup_estimation payload."""
    if not value:
        return ""
    try:
        import ast

        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = json.loads(value)
        if isinstance(value, dict):
            return f"{float(value['speedup']):.1f}x"
    except Exception:
        pass
    return ""


def _ncu_rules_new(action, kp: KernelProfile, out_rules: list, out_sections: dict):
    """New PyPI ncu_report API: rule_results_as_dicts() + metric names."""
    try:
        rd = action.rule_results_as_dicts()
    except Exception:
        return
    for rule in rd:
        try:
            msg = rule.get("rule_message") or {}
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except Exception:
                    msg = {"message": msg}
            sev = _ncu_severity(msg.get("type", 2))
            focus = rule.get("focus_metrics") or []
            if not isinstance(focus, list):
                try:
                    focus = json.loads(focus)
                except Exception:
                    focus = []
            out_rules.append(
                RuleResult(
                    rid=str(rule.get("rule_identifier", rule.get("name", ""))),
                    name=str(rule.get("name", rule.get("rule_identifier", ""))),
                    severity=sev,
                    message=str(msg.get("message", "")),
                    est=_ncu_speedup(rule.get("speedup_estimation")),
                    source="ncu",
                    focus={str(f.get("name")): float(f.get("value", 0.0)) for f in focus},
                    focus_info={str(f.get("name")): str(f.get("info", "")) for f in focus
                                if f.get("info")},
                )
            )
            section = str(rule.get("section_identifier", ""))
            out_sections.setdefault(section, []).append(out_rules[-1])
        except Exception:
            continue


NCU_SECTION_TITLES = {
    # NVIDIA's raw section ids -> readable titles (ncu's own GUI names)
    "SpeedOfLight": "Speed Of Light",
    "SpeedOfLight_RooflineChart": "Speed Of Light \u2014 Roofline",
    "SpeedOfLight_HierarchicalDoubleRooflineChart":
        "Speed Of Light \u2014 Hierarchical Roofline (Double)",
    "SpeedOfLight_HierarchicalHalfRooflineChart":
        "Speed Of Light \u2014 Hierarchical Roofline (Half)",
    "SpeedOfLight_HierarchicalSingleRooflineChart":
        "Speed Of Light \u2014 Hierarchical Roofline (Single)",
    "SpeedOfLight_HierarchicalTensorRooflineChart":
        "Speed Of Light \u2014 Hierarchical Roofline (Tensor)",
    "SchedulerStats": "Scheduler Statistics",
    "WarpStateStats": "Warp State Statistics",
    "ComputeWorkloadAnalysis": "Compute Workload Analysis",
    "InstructionStats": "Instruction Statistics",
    "LaunchStats": "Launch Statistics",
    "MemoryWorkloadAnalysis": "Memory Workload Analysis",
    "MemoryWorkloadAnalysis_Tables": "Memory Workload Analysis",
    "MemoryWorkloadAnalysis_Chart": "Memory Workload Analysis \u2014 Chart",
    "NUMA Affinity": "NUMA Affinity",
    "Occupancy": "Occupancy",
    "SourceCounters": "Source Counters",
    "PM Sampling": "PM Sampling",
    "PM Sampling: Warp States": "PM Sampling: Warp States",
    "Nvlink": "NVLink",
    "Nvlink_Tables": "NVLink Tables",
    "Nvlink_Topology": "NVLink Topology",
    "C2CLink": "C2C Link",
    "WorkloadDistribution": "Workload Distribution",
    "Tile": "Tile Statistics",
}


def _ncu_section(sid: str, title: str | None = None, description: str = "",
                 rows: list[Row] | None = None, table: list[list[str]] | None = None,
                 detailed: bool = False) -> Section:
    """A section from NVIDIA's own report, tagged and human-titled."""
    pretty = NCU_SECTION_TITLES.get(sid, title or sid)
    return Section(
        sid=sid,
        title=pretty,
        description=description or "as reported by NVIDIA's ncu_report module",
        rows=rows or [],
        table=table,
        src="NVIDIA",
        detailed=detailed,
    )


def _sec_csv_kernels(text: str) -> set[str]:
    """Distinct kernel names in a section CSV (empty when no column)."""
    parsed = list(csv.reader(io.StringIO(text)))
    for i, row in enumerate(parsed):
        flat = [c.strip().lower() for c in row]
        if "metric name" in flat:
            header = [c.strip().lower() for c in row]
            ki = header.index("kernel name") if "kernel name" in header else None
            if ki is None:
                return set()
            return {r[ki].strip() for r in parsed[i + 1:]
                    if len(r) > ki and r[ki].strip()}
    return set()


def _parse_sec_csv(text: str, sid: str,
                   kernel_name: str | None = None) -> list[Row]:
    """Rows from `ncu --import rep --section <sid> --csv`.

    ncu 2025+ emits a long CSV: a header row naming the columns, then one
    row per metric with "Section Name","Metric Name","Metric Unit",
    "Metric Value" columns. Rows of every captured kernel share one
    header; `kernel_name` keeps only rows whose kernel column contains it
    (ncu exports full mangled names; the report keys on the short name).
    Older builds may emit plain label/value/unit rows instead; both are
    handled here.
    """
    parsed = list(csv.reader(io.StringIO(text)))
    header_idx = None
    for i, row in enumerate(parsed):
        flat = [c.strip().lower() for c in row]
        if "metric name" in flat:
            header_idx = i
            break
    if header_idx is not None:
        header = [c.strip().lower() for c in parsed[header_idx]]
        i_name = header.index("metric name")
        i_val = header.index("metric value") if "metric value" in header else None
        i_unit = header.index("metric unit") if "metric unit" in header else None
        i_kernel = header.index("kernel name") if "kernel name" in header else None
        if i_val is None:
            return []
        rows = []
        for row in parsed[header_idx + 1:]:
            if len(row) <= max(i_name, i_val):
                continue
            if i_kernel is not None and kernel_name is not None \
                    and kernel_name not in row[i_kernel]:
                continue
            label = row[i_name].strip()
            if not label or label.lower() == "metric name":
                continue
            value_s = row[i_val].replace(",", "").strip()
            try:
                value = float(value_s)
            except ValueError:
                continue
            unit = row[i_unit].strip() if i_unit is not None and i_unit < len(row) else ""
            bar = value if (unit == "%" and 0.0 <= value <= 100.0) else None
            rows.append(Row(label=label, value=f"{value:g}", unit=unit, bar=bar))
        return _dedupe_rows(rows)
    # Fallback: simple label/value(/unit) rows.
    out = []
    for row in parsed:
        cells = [c.strip() for c in row]
        if not cells or not cells[0]:
            continue
        first = cells[0].lower()
        if first.startswith("==") or first == "section" or "metric name" in first:
            continue
        if len(cells) < 2:
            continue
        try:
            value = float(cells[1].replace(",", "").strip())
        except ValueError:
            continue
        unit = cells[2].strip() if len(cells) > 2 else ""
        bar = value if (unit == "%" and 0.0 <= value <= 100.0) else None
        out.append(Row(label=cells[0], value=f"{value:g}", unit=unit, bar=bar))
    return _dedupe_rows(out)


def _dedupe_rows(rows: list[Row]) -> list[Row]:
    """A multi-launch capture exports one row block per launch; keep the
    last block (steady state, matching the deduped kernel profile)."""
    by_label: dict[str, Row] = {}
    for r in rows:
        by_label[r.label] = r
    return list(by_label.values())


def _sec_csv_files(path: Path) -> list[tuple[str, Path]]:
    """Sibling '<stem>.sec-<sid>.csv' exports next to a report file."""
    out = []
    for p in sorted(path.parent.glob(path.stem + ".sec-*.csv")):
        sid = p.stem.split(".sec-", 1)[-1]
        if sid and sid not in (s for s, _ in out):
            out.append((sid, p))
    return out


def _apply_sec_csvs(rep_path: Path, profs: list[KernelProfile],
                    kernel: str | None = None) -> None:
    """Overlay NVIDIA's own detailed sections exported next to the report.

    `ncu --import <rep> --section <X> --csv` produces one file per section
    (see profile.py / modal_app.py). When present, the one-sentence rule
    section of the same sid is replaced by NVIDIA's full table.
    """
    for sid, p in _sec_csv_files(rep_path):
        try:
            text = p.read_text()
        except OSError:
            continue
        multi = len(_sec_csv_kernels(text)) > 1
        for kp in profs:
            if kernel and kernel not in kp.key:
                continue
            rows = _parse_sec_csv(text, sid, kp.name if multi else None)
            if not rows:
                continue
            sec = _ncu_section(
                sid=sid,
                description=f"NVIDIA's own {NCU_SECTION_TITLES.get(sid, sid)} "
                            "table (ncu --import --section)",
                rows=rows,
                detailed=True,
            )
            for i, old in enumerate(kp.ncu_sections):
                if old.sid == sid:
                    kp.ncu_sections[i] = sec
                    break
            else:
                kp.ncu_sections.append(sec)


def ingest_report(path: str | Path, kernel: str | None = None) -> list[KernelProfile]:
    """Read a .ncu-rep report via NVIDIA's ncu_report module.

    Supports both API generations: the extras/python module shipped with
    Nsight Compute (sections()/rules()/get_metrics_by_name) and the PyPI
    ncu-report package (metric_names()/rule_results_as_dicts()/
    timed_warp_samples()). Raises ImportError when neither is available.

    Metrics from the new API are domain-prefixed (e.g.
    "FBSP.TriageCompute.smsp__…"); they are also aliased to their bare
    counter name so sections and the diff harness work unchanged.
    """
    import ncu_report  # type: ignore

    ctx = ncu_report.load_report(str(path))
    profs = []
    for action in _ncu_kernels(ctx):
        name = action.name()
        if kernel and kernel not in name:
            continue
        metrics = {}
        str_metrics: dict[str, str] = {}
        try:
            names = list(action.metric_names())
            has_metric_by_name = hasattr(action, "metric_by_name")
        except Exception:
            names = list(action)
            has_metric_by_name = False
        for metric_name in names:
            try:
                if has_metric_by_name:
                    m = action.metric_by_name(metric_name)
                else:
                    m = action[metric_name]
                val = _ncu_value(m)
            except Exception:
                continue
            if val is None:
                try:
                    s = m.as_string()
                except Exception:
                    s = None
                if s:
                    str_metrics[metric_name] = s
                continue
            metrics[metric_name] = val
            base = metric_name.rsplit(".", 1)[-1]
            if base != metric_name and base not in metrics:
                metrics[base] = val
        kp = KernelProfile(
            key=name,
            name=_display_name(name),
            metrics=_canonical(metrics),
            str_metrics=str_metrics,
            provenance={"source": "ncu-rep", "path": str(path)},
        )
        if hasattr(action, "sections"):
            for sec in action.sections():
                try:
                    table = None
                    if hasattr(sec, "table"):
                        raw = sec.table()
                        if raw:
                            table = [[str(c) for c in row] for row in raw]
                    kp.ncu_sections.append(
                        _ncu_section(
                            sid=sec.name(),
                            title=sec.title() if hasattr(sec, "title") else sec.name(),
                            description=sec.description() if hasattr(sec, "description") else "",
                            rows=_ncu_rows(sec),
                            table=table,
                        )
                    )
                except Exception:
                    continue
        # New API: rules + synthetic per-section views.
        by_section: dict[str, list] = {}
        _ncu_rules_new(action, kp, kp.ncu_rules, by_section)
        for sid, rules in by_section.items():
            kp.ncu_sections.append(
                _ncu_section(
                    sid=sid,
                    description="NVIDIA rule engine output (PyPI ncu-report)",
                    rows=[
                        Row(f"{r.name} [{r.severity}]", r.message,
                            derived=True, note=f"Est. speedup: {r.est}" if r.est else "")
                        for r in rules
                    ],
                )
            )
        # PM sampling (new API). On builds where timed_warp_samples() is
        # empty even with --pm-sampling-max-passes, the sampling counters
        # (smsp__pcsamp_*) and per-PC source markers still carry the data.
        if hasattr(action, "timed_warp_samples"):
            try:
                samples = action.timed_warp_samples() or []
                markers = []
                if hasattr(action, "source_markers"):
                    try:
                        markers = action.source_markers()
                    except Exception:
                        markers = []
                rows = []
                for label, key in [
                    ("Sample count", "smsp__pcsamp_sample_count"),
                    ("Aggregated passes", "smsp__pcsamp_aggregated_passes"),
                    ("Dropped bytes", "smsp__pcsamp_dropped_bytes"),
                    ("Sampling interval (cycles)", "smsp__pcsamp_interval_cycles"),
                ]:
                    v = metrics.get(key)
                    if v is not None:
                        rows.append(Row(label, f"{v:,.0f}"))
                table = None
                if markers or samples:
                    table = [["Source address", "Instruction (SASS)", "Attribution"]]
                    seen = set()
                    for mk in markers:
                        addr = mk.get("source_address")
                        if addr is None or addr in seen:
                            continue
                        seen.add(addr)
                        sass = ""
                        with suppress(Exception):
                            sass = (action.sass_by_pc(int(addr)) or "").strip()
                        table.append([
                            f"0x{int(addr):x}" if isinstance(addr, int) else str(addr),
                            sass,
                            str(mk.get("message", "")),
                        ])
                    for s in samples:
                        addr = getattr(s, "pc", None)
                        try:
                            addr = addr() if callable(addr) else None
                        except Exception:
                            addr = None
                        if addr is None or addr in seen:
                            continue
                        seen.add(addr)
                        table.append([
                            f"0x{int(addr):x}",
                            "",
                            str(s.num_samples()) + " samples",
                        ])
                    if len(table) > 17:
                        table = table[:17]
                        table.append(["…", f"{len(seen) - 15} more sampled locations", ""])
                if rows or table:
                    kp.ncu_sections.append(
                        _ncu_section(
                            sid="PM Sampling",
                            title="PM Sampling",
                            description="Sampled warps' PCs and source attribution "
                                        "(sampling counters + per-PC source markers)",
                            rows=rows,
                            table=table,
                            detailed=True,
                        )
                    )
            except Exception:
                pass
        if hasattr(action, "rules"):
            for rule in action.rules():
                try:
                    est = _ncu_value(rule.estimated_speedup()) if hasattr(rule, "estimated_speedup") else None
                    kp.ncu_rules.append(
                        RuleResult(
                            rid=str(rule.name()),
                            name=str(rule.name()),
                            severity=_ncu_severity(rule.severity()),
                            message=str(rule.message()),
                            est=f"{est * 100.0:.1f}%" if est is not None else "",
                            source="ncu",
                        )
                    )
                except Exception:
                    continue
        profs.append(kp)
    return profs


def _attach_device(kp: KernelProfile) -> None:
    """Record what the profile reports about its own device.

    All device__attribute_* metrics are NVIDIA's own - nothing hardcoded.
    The full set is kept raw in `device_attrs`; the formatted summary is
    built in report.py.
    """
    kp.device_attrs = {k: v for k, v in kp.metrics.items()
                       if k.startswith("device__attribute_")}
    kp.device_attrs.update({k: v for k, v in kp.str_metrics.items()
                            if k.startswith("device__attribute_")})
    name = kp.device_attrs.get("device__attribute_display_name")
    kp.device_name = str(name) if name is not None else None


def dram_pct_of_peak(kp: KernelProfile) -> float | None:
    """NVIDIA's own DRAM utilization vs peak.

    Priority: the pct_of_peak_sustained counter ncu reports directly, else
    the "DRAM Throughput" row of NVIDIA's Speed Of Light section. Never a
    division by a hardcoded spec - the peak comes from the profile itself.
    """
    v = kp.metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    if v is not None:
        return v
    for sec in kp.ncu_sections:
        if sec.sid != "SpeedOfLight":
            continue
        for r in sec.rows:
            if r.label == "DRAM Throughput":
                try:
                    return float(r.value)
                except (TypeError, ValueError):
                    return None
    return None


def stall_total(kp: KernelProfile) -> float | None:
    """Total warp-stall cycles per issued instruction (NVIDIA's metric:
    the sum of its per-reason stall counters)."""
    vals = [kp.metrics.get(b) for b in STALL_BASES]
    vals = [x for x in vals if x is not None]
    return sum(vals) if vals else None


def stall_reasons(kp: KernelProfile) -> list[dict]:
    """Per-reason warp-stall cycles per issued instruction, NVIDIA's own
    stall counters, sorted largest first (feeds the Warp State chart)."""
    out = []
    for base in STALL_BASES:
        v = kp.metrics.get(base)
        if v and v > 0:
            out.append({"name": STALL_SHORT[base], "cycles": round(v, 2)})
    out.sort(key=lambda r: r["cycles"], reverse=True)
    return out


def ingest(path: str | Path, kernel: str | None = None) -> list[KernelProfile]:
    global noise_dropped
    if isinstance(path, (list, tuple)):
        profs: list[KernelProfile] = []
        dropped = 0
        for p in path:
            profs.extend(ingest(p, kernel))
            dropped += noise_dropped
        noise_dropped = dropped
        return profs
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        profs = ingest_json(path, kernel)
    elif suffix == ".csv":
        profs = ingest_csv(path, kernel)
    elif suffix in (".ncu-rep", ".ncu-repz"):
        profs = ingest_report(path, kernel)
        _apply_sec_csvs(path, profs, kernel)
    else:
        raise ValueError(f"unsupported input {path} (want .json, .csv or .ncu-rep)")
    kept: dict[str, KernelProfile] = {}
    for kp in profs:
        kept[kp.name] = kp
    profs = list(kept.values())
    noise_dropped = sum(_is_noise(kp) for kp in profs)
    profs = [kp for kp in profs if not _is_noise(kp)]
    for kp in profs:
        _attach_device(kp)
    return profs
