"""Terminal printouts for autonomous agents, printed after every run.

  1. an nvidia-smi-style device panel — real device facts from the profile,
  2. the top optimization signals — NVIDIA's own rule engine, sorted by
     NVIDIA's estimated speedup (the banner verdict is the #1 signal),
  3. a machine-readable JSON tail an agent loop can parse directly.

Values are NVIDIA's own (device attributes and rule results from the
capture) — nothing invented, nothing hardcoded to any kernel.
"""

from __future__ import annotations

import json
import re

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .report import _est_val

_console = Console(width=100)

SEV_STYLE = {"critical": "bold red", "warning": "bold yellow",
             "suggestion": "bold cyan", "info": "bold blue"}
SEV_TAG = {"critical": "CRIT", "warning": "WARN", "suggestion": "SUGG",
           "info": "INFO"}
SEV_RANK = {"critical": 0, "warning": 1, "suggestion": 2, "info": 3, "hint": 4}

TOP_SIGNALS = 5
GREEN = "bold green"
DIM = "dim"


def _dv(dev: dict, key: str) -> str:
    """A device attribute {v, unit} -> '178.35 GiB' (missing -> '—')."""
    e = dev.get(key)
    if not e or not isinstance(e, dict) or e.get("v") is None:
        return "—"
    val = float(e["v"])
    s = f"{int(val)}" if val == int(val) else f"{val:.2f}".rstrip("0").rstrip(".")
    unit = e.get("unit") or ""
    return f"{s} {unit}".strip()


def _si(x: float | None, unit: str = "") -> str:
    """Humanized count: 4350000 -> '4.35 M cyc'; None -> '—'."""
    if x is None:
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.2f} G {unit}".strip()
    if a >= 1e6:
        return f"{x / 1e6:.2f} M {unit}".strip()
    if a >= 1e3:
        return f"{x / 1e3:.2f} K {unit}".strip()
    return f"{x:.2f} {unit}".strip()


def _time_us(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x / 1e3:.2f} ms" if x >= 1000 else f"{x:.1f} µs"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}%"


def _kshort(n: str) -> str:
    """Shorten a mangled kernel name: first 14 chars + '…' + last 3
    underscore segments (mirrors the HTML report's kshort)."""
    n = n or "kernel"
    if len(n) <= 40:
        return n
    tail = n.split("_")[-3:]
    return n[:14] + "…" + "_".join(tail)


# --------------------------------------------------------------------------
# 1. nvidia-smi-style device panel
# --------------------------------------------------------------------------

def print_device(dev: dict, console: Console | None = None) -> None:
    c = console or _console
    if not dev:
        c.print("GPU: unknown — this input carries no device attributes.",
                style=DIM)
        c.print()
        return
    name = dev.get("name") or "unknown"
    idx = dev.get("device_index")
    gpu = f"{idx}  {name}" if idx is not None else name
    facts = []
    if dev.get("pci"):
        facts.append(f"Bus-Id {dev['pci']}")
    if dev.get("ecc"):
        facts.append(f"ECC {dev['ecc']}")
    if dev.get("cc"):
        facts.append(f"CC {dev['cc']}")
    sm = _dv(dev, "sm_count")
    if sm != "—":
        facts.append(f"SM {sm}")
    clk = _dv(dev, "clock_mhz")
    mclk = _dv(dev, "mem_clock_mhz")
    bus = _dv(dev, "mem_bus_bits")
    table = Table(show_header=False, box=box.SQUARE_DOUBLE_HEAD,
                  border_style="green", expand=True,
                  title="[bold green]NVIDIA-SMI[/]",
                  title_style="", title_justify="left",
                  caption=f"captured via ncu · ncu-view {__version__}",
                  caption_style="dim", caption_justify="right")
    table.add_column(style="bold green", justify="right", no_wrap=True)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_row("GPU  Name", gpu)
    table.add_row("Bus · ECC · CC · SM", "  ·  ".join(facts) or "—")
    table.add_row("Memory · L2",
                  f"{_dv(dev, 'fb_gib')} · {_dv(dev, 'l2_mib')}")
    table.add_row("Clocks",
                  f"SM {clk} · Mem {mclk}"
                  + (f" · Bus {bus}" if bus != "—" else ""))
    c.print(table)
    c.print()


# --------------------------------------------------------------------------
# 2. top optimization signals (NVIDIA rule engine)
# --------------------------------------------------------------------------

def _rule_sort_key(r: dict) -> tuple:
    return (-_est_val(r), SEV_RANK.get(r.get("severity"), 9))


def _focus_lines(r: dict) -> list[str]:
    info = r.get("focus_info") or {}
    out = []
    for name, value in (r.get("focus") or {}).items():
        vs = _si(value) if isinstance(value, (int, float)) and abs(value) >= 1e3 \
            else f"{value:g}" if isinstance(value, (int, float)) else str(value)
        out.append(f"[dim]{name}:[/] {vs}")
        if info.get(name):
            out.append(f"[dim]  {info[name]}[/]")
    return out


def _clean_msg(m: str) -> str:
    """NVIDIA rule text uses @url:Label:url@ / @section:Id:Label@ markup —
    keep only the display label."""
    m = re.sub(r"@url:([^@]*?):[^@]*@", r"\1", m)
    m = re.sub(r"@section:[^@]*?:([^@]*)@", r"\1", m)
    return m


def _rule_block(r: dict) -> str:
    lines = [f"[bold]{r.get('name', '')}[/]"]
    lines.append(f"[dim]{_clean_msg(r.get('message', ''))}[/]")
    lines.extend(_focus_lines(r))
    return "\n".join(lines)


def print_signals(report: dict, console: Console | None = None) -> None:
    c = console or _console
    kernels = report.get("kernels") or []
    if not kernels:
        return
    for k in kernels:
        _print_kernel_signals(k, c)
    if len(kernels) > 1:
        _print_overall(report, c)
    c.print("machine-readable signals (agent-parsable JSON):", style=DIM)
    print(json.dumps(_signals_json(report), indent=1))
    print()


def _print_kernel_signals(k: dict, c: Console) -> None:
    s = k.get("stats") or {}
    rules = sorted(k.get("rules") or [], key=_rule_sort_key)[:TOP_SIGNALS]
    body = [Text(f"{_kshort(k.get('name', 'kernel'))}", style="bold cyan"),
            Text(
                f"{_time_us(s.get('time_us'))} · pipe {_pct(s.get('pipe_pct'))} · "
                f"dram {_pct(s.get('dram_pct'))} · occupancy {_pct(s.get('occupancy_pct'))} · "
                f"stall {_si(s.get('stall_cycles'), 'cyc')} · "
                f"SM {_si(s.get('clock_ghz'), 'GHz')}", style=DIM)]
    if rules:
        table = Table(show_header=False, box=None, pad_edge=False, expand=True)
        table.add_column(style="bold", justify="right", no_wrap=True)
        table.add_column(style="bold", justify="right", no_wrap=True)
        table.add_column()
        table.add_column(style=GREEN, justify="right", no_wrap=True)
        for i, r in enumerate(rules, 1):
            sev = r.get("severity") or "info"
            est = f"est. {r['est']}" if r.get("est") else "est. —"
            table.add_row(str(i), f"[{SEV_STYLE.get(sev, 'bold')}]"
                          f"{SEV_TAG.get(sev, 'INFO')}[/]",
                          _rule_block(r), est)
        body.append(Text(""))
        body.append(table)
    else:
        body.append(Text("no NVIDIA rule results for this input.", style=DIM))
    c.print(Panel(Group(*body),
                  title="[bold cyan]TOP OPTIMIZATION SIGNALS[/]",
                  border_style="cyan", padding=(0, 1)))
    c.print()


def _print_overall(report: dict, c: Console) -> None:
    kernels = report.get("kernels") or []
    dom = max(kernels, key=lambda k: (k.get("stats") or {}).get("time_us") or 0)
    v = dom.get("verdict")
    s = dom.get("stats") or {}
    text = Text()
    text.append(f"{_kshort(dom.get('name', ''))} · {_time_us(s.get('time_us'))} → ",
                style="bold cyan")
    if v:
        text.append(v.get("name", "no signal"), style="bold green")
        if v.get("est"):
            text.append(f" · est. {v['est']}", style=GREEN)
    else:
        text.append("no signal", style=DIM)
    c.print(Panel(text, title="[bold yellow]OVERALL TOP SIGNAL[/]",
                  border_style="yellow", padding=(0, 1)))
    c.print()


# --------------------------------------------------------------------------
# 3. machine-readable tail (for autonomous agent loops)
# --------------------------------------------------------------------------

def _signals_json(report: dict) -> dict:
    out = {"device": (report.get("meta") or {}).get("device") or {},
           "kernels": []}
    for k in report.get("kernels") or []:
        s = k.get("stats") or {}
        signals = sorted(k.get("rules") or [], key=_rule_sort_key)[:TOP_SIGNALS]
        out["kernels"].append({
            "name": k.get("name"),
            "duration_us": s.get("time_us"),
            "stats": {key: s.get(key) for key in
                      ("pipe_pct", "dram_pct", "occupancy_pct",
                       "stall_cycles", "clock_ghz")},
            "signals": [{
                "rid": r.get("rid"),
                "name": r.get("name"),
                "severity": r.get("severity"),
                "est": r.get("est"),
                "est_value": None if _est_val(r) < 0 else round(_est_val(r), 3),
                "message": r.get("message"),
                "focus": r.get("focus"),
            } for r in signals],
        })
    return out
