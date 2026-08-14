"""Data model for ncu-view: kernels, sections and rule results.

The model is deliberately plain data: every renderer (HTML report,
terminal summary, live server) consumes the same objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gpus import Device


@dataclass
class Row:
    """One row of a section. `bar` is an optional 0-100 utilization bar."""

    label: str
    value: str
    unit: str = ""
    bar: float | None = None
    derived: bool = False
    note: str = ""


@dataclass
class Section:
    sid: str
    title: str
    description: str = ""
    rows: list[Row] = field(default_factory=list)
    table: list[list[str]] | None = None
    src: str = ""  # "NVIDIA" when this section came from the .ncu-rep itself
    detailed: bool = False  # NVIDIA's own full table (--import --section), not a rule one-liner


@dataclass
class RuleResult:
    """A rule verdict. severity: critical | warning | suggestion | info."""

    rid: str
    name: str
    severity: str
    message: str
    kernel: str | None = None
    est: str = ""
    source: str = "our"  # "our" (documented heuristic) or "ncu" (NVIDIA's own)
    focus: dict[str, float] = field(default_factory=dict)
    focus_info: dict[str, str] = field(default_factory=dict)  # per-focus evidence, e.g. "22.8 < 80.0"


@dataclass
class KernelProfile:
    """One profiled kernel, normalized to a flat counter dict."""

    key: str
    name: str
    metrics: dict[str, float]
    provenance: dict[str, str]
    # String-typed metrics (e.g. device__attribute_display_name) that do not
    # fit the float dict above.
    str_metrics: dict[str, str] = field(default_factory=dict)
    # The device the profile ran on, resolved from the profile itself.
    device: Device | None = None
    # Phase 2: when the input is an .ncu-rep, NVIDIA's own section rows and
    # rule results ride along untouched, rendered with their source label.
    ncu_sections: list[Section] = field(default_factory=list)
    ncu_rules: list[RuleResult] = field(default_factory=list)
