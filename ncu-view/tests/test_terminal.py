"""Terminal printout smoke tests: the agent-facing panels must render and
the JSON tail must be parseable with the expected top signal first."""

import io
import json
import sys
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from ncu_view.terminal import _clean_msg, print_device, print_signals  # noqa: E402

DEV = {
    "name": "NVIDIA B200", "cc": "10.0", "device_index": 0, "pci": "5b:00.0",
    "ecc": "on", "sm_count": {"v": 148.0, "unit": ""},
    "fb_gib": {"v": 178.35, "unit": "GiB"},
    "l2_mib": {"v": 126.5, "unit": "MiB"},
    "clock_mhz": {"v": 1965.0, "unit": "MHz"},
    "mem_clock_mhz": {"v": 3996.0, "unit": "MHz"},
    "mem_bus_bits": {"v": 7680.0, "unit": "bit"},
}

KERNEL = {
    "name": "matmul_v1", "verdict": {
        "rid": "UncoalescedGlobalAccess", "name": "Uncoalesced Global Accesses",
        "severity": "warning", "est": "86.3x",
        "message": "uncoalesced global accesses"},
    "rules": [
        {"rid": "UncoalescedGlobalAccess", "name": "Uncoalesced Global Accesses",
         "severity": "warning", "est": "86.3x", "message": "uncoalesced",
         "focus": {"derived__memory_l2_theoretical_sectors_global_excessive": 62914560.0},
         "focus_info": {"derived__memory_l2_theoretical_sectors_global_excessive":
                        "Reduce the number of excessive wavefronts in L2"}},
        {"rid": "SOLBottleneck", "name": "Bottleneck",
         "severity": "warning", "est": "", "message": "TC is the bottleneck"},
        {"rid": "CPIStall", "name": "Warp Stall", "severity": "info",
         "est": "5700.0%", "message": "long scoreboard stalls",
         "focus": {"smsp_average_long_scoreboard": 105.259},
         "focus_info": {"smsp_average_long_scoreboard": "Decrease stalls"}},
    ],
    "stats": {"time_us": 2423.84, "pipe_pct": 20.7, "dram_pct": 11.79,
              "occupancy_pct": 6.26, "stall_cycles": 4350000.0,
              "clock_ghz": 1.83},
}

REPORT = {"meta": {"device": DEV}, "kernels": [KERNEL]}


def _capture(fn) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    with unittest.mock.patch("sys.stdout", buf):
        fn(console)
    return buf.getvalue()


def _strip(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def test_device_panel():
    out = _strip(_capture(lambda c: print_device(DEV, c)))
    assert "NVIDIA-SMI" in out and "NVIDIA B200" in out
    assert "178.35 GiB" in out and "1965 MHz" in out and "7680 bit" in out
    assert "┌" in out and "└" in out


def test_device_panel_unknown():
    out = _strip(_capture(lambda c: print_device({}, c)))
    assert "unknown" in out


def test_clean_msg():
    m = ("See @section:WarpStateStats:Warp State Statistics@ and "
         "@url:CUDA Programming Guide:https://docs.nvidia.com/@ for details.")
    out = _clean_msg(m)
    assert "Warp State Statistics" in out
    assert "CUDA Programming Guide" in out
    assert "@url" not in out and "@section" not in out and "https://" not in out


def test_signals_order_and_content():
    out = _strip(_capture(lambda c: print_signals(REPORT, c)))
    assert "TOP OPTIMIZATION SIGNALS" in out
    assert "WARN" in out
    assert "est. 86.3x" in out
    i1 = out.index("Uncoalesced Global Accesses")
    i2 = out.index("Bottleneck")
    assert i1 < i2
    assert "2.42 ms" in out and "pipe 20.7%" in out and "4.35 M cyc" in out


def test_signals_short_name():
    k = dict(KERNEL, name="kernel_cutlass_kernel_TiledMMA_ThrLayoutVMNK11110000"
                         "_PermutationMNK____MMAAtom_ThrID10_ShapeMNK12825616")
    out = _strip(_capture(lambda c: print_signals({"meta": REPORT["meta"],
                                                   "kernels": [k]}, c)))
    assert "…" in out


def test_signals_json_tail():
    out = _capture(lambda c: print_signals(REPORT, c))
    payload = out[out.index("machine-readable signals"):]
    data = json.loads(payload[payload.index("{"):])
    sig = data["kernels"][0]["signals"][0]
    assert sig["rid"] == "UncoalescedGlobalAccess"
    assert sig["est_value"] == 86.3
    assert data["kernels"][0]["stats"]["stall_cycles"] == 4350000.0
    assert data["device"]["name"] == "NVIDIA B200"


def test_signals_pct_est_ranking():
    k = dict(KERNEL)
    out = _capture(lambda c: print_signals({"meta": REPORT["meta"],
                                            "kernels": [k]}, c))
    data = json.loads(out[out.index("{"):])
    order = [s["rid"] for s in data["kernels"][0]["signals"]]
    assert order[0] == "UncoalescedGlobalAccess"
    assert order[1] == "CPIStall"  # 5700.0% -> 57.0x, beats 'no est' rule
