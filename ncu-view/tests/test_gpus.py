"""GPU detection unit tests: catalog match, fallback, measured-attr override."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncu_view.gpus import DEVICES, FALLBACK, cfg_for_device, detect_device


def test_name_match_b200():
    dev = detect_device(name="NVIDIA B200")
    assert dev.matched
    assert dev.spec["tensor_peak"] == 2250.0
    assert dev.spec["dram_peak"] == 8e12


def test_name_match_h100():
    dev = detect_device(name="NVIDIA H100")
    assert dev.matched
    assert dev.spec["tensor_peak"] == 989.0
    assert dev.spec["dram_peak"] == 3.35e12


def test_unknown_name_falls_back():
    dev = detect_device(name="NVIDIA Quantum X1")
    assert not dev.matched
    assert dev.note and "not in catalog" in dev.note
    assert dev.spec["tensor_peak"] == FALLBACK["tensor_peak"]


def test_no_info_keeps_defaults():
    dev = detect_device(name=None, attributes={})
    assert dev.name == "unknown"
    assert not dev.spec.get("tensor_peak")  # caller falls back to defaults
    assert dev.note


def test_measured_attrs_win_over_catalog():
    dev = detect_device(name="NVIDIA B200",
                        attributes={"device__attribute_max_warps_per_multiprocessor": 128,
                                    "device__attribute_max_shared_memory_per_multiprocessor": 1024.0})
    assert dev.spec["max_warps_per_sm"] == 128
    assert dev.spec["max_smem_per_sm"] == 1024.0
    assert dev.spec["tensor_peak"] == 2250.0  # catalog still fills the peak


def test_smsp_from_attrs():
    dev = detect_device(name="NVIDIA B200",
                        attributes={"device__attribute_max_warps_per_scheduler": 16})
    assert dev.spec["smsp_per_sm"] == 4


def test_cfg_for_device_merges_without_none():
    dev = detect_device(name="NVIDIA B200")
    cfg = cfg_for_device(dev)
    assert cfg["M"] == 8192  # defaults survive
    assert cfg["tensor_peak"] == 2250.0


def test_user_cfg_wins():
    dev = detect_device(name="NVIDIA B200")
    cfg = cfg_for_device(dev, {"tensor_peak": 1800.0, "M": 4096})
    assert cfg["tensor_peak"] == 1800.0
    assert cfg["M"] == 4096


def test_every_catalog_entry_is_complete():
    required = {"tensor_peak", "dram_peak", "max_warps_per_sm",
                "smsp_per_sm", "max_smem_per_sm", "nvcc_arch"}
    for name, spec in DEVICES.items():
        missing = required - set(spec)
        assert not missing, f"{name} missing {missing}"
        assert spec["tensor_peak"] > 0 and spec["dram_peak"] > 0
