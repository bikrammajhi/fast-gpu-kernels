"""CLI smoke tests: subcommand dispatch, output files, device override."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncu_view.cli import main

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root (ncu-view/tests -> fast-gpu-kernels)
GOLDEN_DIR = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "golden"
GOLDEN = GOLDEN_DIR / "matmul_v1.ncu-rep"
BLOG_JSON = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "ncu_counters.json"


def _input():
    # The .ncu-rep needs NVIDIA's reader; fall back to the JSON probe
    # without it (and without the golden rep at all).
    if GOLDEN.exists() and importlib.util.find_spec("ncu_report"):
        return GOLDEN
    return BLOG_JSON


def test_version():
    try:
        main(["--version"])
        return False
    except SystemExit as e:
        assert e.code == 0


def test_missing_input_errors():
    try:
        main(["report"])
        return False
    except SystemExit as e:
        assert e.code != 0


def test_report_writes_html_and_json(tmp_path):
    rc = main(["report", str(_input()), "-o", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / f"{_input().stem}.html").exists()
    assert (tmp_path / f"{_input().stem}.json").exists()


def test_default_command_is_report(tmp_path):
    rc = main(["--path", str(_input()), "-o", str(tmp_path)])
    assert rc == 0
    assert list(tmp_path.glob("*.html"))


def test_summary_exits_zero():
    assert main(["summary", str(_input())]) == 0


def test_gpu_override(tmp_path):
    rc = main(["report", str(_input()), "-o", str(tmp_path), "--gpu", "H100 SXM"])
    assert rc == 0
    payload = (tmp_path / f"{_input().stem}.json").read_text()
    assert '"name": "H100 SXM"' in payload


def test_kernel_regex_filters():
    rc = main(["summary", str(BLOG_JSON), "--kernel-regex", "v[0-9]+"])
    assert rc == 0


def test_bad_gpu_keeps_working(tmp_path):
    rc = main(["report", str(_input()), "-o", str(tmp_path),
               "--gpu", "NVIDIA Quantum X1"])
    assert rc == 0
    payload = (tmp_path / f"{_input().stem}.json").read_text()
    assert "not in catalog" in payload
