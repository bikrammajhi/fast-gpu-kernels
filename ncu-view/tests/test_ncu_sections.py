"""Section-CSV overlay: NVIDIA's own detailed tables replace one-liners."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncu_view.ingest import (  # noqa: E402
    _apply_sec_csvs,
    _parse_sec_csv,
    _sec_csv_files,
    ingest,
)

FIXTURE = '''"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name"
"0","kernel_matmul","Scheduler Statistics","Cycles Active","cycle","1,012,345",""
"0","kernel_matmul","Scheduler Statistics","Cycles Per Issued Instruction","","1.00",""
"0","kernel_matmul","Scheduler Statistics","Issue Slots Busy","","0.90",""
"0","kernel_matmul","Scheduler Statistics","Scheduler Issue Slot Utilization","%","90.0",""
"0","kernel_matmul","Scheduler Statistics","","","",""
'''

FIXTURE_SIMPLE = '''"section","Scheduler Statistics"
"Metric Name","Metric Value","Metric Unit"
"Cycles Active","1,012,345","cycle"
"Scheduler Issue Slot Utilization","90.0","%"
'''


def test_parse_sec_csv_rows():
    rows = _parse_sec_csv(FIXTURE, "SchedulerStats")
    labels = [r.label for r in rows]
    assert labels == ["Cycles Active", "Cycles Per Issued Instruction",
                      "Issue Slots Busy", "Scheduler Issue Slot Utilization"]
    assert rows[0].value == "1.01234e+06"
    assert rows[3].unit == "%"
    assert rows[3].bar == 90.0
    assert all(r.bar is None for r in rows[:3])


def test_parse_sec_csv_simple_fallback():
    rows = _parse_sec_csv(FIXTURE_SIMPLE, "SchedulerStats")
    assert [r.label for r in rows] == [
        "Cycles Active", "Scheduler Issue Slot Utilization"]
    assert rows[1].bar == 90.0


def test_parse_sec_csv_empty_and_junk():
    assert _parse_sec_csv("", "X") == []
    assert _parse_sec_csv('"section","Nothing"\n"Metric Name","V"\n', "X") == []
    assert _parse_sec_csv('"label","not a number","cycle"\n', "X") == []


def test_sec_csv_files_siblings(tmp_path):
    rep = tmp_path / "a.ncu-rep"
    rep.write_text("x")
    (tmp_path / "a.sec-SchedulerStats.csv").write_text("y")
    (tmp_path / "a.sec-WarpStateStats.csv").write_text("z")
    (tmp_path / "other.sec-X.csv").write_text("q")
    found = _sec_csv_files(rep)
    assert [sid for sid, _ in found] == ["SchedulerStats", "WarpStateStats"]


def test_apply_sec_csvs_replaces_by_sid(tmp_path):
    kp = _fake_profile()
    kp.ncu_sections = [_fake_ncu_section("SchedulerStats")]
    (tmp_path / "a.sec-SchedulerStats.csv").write_text(FIXTURE)
    _apply_sec_csvs(tmp_path / "a.ncu-rep", [kp])
    assert len(kp.ncu_sections) == 1
    sec = kp.ncu_sections[0]
    assert sec.detailed is True
    assert sec.src == "NVIDIA"
    assert [r.label for r in sec.rows] == [
        "Cycles Active", "Cycles Per Issued Instruction",
        "Issue Slots Busy", "Scheduler Issue Slot Utilization"]


def test_apply_sec_csvs_appends_unknown_sid(tmp_path):
    kp = _fake_profile()
    kp.ncu_sections = []
    (tmp_path / "a.sec-SourceCounters.csv").write_text(FIXTURE)
    _apply_sec_csvs(tmp_path / "a.ncu-rep", [kp])
    assert [s.sid for s in kp.ncu_sections] == ["SourceCounters"]
    assert kp.ncu_sections[0].detailed is True


def test_apply_sec_csvs_ignores_empty(tmp_path):
    kp = _fake_profile()
    kp.ncu_sections = [_fake_ncu_section("SchedulerStats")]
    (tmp_path / "a.sec-SchedulerStats.csv").write_text('"section","x"\n')
    _apply_sec_csvs(tmp_path / "a.ncu-rep", [kp])
    assert kp.ncu_sections[0].detailed is False


def test_ingest_overlays_sec_csvs():
    pytest.importorskip("ncu_report")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rep = td / "k.ncu-rep"
        shutil.copy(_GOLDEN_REP, rep)
        (td / "k.sec-SchedulerStats.csv").write_text(FIXTURE)
        profs = ingest(rep)
        secs = {s.sid: s for s in profs[0].ncu_sections}
        assert secs["SchedulerStats"].detailed is True
        assert secs["SchedulerStats"].rows[0].label == "Cycles Active"
        assert secs["WarpStateStats"].detailed is False


def test_ingest_unchanged_without_sec_csvs():
    pytest.importorskip("ncu_report")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rep = td / "k.ncu-rep"
        shutil.copy(_GOLDEN_REP, rep)
        profs = ingest(rep)
        secs = {s.sid: s for s in profs[0].ncu_sections}
        assert secs["SchedulerStats"].detailed is False
        assert secs["SchedulerStats"].rows[0].derived is True  # one-liner rule


def _fake_profile():
    from ncu_view.model import KernelProfile
    return KernelProfile(key="matmul_v1", name="v1", metrics={},
                         provenance={"source": "ncu-rep", "path": "test"})


def _fake_ncu_section(sid):
    from ncu_view.ingest import _ncu_section
    return _ncu_section(sid, rows=_fake_one_line())


def _fake_one_line():
    from ncu_view.model import Row
    return [Row("Rule [warning]", "one sentence", derived=True)]


ROOT = Path(__file__).resolve().parent.parent.parent
_GOLDEN_REP = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "golden" / "matmul_v1.ncu-rep"
