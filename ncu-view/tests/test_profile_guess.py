"""_guess_run_cmd: directory of .cu files, single-driver python dirs,
and cutlass include flags."""
from __future__ import annotations

import pytest

from ncu_view.profile import _cutlass_include, _guess_run_cmd


def test_cu_dir_compiles_with_cutlass_includes(monkeypatch, tmp_path):
    (tmp_path / "mat.cu").write_text("__global__ void k() {}\n")
    monkeypatch.setattr("ncu_view.profile._cutlass_include", lambda: " -I/opt/cutlass/include")
    cmd = _guess_run_cmd(tmp_path)
    assert cmd.startswith(
        f"nvcc -arch=native -O3 -I/opt/cutlass/include -lcublas -o /tmp/{tmp_path.name}-run")
    assert "mat.cu" in cmd and "&& /tmp/" in cmd


def test_single_cu_file_gets_include_flags(monkeypatch, tmp_path):
    f = tmp_path / "one.cu"
    f.write_text("__global__ void k() {}\nint main() { return 0; }\n")
    monkeypatch.setattr("ncu_view.profile._cutlass_include", lambda: " -I/opt/cutlass/include")
    cmd = _guess_run_cmd(f)
    assert f"-o /tmp/one-run {f.name}" in cmd
    assert "-I/opt/cutlass/include" in cmd


def test_cu_without_main_picks_sibling_driver(tmp_path):
    (tmp_path / "kern.cu").write_text("__global__ void k() {}\n")
    (tmp_path / "driver.cu").write_text("int main() { return 0; }\n")
    cmd = _guess_run_cmd(tmp_path / "kern.cu")
    assert "kern.cu driver.cu" in cmd and "-o /tmp/kern-run" in cmd


def test_cu_without_main_or_driver_errors(tmp_path):
    (tmp_path / "orphan.cu").write_text("__global__ void k() {}\n")
    with pytest.raises(SystemExit, match="--build-cmd"):
        _guess_run_cmd(tmp_path / "orphan.cu")


def test_run_py_wins(monkeypatch, tmp_path):
    (tmp_path / "run.py").write_text("")
    (tmp_path / "mat.cu").write_text("")
    cmd = _guess_run_cmd(tmp_path)
    assert cmd == "python3 run.py"


def test_single_python_driver_dir(tmp_path):
    (tmp_path / "matmul_v1.py").write_text("")
    cmd = _guess_run_cmd(tmp_path)
    assert cmd == "python3 matmul_v1.py"


def test_ambiguous_dir_errors(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    with pytest.raises(SystemExit, match="--build-cmd"):
        _guess_run_cmd(tmp_path)


def test_cutlass_include_flag_missing_package():
    assert _cutlass_include() == "" or _cutlass_include().startswith(" -I")
