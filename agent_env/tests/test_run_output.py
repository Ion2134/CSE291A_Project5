from __future__ import annotations

from pathlib import Path

from discovery_sim.sim.run_output import RUN_MARKER, resolve_run_dir


def test_resolve_run_dir_uses_requested_when_missing(tmp_path: Path):
    out = tmp_path / "run-a"
    assert resolve_run_dir(out, seed=1, overwrite=False) == out


def test_resolve_run_dir_unique_when_marker_present(tmp_path: Path):
    out = tmp_path / "run-a"
    out.mkdir()
    (out / RUN_MARKER).write_text("{}\n")
    resolved = resolve_run_dir(out, seed=42, overwrite=False)
    assert resolved != out
    assert resolved.name.startswith("run-a-")
    assert resolved.name.endswith("-seed42")


def test_resolve_run_dir_overwrite(tmp_path: Path):
    out = tmp_path / "run-a"
    out.mkdir()
    (out / RUN_MARKER).write_text("{}\n")
    assert resolve_run_dir(out, seed=42, overwrite=True) == out
