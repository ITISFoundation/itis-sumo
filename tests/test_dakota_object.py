"""Tests for `mmux_flaskapi.dakota.dakota_object`.

Regression coverage for flaskapi/SPEC.md B16: Dakota execution failures used to
raise Dakota's own generic exception (e.g. "Dakota aborted: Unknown error 254")
without ever reading back the captured stdout/stderr, discarding the real
diagnostic message that explains *why* Dakota aborted.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from itis_sumo.core import dakota_object as dakobj
from itis_sumo.core.dakota_object import (
    DakotaObject,
    _dak_exec_static,
    working_directory,
)


def test_working_directory_changes_and_restores_cwd(tmp_path):
    original_cwd = Path.cwd()
    with working_directory(tmp_path):
        assert Path.cwd() == tmp_path.resolve() or Path.cwd() == tmp_path
    assert Path.cwd() == original_cwd


def test_working_directory_restores_cwd_even_on_exception(tmp_path):
    original_cwd = Path.cwd()
    try:
        with working_directory(tmp_path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert Path.cwd() == original_cwd


def test_dak_exec_static_runs_study_and_captures_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_study = MagicMock()

    def fake_execute():
        os.write(1, b"dakota stdout output\n")

    fake_study.execute.side_effect = fake_execute

    with patch(
        "itis_sumo.core.dakota_object.dakenv.study", return_value=fake_study
    ) as mock_study:
        stdoutstr, stderrstr = _dak_exec_static("some dakota conf")

    mock_study.assert_called_once_with(callback=None, input_string="some dakota conf")
    fake_study.execute.assert_called_once()
    assert stdoutstr is not None
    assert "dakota stdout output" in stdoutstr
    assert stderrstr == ""


def test_dakota_object_run_writes_stdout_and_stderr_files(tmp_path):
    dakobj_instance = DakotaObject()
    with patch.object(
        dakobj_instance,
        "future_exec",
        return_value=("stdout content", "stderr content"),
    ):
        dakobj_instance.run("dummy conf", tmp_path)

    assert (tmp_path / "dakota_stdout.txt").read_text() == "stdout content"
    assert (tmp_path / "dakota_stderr.txt").read_text() == "stderr content"


def test_dakota_object_run_handles_empty_stdout_stderr(tmp_path):
    dakobj_instance = DakotaObject()
    with patch.object(dakobj_instance, "future_exec", return_value=(None, None)):
        dakobj_instance.run("dummy conf", tmp_path)

    assert (tmp_path / "dakota_stdout.txt").read_text() == ""
    assert (tmp_path / "dakota_stderr.txt").read_text() == ""


class _FakeStudySuccess:
    def execute(self):
        # flush=True: Python's own stdout buffering is independent from the
        # libc-level FILE* buffer that wiofiles.capture_to_file flushes on
        # exit, so a fake (pure-Python) study must flush explicitly. Real
        # Dakota output goes through libc stdio and is already covered by
        # that existing flush.
        print("dakota ran fine", flush=True)


class _FakeStudyFailure:
    def execute(self):
        print("Some real Dakota parse error before abort", flush=True)
        raise RuntimeError("Dakota aborted: Unknown error 254")


def _fake_study_constructor_failure(callback, input_string):
    # Simulates Dakota's NIDR input-file parsing failing during study
    # *construction* (e.g. a malformed conf like `seed = 0`), not during
    # study.execute() - flaskapi/SPEC.md B17.
    print("Input line 29: seed must be > 0.", flush=True)
    raise RuntimeError("Dakota aborted: Unknown error 254")


def test_dak_exec_static_success_returns_captured_stdout(monkeypatch, tmp_path, capfd):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dakobj.dakenv, "study", lambda callback, input_string: _FakeStudySuccess()
    )

    # Run via future_exec (ProcessPoolExecutor), matching production's call path.
    # pytest's own fd-level capture must be disabled around this call: it would
    # otherwise dup2 fd 1/2 in the (forked) worker over wiofiles' own redirection.
    with capfd.disabled():
        stdout, stderr = dakobj.DakotaObject().future_exec(conf="dummy conf")

    assert "dakota ran fine" in stdout
    assert stderr == ""


def test_dak_exec_static_failure_raises_with_captured_diagnostics(
    monkeypatch, tmp_path, capfd
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dakobj.dakenv, "study", lambda callback, input_string: _FakeStudyFailure()
    )

    with capfd.disabled(), pytest.raises(RuntimeError) as exc_info:
        dakobj.DakotaObject().future_exec(conf="dummy conf")

    message = str(exc_info.value)
    assert "Some real Dakota parse error before abort" in message
    assert "Dakota aborted: Unknown error 254" in message


def test_dak_exec_static_construction_failure_raises_with_captured_diagnostics(
    monkeypatch, tmp_path, capfd
):
    """Regression (flaskapi/SPEC.md B17): a Dakota abort during `dakenv.study(...)`
    construction (e.g. NIDR rejecting `seed = 0`) must also be captured/enriched,
    not just aborts raised from `study.execute()`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dakobj.dakenv, "study", _fake_study_constructor_failure)

    with capfd.disabled(), pytest.raises(RuntimeError) as exc_info:
        dakobj.DakotaObject().future_exec(conf="dummy conf")

    message = str(exc_info.value)
    assert "Input line 29: seed must be > 0." in message
    assert "Dakota aborted: Unknown error 254" in message
