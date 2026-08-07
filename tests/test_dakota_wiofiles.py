"""Tests for mmux_flaskapi.dakota.wiofiles (fd-level stdout/stderr capture)."""

import os

from itis_sumo.core.wiofiles import capture_to_file


def test_capture_to_file_captures_real_fd_stdout_and_stderr(tmp_path):
    stdout_path = tmp_path / "stdout"
    stderr_path = tmp_path / "stderr"

    with capture_to_file(stdout=str(stdout_path), stderr=str(stderr_path)) as (stdout, stderr):
        assert stdout == str(stdout_path)
        assert stderr == str(stderr_path)
        # write directly to the real OS-level stdout/stderr fds (bypasses sys.stdout,
        # simulating what a C-extension like Dakota's pybind bindings would do)
        os.write(1, b"hello from stdout\n")
        os.write(2, b"hello from stderr\n")

    assert stdout_path.read_text() == "hello from stdout\n"
    assert stderr_path.read_text() == "hello from stderr\n"


def test_capture_to_file_restores_original_fds_after_exit(tmp_path):
    stdout_path = tmp_path / "stdout"
    stderr_path = tmp_path / "stderr"

    with capture_to_file(stdout=str(stdout_path), stderr=str(stderr_path)):
        os.write(1, b"captured\n")

    # after the context manager exits, fd 1 must no longer point at the file
    os.write(1, b"not captured to file\n")
    assert stdout_path.read_text() == "captured\n"


def test_capture_to_file_stdout_only(tmp_path):
    stdout_path = tmp_path / "stdout"

    with capture_to_file(stdout=str(stdout_path), stderr=None) as (stdout, stderr):
        assert stdout == str(stdout_path)
        assert stderr is None
        os.write(1, b"only stdout\n")

    assert stdout_path.read_text() == "only stdout\n"
