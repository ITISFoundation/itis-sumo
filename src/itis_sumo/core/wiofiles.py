"""
wurlitzer-lite using files instead of pipes

avoids max-pipe-size issue in exchange for dealing with files
"""

import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from typing import IO

_logger = logging.getLogger(__name__)

# copied from wurlitzer
libc = ctypes.CDLL(None)
try:
    c_stdout_p = ctypes.c_void_p.in_dll(libc, "stdout")
    c_stderr_p = ctypes.c_void_p.in_dll(libc, "stderr")
except ValueError:
    # libc.stdout has a funny name on macOS
    c_stdout_p = ctypes.c_void_p.in_dll(libc, "__stdoutp")
    c_stderr_p = ctypes.c_void_p.in_dll(libc, "__stderrp")


@contextmanager
def capture_to_file(stdout="./stdout", stderr="./stderr"):
    stdout_f: IO[bytes] | None = None
    stderr_f: IO[bytes] | None = None
    if stdout:
        # Handles must stay open across the yield: dup2 redirects to them and
        # the finally block restores + closes them.
        stdout_f = open(stdout, mode="wb")  # noqa: SIM115
        if sys.__stdout__ is None:
            raise RuntimeError("sys.__stdout__ is not available")
        real_stdout = sys.__stdout__.fileno()
        save_stdout = os.dup(real_stdout)
        os.dup2(stdout_f.fileno(), real_stdout)
    if stderr:
        stderr_f = open(stderr, mode="wb")  # noqa: SIM115
        if sys.__stderr__ is None:
            raise RuntimeError("sys.__stderr__ is not available")
        real_stderr = sys.__stderr__.fileno()
        save_stderr = os.dup(real_stderr)
        os.dup2(stderr_f.fileno(), real_stderr)
    try:
        yield stdout, stderr
    finally:
        # flush to capture the last of it
        libc.fflush(c_stdout_p)
        libc.fflush(c_stderr_p)

        if stdout_f is not None:
            os.dup2(save_stdout, real_stdout)
            os.close(save_stdout)
            stdout_f.close()
        if stderr_f is not None:
            os.dup2(save_stderr, real_stderr)
            os.close(save_stderr)
            stderr_f.close()


# ======================================================================
# example (manual smoke-test script, not part of the public API)

# calls via PyDLL hold the GIL,
# which blocks when using wurlitzer with pipes
# if the message size exceeds max-pipe-size
pylibc = ctypes.PyDLL(None)


def main():  # pragma: no cover
    sz = 64000
    # while sz < 100_000_000:
    while sz < 100_000:
        _logger.debug("Writing %d bytes", sz)
        buf = b"1" * sz

        with capture_to_file() as (stdout, stderr):
            # execute anything that outputs to stdout, stderr
            pylibc.printf(buf + b"\0")
        # gather from files
        stdoutstr, stderrstr = None, None
        with open(stdout) as outf, open(stderr) as errf:
            stdoutstr = outf.read()
            stderrstr = errf.read()
        # print strings
        _logger.debug("stdout: %s", stdoutstr)
        _logger.debug("stderr: %s", stderrstr)
        with capture_to_file() as (stdout, stderr):
            # execute anything that outputs to stdout, stderr
            pylibc.printf(buf + b"\0")
        # gather from files
        stdoutstr, stderrstr = None, None
        with open(stdout) as outf, open(stderr) as errf:
            stdoutstr = outf.read()
            stderrstr = errf.read()
        # print strings
        _logger.debug("stdout: %s", stdoutstr)
        _logger.debug("stderr: %s", stderrstr)


if __name__ == "__main__":
    main()
