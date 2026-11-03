"""
wurlitzer-lite using files instead of pipes

avoids max-pipe-size issue in exchange for dealing with files
"""

import ctypes
import os
import sys
from contextlib import contextmanager
from typing import IO

# libc bindings (wurlitzer pattern). POSIX: CDLL(None) = the main program's
# already-loaded libc, which exports the `stdout`/`stderr` FILE* symbols.
# win32: CDLL(None) raises (LoadLibrary requires a str) — bind the UCRT
# (`ucrtbase.dll`), the CRT the interpreter AND the dakota extension share,
# which exports `fflush` and `__acrt_iob_func(0/1/2)` -> the process FILE*s.
# Same FILE* state the extension's printf/C-writes flush through, so the
# dup2 capture below (fd-level, works on both) is flanked by real flushes.
if sys.platform == "win32":
    libc = ctypes.CDLL("ucrtbase.dll")
    _iob = getattr(libc, "__acrt_iob_func")
    _iob.restype = ctypes.c_void_p
    _iob.argtypes = [ctypes.c_int]
    c_stdout_p = ctypes.c_void_p(_iob(1))
    c_stderr_p = ctypes.c_void_p(_iob(2))
else:
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
