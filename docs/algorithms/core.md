# Dakota engine core

`itis_sumo.core` — the thin layer that actually talks to the Dakota wheel.

## `DakotaObject` (`core/dakota_object.py`)

The single entry point for running Dakota. Composes a NIDR input string
(from `itis_sumo.config`), executes it, and returns the run directory
containing whatever output files that study type produces
(`dakota_stdout.txt`, `dakota_stderr.txt`, `*.dat` tabular data,
`*.sps`/`*.alg` surrogate archives for the export/import pathway).

- **Execution model** (`V1pm`): `dakota.environment.study(input_string=...)`
  is called inside a `concurrent.futures.ProcessPoolExecutor` worker, via
  the module-level `_dak_exec_static` function (must stay module-level to
  stay picklable for the executor). The package **never** shells out to a
  `dakota` binary — Dakota is a library call, not a subprocess.
- **Working directory** (`V2qw`): `working_directory()` is a
  context-manager that `os.chdir`s into the run dir and restores the
  previous cwd on exit — `try`/`finally`, so it restores even if the
  wheel call raises. This `chdir` is confined to the worker process; it
  never leaks into the caller's process because the executor runs it in a
  separate OS process, not just a separate thread.
- **Run dirs are explicit paths** the caller supplies — `DakotaObject`
  never infers or defaults a run directory from global state.

## `wiofiles` (`core/wiofiles.py`)

A "wurlitzer-lite" stdout/stderr capture utility, using files instead of
OS pipes. Dakota's C++ layer writes directly to the process's `stdout`/
`stderr` file descriptors (bypassing Python's `sys.stdout`), so capturing
its output for logging requires redirecting at the file-descriptor level —
`capture_to_file()` does that via `ctypes` access to libc's `stdout`/
`stderr` pointers. Pipes have a fixed OS buffer size and can deadlock or
truncate on Dakota's larger verbose-output runs; files don't have that
ceiling, which is why this exists instead of the original `wurlitzer`
pipe-based approach.
