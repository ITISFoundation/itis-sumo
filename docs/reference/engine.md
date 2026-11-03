# Engine internals

Everything below the consumer surface — `itis_sumo.engine`, `itis_sumo.tabular`,
`itis_sumo.surrogate` — is in-package business (V50kv). This page explains how
it works for people who need to understand or debug it; it is not an API
reference, and consumer code must not import these modules.

itis-sumo wraps the [Dakota](https://dakota.sandia.gov/) 6.24 engine (via the
`itis-dakota==6.24.13` PyPI wheel). Every "algorithm" is really a **NIDR
config composer** (build the Dakota input block as a string) paired with a
**run + parse** step (execute the wheel, read back its tabular output). No
Dakota binary is ever shelled out to — everything runs in-process through
`dakota.environment.study`.

## Pipeline shape

<div class="tip-workflow">
  <a class="tip-workflow__step twf-blue" href="functions.md#sampling">Sampling</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-indigo" href="#nidr-string-composers">Config</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-orange" href="#the-dakota-object">Core</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-teal" href="#run--parse">Evaluate</a>
</div>

The consumer-facing layers on top of that pipeline are the `fit` handle
(the run-directory lifetime is the handle's lifetime, V24af) and the one-shot
function modules — both documented on [SuMo handle](su-mo.md) and
[Function modules](functions.md).

## The Dakota object

`itis_sumo.engine.dakota_object.DakotaObject` — the single entry point for
running Dakota. It composes a NIDR input string, executes it, and returns the
run directory containing whatever output files that study type produces
(`dakota_stdout.txt`, `dakota_stderr.txt`, `*.dat` tabular data,
`*.sps`/`*.alg` surrogate archives for the export/import pathway).

- **Execution model** (V1pm): `dakota.environment.study(input_string=...)` is
  called inside a `concurrent.futures.ProcessPoolExecutor` worker (fork on
  POSIX, spawn on Windows, V50gm) — the Python bindings hold a process-global
  problem database, so studies must not share a process. The package
  **never** shells out to a `dakota` binary; Dakota is a library call.
- **Working directory** (V2qw): a `working_directory()` context manager
  `os.chdir`s into the run dir and restores the previous cwd on exit —
  `try`/`finally`, so it restores even if the wheel call raises. The `chdir`
  is confined to the worker process.
- **Run dirs are explicit paths** the caller supplies — never inferred from
  global state (V24af: on the consumer surface the run dir lives and dies
  with the `SuMo` handle).

## NIDR string composers

`itis_sumo.engine.nidr` builds Dakota's NIDR (Named Input, Reduced) input
syntax as plain Python f-strings — no templating engine, no intermediate
object model. This is deliberate (V5ui): the proven, near-verbatim port of the
mmux/vite flaskapi composers, with a seam left for a JSON input layer later
(the Dakota 6.24 experimental JSON-input route is now in use for study input,
see `DAKOTA-STUBS.md` upstream).

## Run + parse

`itis_sumo.engine.evaluation` pairs the composers with execution: stage the
training/samples files, run the study, read back `*.dat` columns, and map
Dakota's internal names (`x1`, `y1_hat`, `y1_std_hat`) back through the
tabular preprocessor to the caller's names and units. UQ/correlation/Sobol
pipelines draw their sample sets in pure Python (`itis_sumo.tabular`) and
evaluate the already-built surrogate on them.

!!! warning "Design invariants"
    Enforced by the test suite (see [`SPEC.md`](../about/spec.md) §V for the
    full, authoritative list):

    - **No global RNG state** — every value producer takes an explicit seed
      (`np.random.Generator`), never numpy/scipy global state (`V3er`).
    - **No web-framework imports anywhere in the package** — zero Flask/oSPARC
      dependencies; the pydantic job models of the web layer were not ported
      (`V4ty`).
    - **Run directories are explicit paths**, and the only `os.chdir` is the
      worker-process-scoped guard above (`V2qw`).
    - **Engine internals stay internal** — consumers import the root manifest,
      `SuMo` methods, and `itis_sumo.report`, nothing else, CI-enforced by
      `tests/test_import_surface.py` (`V50kv`).

## Engine caveats and the wheel

The runtime is the `itis-dakota` wheel line (6.24.x); the 6.23+
`Interface::interface_cache` regression (still upstream) is shimmed in-repo
via `add_r5_interface_cache_workaround()`. Wheel-build process, regressions,
and the reference-only in-repo Dakota fork are documented in
[`DAKOTA-STUBS.md`](https://github.com/ITISFoundation/itis-sumo/blob/main/DAKOTA-STUBS.md)
and the repository README.
