# About itis-sumo

## What this is

For what itis-sumo is and who it's for, see the [landing page](../index.md). This section covers the **algorithms and implementation**—design, mechanics, and assumptions behind each module.

## Engine

- Runtime engine = the **PyPI wheel `itis-dakota==1.5.9`** (Dakota 6.20),
  pinned for exact parity with the mmux/vite flaskapi engine, on
  **Python 3.10-3.13** (with 3.13 pending itis-dakota cp313 wheels).
- The in-repo `itis-dakota/` checkout is the IT'IS Dakota fork, kept **for
  reference only** — this package consumes the pre-built wheels published to
  PyPI and does not build the fork itself.
- Engine calls happen through `dakota.environment.study(input_string=...)`
  executed inside a `ProcessPoolExecutor`; the package never shells out to a
  `dakota` binary.
- Engine caveats, the wheel-build process, and the Dakota 6.23+
  `Interface::interface_cache` regression (why the pin stays at 1.5.9) are
  documented in [`DAKOTA-STUBS.md`](https://github.com/ITISFoundation/itis-sumo/blob/main/DAKOTA-STUBS.md).

## Package layout

```
src/itis_sumo/
  core/       Dakota execution primitives (DakotaObject, wiofiles)
  config/     NIDR string composers (surrogate/MOGA/UQ confs)
  data/       data processing (healing, bounds, correlation, dominance)
  sampling/   LHS / manual-UQ / grid sampling
  evaluate/   end-to-end runs (sumo eval, CV+metrics, MOGA, Sobol, UQ)
  preprocess/ DataPreprocessor + function-jobs pydantic models
  utils/      engine validation + NIDR config guard
```

## Provenance

Where this standalone package came from, what branches/repos fed it, and
how it relates to the mmux/vite flaskapi it was ported out of:
[Porting notes](porting.md).

## Design spec

The living design document — invariants (`V1`, `V2`, …), open questions,
and rationale behind the module boundaries above: [Spec](spec.md).
