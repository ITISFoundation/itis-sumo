# itis-sumo

Headless core of the SuMo (**Su**rrogate **Mo**del) meta-modeling tools: build,
evaluate, cross-validate, and interrogate surrogate models — UQ (Sobol indices,
uncertainty propagation, correlation), sampling (LHS / grid / manual-UQ),
and MOGA optimization — as a standalone, importable Python package.

Aggregates the computational core of the mmux/vite flaskapi Dakota modules plus
recycled utilities from earlier MetaModeling trials. No Flask, no oSPARC, no UI:
pure computation on top of the Dakota wheel.

## Engine

- Runtime engine = the **PyPI wheel `itis-dakota==1.5.9`** (Dakota 6.20), pinned for
  exact parity with the mmux/vite flaskapi engine, on **Python 3.10-3.13** (3.10-3.12 currently install; 3.13 is exercised as a compatibility warning until itis-dakota publishes cp313 wheels).
- The in-repo `itis-dakota/` checkout is the IT'IS Dakota fork, kept **for reference
  only** — this package consumes the pre-built wheels published to PyPI and
  **does not build the fork**.
- Engine calls happen through `dakota.environment.study(input_string=...)` executed
  inside a `ProcessPoolExecutor`; the package never shells out to a `dakota` binary.
- Engine caveats, the wheel-build process, and the 6.23+ `Interface::interface_cache`
  regression (why we stay on 1.5.9) are documented in
  [`DAKOTA-STUBS.md`](DAKOTA-STUBS.md), including the T16mo upgrade ladder.

## Provenance

Every module in this package was either ported from an earlier mmux repo or
deliberately written new. The port source is mmux_vite's `flaskapi` (branch
`jgo/sobol-indices` @ `22685f7`); utilities were recycled from
`itis_dakota_projects` / `itis_dakota_projects_clean`; the lineage reaches back to
`mmux_python` (`mmux_gui`). The in-repo `itis-dakota/` fork checkout is reference-only.

**Full per-repo / per-branch / per-commit inventory — what was ported, what was
deliberately not ported, and what remains as porting candidates — is in
[`PORTING.md`](PORTING.md).**

## Install & verify

```sh
uv sync                          # Python 3.10-3.12 currently resolve itis-dakota==1.5.9
uv run itis-sumo validate        # engine probe (expect: Version 1.5.9)
uv run pytest                    # 123 tests, standalone
uv run python examples/headless_smoke.py   # surrogate -> CV -> Sobol
```

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

## Docs

- Published docs: https://itisfoundation.github.io/itis-sumo/
- [`SPEC.md`](SPEC.md) — living spec (goals, constraints, invariants, research, tasks)
- [`PORTING.md`](PORTING.md) — port provenance per source repo/branch/commit
- [`DAKOTA-STUBS.md`](DAKOTA-STUBS.md) — engine stubs, wheel build, regressions
