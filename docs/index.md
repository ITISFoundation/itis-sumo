# itis-sumo

Headless core of the SuMo (**Su**rrogate **Mo**del) meta-modeling tools: build,
evaluate, cross-validate, and interrogate surrogate models — UQ (Sobol indices,
uncertainty propagation, correlation), sampling (LHS / grid / manual-UQ), and
MOGA optimization — as a standalone, importable Python package.

This site documents the **algorithms and implementation** — what each module
does, how it works, and what assumptions it makes. For how to *use* SuMo
through the MMUX web UI, see the separate `mmux_documentation` site; this
package is the headless computational core underneath that UI (and usable
directly, without it).

## What this is

Aggregates the computational core of the mmux/vite flaskapi Dakota modules
plus recycled utilities from earlier MetaModeling trials. No Flask, no
oSPARC, no UI dependency: pure computation on top of the [itis-dakota
wheel](https://pypi.org/project/itis-dakota/).

## Engine

- Runtime engine = the **PyPI wheel `itis-dakota==1.5.9`** (Dakota 6.20),
  pinned for exact parity with the mmux/vite flaskapi engine, on
  **Python 3.11**.
- The in-repo `itis-dakota/` checkout is the IT'IS Dakota fork, kept **for
  reference only** — this package consumes the pre-built wheels published to
  PyPI and does not build the fork itself.
- Engine calls happen through `dakota.environment.study(input_string=...)`
  executed inside a `ProcessPoolExecutor`; the package never shells out to a
  `dakota` binary.
- Engine caveats, the wheel-build process, and the Dakota 6.23+
  `Interface::interface_cache` regression (why the pin stays at 1.5.9) are
  documented in [`DAKOTA-STUBS.md`](https://github.com/ITISFoundation/itis-sumo/blob/main/DAKOTA-STUBS.md).

## Install & verify

```sh
uv sync                          # Python 3.11, resolves itis-dakota==1.5.9
uv run itis-sumo validate        # engine probe (expect: Version 1.5.9)
uv run pytest                    # standalone test suite
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

New to surrogate modeling or Gaussian Processes? Start with
[Why surrogate modeling](theory/surrogate-modeling.md) and
[How Gaussian Processes work](theory/gaussian-processes.md), then see them in
action in [Worked examples](theory/examples.md). Already familiar with the
theory? Start with the [Algorithms overview](algorithms/index.md), or jump
straight to the [Verification & Validation report](verification-validation.md)
for evidence that the pipeline reproduces known analytical solutions.
