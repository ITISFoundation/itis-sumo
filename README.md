# itis-sumo

Headless core of the SuMo (**Su**rrogate **Mo**del) meta-modeling tools: build,
evaluate, cross-validate, and interrogate surrogate models — UQ (Sobol indices,
uncertainty propagation, correlation), sampling (LHS / grid / manual-UQ),
and MOGA optimization — as a standalone, importable Python package.

Aggregates the computational core of the mmux/vite flaskapi Dakota modules plus
recycled utilities from earlier MetaModeling trials. No Flask, no oSPARC, no UI:
pure computation on top of the Dakota wheel.

```python
from itis_sumo import fit
from itis_sumo.report import render_cross_validation  # needs itis-sumo[report]

model = fit(samples, ["width", "height"], "stress")   # -> SuMo handle
cv = model.cross_validate(folds=5, seed=42)           # typed result, your units
render_cross_validation(cv)                           # one result in, one figure out
```

## Engine

Runtime engine = the **PyPI wheel `itis-dakota==6.24.13`** (Dakota 6.24) on
Python 3.12-3.14, Linux, macOS and Windows — executed in-process through
`dakota.environment.study`, never a shelled-out binary. Engine caveats, the
wheel-build process, and the known upstream regressions (and their in-repo
shims) are documented in
[`DAKOTA-STUBS.md`](DAKOTA-STUBS.md) and the published docs
([Engine internals](https://itisfoundation.github.io/itis-sumo/reference/engine/)).
The in-repo `itis-dakota/` checkout is the IT'IS Dakota fork, kept **for
reference only** — this package consumes the pre-built wheels and does not
build the fork.

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
uv sync                          # Python 3.12-3.14; resolves itis-dakota==6.24.13
uv run itis-sumo validate        # engine probe (expect: Version 6.24.13)
uv run pytest                    # full suite, standalone
uv run python examples/headless_smoke.py   # surrogate -> CV -> Sobol
```

## Development

### Code quality

`ruff` (lint + format) and `ty` (type checking) run both at commit time and in
CI. Install the local git hook once:

```sh
uvx prek install
```

Or run all hooks on demand without installing them:

```sh
make prek
```

### Version channels and releases

Package versions are single-sourced in `pyproject.toml`. Use these channels:

- Feature branches: manual `.devN` development versions.
- `develop`: alpha versions (`aN`).
- `main`: beta versions (`bN`) until the project graduates to `1.0.0`.

PR CI checks that a proposed version matches its target branch and is newer than
existing git tags under PEP 440 ordering. After a non-docs merge to `develop` or
`main`, CI creates the matching `v<version>` tag without committing back to the
protected branch. Docs-only changes do not create tags.

CI builds and tests alpha, beta, and release-candidate versions. Real PyPI
publishing and GitHub Releases remain manual: use the `Publish` workflow's
`workflow_dispatch` input with the existing tag after validation. Feature-branch
`.devN` versions use the fast local TestPyPI target below; they do not need tags
or CI.

### Manual TestPyPI publishing

Create a local, gitignored `.env` file containing:

```text
TESTPYPI_TOKEN=pypi-your-testpypi-token
```


For a feature-branch development release, commit your code changes and run:

```sh
make publish-testpypi-dev
```

This temporarily writes the next `.devN` version for the build, checks the package, uploads directly to TestPyPI using `TESTPYPI_TOKEN` from the local `.env`, and restores `pyproject.toml` afterward, including when the command fails. It refuses a dirty worktree. This path intentionally does not create commits or tags and does not run CI, so broken development versions can be published quickly. Alpha, beta, and release-candidate versions remain gated by CI before real-PyPI publication.

## Package layout

```
src/itis_sumo/
  __init__.py      root manifest — THE consumer surface (V50kv)
  surrogate/       SuMo handle: fit/load, run dir = handle lifetime (V24af)
  validation.py sensitivity.py uncertainty.py exploration.py
  sampling.py optimization.py diagnostics.py calibration.py
  convergence.py               ≤2 public functions per module (§I)
  types.py errors.py           typed results (original units) + typed failures
  report/          renderings: figures.py + text.py + sections() (V54hz);
                   optional extra `itis-sumo[report]` (plotly, marimo)
  engine/          Dakota execution, NIDR composers, study runners — internal
  tabular/         preprocessor + pure data maths — internal
```

## Docs

- Published docs: https://itisfoundation.github.io/itis-sumo/
- [`SPEC.md`](SPEC.md) — living spec (goals, constraints, invariants, research, tasks)
- [`PORTING.md`](PORTING.md) — port provenance per source repo/branch/commit
- [`DAKOTA-STUBS.md`](DAKOTA-STUBS.md) — engine stubs, wheel build, regressions
