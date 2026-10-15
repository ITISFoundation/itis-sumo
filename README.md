# itis-sumo

Headless core of the SuMo (**Su**rrogate **Mo**del) meta-modeling tools: build,
evaluate, cross-validate, and interrogate surrogate models — UQ (Sobol indices,
uncertainty propagation, correlation), sampling (LHS / grid / manual-UQ),
and MOGA optimization — as a standalone, importable Python package.

Aggregates the computational core of the mmux/vite flaskapi Dakota modules plus
recycled utilities from earlier MetaModeling trials. No Flask, no oSPARC, no UI:
pure computation on top of the Dakota wheel.

## Engine

- Runtime engine = the **PyPI wheel `itis-dakota==6.24.13`** (Dakota 6.24), on
  **Python 3.12-3.14**, Linux, macOS **and Windows** (first `itis-dakota` release with
  `win_amd64` wheels; engine parity with mmux/vite is now a process, not a version — the
  consumer may anchor on 1.5.11, see SPEC §C py).
- The in-repo `itis-dakota/` checkout is the IT'IS Dakota fork, kept **for reference
  only** — this package consumes the pre-built wheels published to PyPI and
  **does not build the fork**.
- Engine calls happen through `dakota.environment.study(input_string=...)` executed
  inside a `ProcessPoolExecutor` with an explicit start-method context (fork on POSIX,
  spawn on Windows — SPEC V50gm); the package never shells out to a `dakota` binary.
- Engine caveats, the wheel-build process, and the 6.23+ `Interface::interface_cache`
  regression (still upstream; shimmed in-repo via `add_r5_interface_cache_workaround()`)
  are documented in [`DAKOTA-STUBS.md`](DAKOTA-STUBS.md).

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
uv run pytest                    # 123 tests, standalone
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
