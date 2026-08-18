# PORTING — where itis-sumo came from

Every line of `itis-sumo` is either ported from an earlier repo in the mmux
ecosystem or deliberately new. This document records, per source repository,
per branch, **what was ported, what was not, and at which commit** — so future
porting decisions can be made from facts instead of memory.

All commit hashes/dates were verified against the local checkouts on
**2026-08-07**. The source-of-truth scope decisions live in `SPEC.md` §C.

## Source repository map

| repo (local path) | remote | role |
|---|---|---|
| `../mmux_vite` | https://github.com/GiulianaRenc/mmux_vite.git | primary port source (flaskapi) |
| `../itis_dakota_projects` | https://github.com/ITISFoundation/itis-a-dakota-project.git | recycled utils |
| `../itis_dakota_projects_clean` | https://github.com/ITISFoundation/itis_dakota_projects | recycled utils (CLI/helpers) |
| `../mmux_python` | https://github.com/ITISFoundation/mmux_gui | earliest lineage (ancestor only) |
| `itis-dakota/` (in-repo) | https://github.com/ITISFoundation/itis-dakota (fork) | **engine reference only — NOT built here** |

---

## 1. mmux_vite / flaskapi — primary port source

- **Ported from branch `jgo/sobol-indices` @ `22685f7` (2026-08-05, "fix: address
  Copilot-flagged bugs on PR #505").** Verified: every ported file matches this
  branch modulo import re-rooting (`mmux_flaskapi.` → `itis_sumo.`) and the
  deliberate prunes listed below. Other branches carry strict subsets
  (`jgo/sumo-validation-stats`, `jgo/uq-uncertainty-propagation`) or the same
  content minus Sobol/CV additions.
- Local checkout currently sits on `grenc-plots` @ `0367fb6` (2026-08-07) — a
  newer branch; the port predates it.

### Ported modules (file-level mapping)

| mmux_vite source | itis-sumo destination | fidelity |
|---|---|---|
| `flaskapi/src/mmux_flaskapi/dakota/dakota_object.py` | `core/dakota_object.py` | near-verbatim (import re-root) |
| `flaskapi/src/mmux_flaskapi/dakota/wiofiles.py` | `core/wiofiles.py` | near-verbatim |
| `flaskapi/src/mmux_flaskapi/dakota/funs_create_dakota_conf.py` | `config/funs_create_dakota_conf.py` | pruned (see below) |
| `flaskapi/src/mmux_flaskapi/dakota/funs_data_processing.py` | `data/funs_data_processing.py` | pruned: `process_json_file` dropped; `_parse_json_dict` kept (still used by the json-loading branch) |
| `flaskapi/src/mmux_flaskapi/dakota/funs_evaluate.py` | `evaluate/funs_evaluate.py` | near-verbatim (1001/1002 lines) |
| `flaskapi/src/mmux_flaskapi/dakota/lhs.py` | `sampling/lhs.py` | byte-identical |
| `flaskapi/src/mmux_flaskapi/data_preprocessor/data_preprocessor.py` | `preprocess/data_preprocessor.py` | near-verbatim |
| `flaskapi/src/mmux_flaskapi/data_preprocessor/data_preprocessor_integration.py` | `preprocess/data_preprocessor_integration.py` | near-verbatim |
| `flaskapi/src/mmux_flaskapi/blueprints/dakota_models.py` | `preprocess/models.py` | **de-webbed** (function-jobs part only) |

`preprocess/models.py` is the pydantic function-jobs part of `dakota_models.py`
(`FunctionJob`, `JobVariableSelection`, `required_completed_jobs`); the web API
request models stay in mmux/vite.

### Pruned from the conf module (`config/funs_create_dakota_conf.py`)
Deliberately dropped composers (dead / web- or S4L-specific / superseded):
`add_interface_s4l`, `add_python_interface`, `add_iterative_sumo_optimization`,
`create_function_sampling_conffile`, `create_export_sumo_conffile`,
`create_moga_iterative_optimization_conffile`. The live MOGA path
(`create_moga_optimization_conffile`, `add_moga_method`,
`perform_moga_optimization`) **is** ported.

### NOT ported (web / ops scope — out of scope for a headless package)
- The Flask web layer: `app.py`, all `blueprints/` routes, `utils/json_serializer.py`,
  `utils/local_job_store.py`, `utils/logger.py`, `utils/webserver_config.py`.

### E1 export/import — IMPLEMENTED and MERGED into `main`
- **Where:** originally developed on branch `feat/sumo-model-export-import`
  (worktree `.claude/worktrees/feat+sumo-model-export-import`, based on main
  `955ec99`) as `cad02c5` ("feat: add SuMo model export/import (E1)"). That
  commit reached `main` via PR #1 (2026-08-18) and is now part of the released
  baseline — the earlier "not merged into main" note is obsolete.
- **What:** `src/itis_sumo/core/sumo_model_store.py` (148 lines, V12 uuid keying /
  V13 single `ITIS_SUMO_MODELS_DIR` env source / V10 metadata sidecar),
  `evaluate/funs_evaluate.py::export_sumo_model` / `import_sumo_model`,
  `config/funs_create_dakota_conf.py` `export_import_format` pass-through,
  and `tests/test_sumo_model_store.py` (8 tests, **real unmocked Dakota
  round-trip** — resolves R2 `?`).
- **Verified:** full suite in the worktree = **131 passed** (123 + 8 E1). Also
  surfaced and fixed a real bug on the way: §B B2 (the `"processed"`-substring
  heuristic in `add_surrogate_model` vs the staged re-import training filename).
- **Upstream counterpart:** the mmux/vite branch `jgo/sumo-model-export-import`
  @ `d5150b9` inspired the feature; the pkg implementation re-roots it into
  `itis_sumo` (its web endpoints stay in mmux/vite).
- **Still open:** T15mn — wiring mmux/vite flaskapi endpoints to the pkg
  export/import — is a **separate PR in mmux_vite**, not part of this repo.

### Other mmux_vite branches (NOT ported)
| branch | HEAD | content / status |
|---|---|---|
| `main` | `6c69488` (2026-07-06) | baseline flaskapi (subset of sobol-indices) |
| `develop` | `8d6af25` (2026-08-04) | integration branch (subset) |
| `local-develop` | `7221c84` (2026-07-08) | local-dev variant (subset) |
| `grenc-plots` | `0367fb6` (2026-08-07) | SuMo publication plotting tool — NOT ported |
| `feature/sumo-publication-plot` (remote) | — | plotting tool (sibling of grenc-plots) — NOT ported |
| `jgo/sumo-model-export-import` | `d5150b9` (2026-07-09) | E1 export/import — **re-implemented in-repo on `feat/sumo-model-export-import`** (see above); endpoints stay in mmux/vite (T15mn) |
| `jgo/sobol-indices` | `22685f7` (2026-08-05) | **PORTED (this is the source)** |
| `jgo/sumo-validation-stats` | `39f86ed` (2026-07-04) | CV stats — already inside sobol-indices / ported |
| `jgo/uq-uncertainty-propagation` | `15cafc4` (2026-07-09) | UQ propagation — already inside ported code |
| `jgo/osparc-backend-resilience` | `75ae50b` (2026-08-04) | oSPARC graceful-degradation (web) — NOT ported |
| `jgo/download_uq_samples` | `a3b8739` (2026-07-08) | UQ sample download endpoint (web) — NOT ported |
| `jgo/fullstack-logscale` | `4aebe7b` (2026-08-06) | log-scale inputs/outputs (backend+frontend) — NOT ported |
| `jgo/stack-04-csv-download` | `ea69222` (2026-07-08) | CSV download endpoint (web) — NOT ported |
| `jgo/docs-migration` | `fe4ca60` (2026-07-13) | docs migration — NOT ported |
| `DO-NOT-MERGE-feature/local-functions` | `b8cf7b3` (2026-07-08) | local-functions experiment — NOT ported |

### More to port from mmux_vite?
- **E1 export/import → already implemented** in-repo on `feat/sumo-model-export-import`
  (see above); only the mmux/vite endpoint wiring (T15mn) remains, and that lives
  in mmux_vite. A future wheel bump unlocks `dakota.surrogates` per §R4.
- Publication plotting tool (`grenc-plots`) and log-scale transforms — candidates
  if the headless package is to cover plotting/UQ-viz; not currently planned.
- CSV/UQ download endpoints — web concerns; expect them to stay in mmux/vite.

---

## 2. itis_dakota_projects — recycled utils

Remote: https://github.com/ITISFoundation/itis-a-dakota-project.git
- `main` @ `cf17cda` (2025-08-20); `feature/populate-basic-repository-scripts` @
  `400e0dc` (2025-09-02, current checkout).

**Ported:** the `validate_dakota_installation()` / `get_dakota_version()` helpers
(the itis-sumo `utils/helpers.py` is closest to the sibling repo's version — see
§3). **Not ported (deliberately ⊥ per SPEC §C):** `DakotaProject` class, `factory`,
`dakota_file_generator`, `validator`, activity-logger (subprocess-based), the
example folders (`CedricDakWheelExamples/SboExample`, `CalmpcExample`), and the
CI/report scripts. **More to port:** nothing — it is example/CI infrastructure.

---

## 3. itis_dakota_projects_clean — recycled CLI + helpers

Remote: https://github.com/ITISFoundation/itis_dakota_projects
- `main` @ `c30ac9a` (2025-08-20); `feature/populate_repo_clean` @ `bb4523a`
  (2025-09-04, current checkout).

**Ported:**
- `utils/helpers.py` → `itis_sumo/utils/helpers.py` (`validate_dakota_installation`,
  `get_dakota_version`, `create_run_dir`).
- `utils/cli.py` → `itis_sumo/cli.py` (the `itis-sumo validate` command; extended
  with `--config` for the NIDR sanity-guard).

**New in itis-sumo (not recycled):** `utils/config_guard.py`
(`validate_nidr_config` — a cheap structural NIDR check, authored here).
**Not ported:** the rest of this minimal repo. **More to port:** none.

---

## 4. mmux_python — earliest lineage (ancestor only)

Remote: https://github.com/ITISFoundation/mmux_gui
- Current checkout: `work/jgo/spinal_pw_rowald_pulse_optimization` @ `cdd3e5b`
  (2025-09-15). Other branches: `work/jgo/flask_mmux_nih`, `nih_visualization`,
  `optistim_pulse_optimization`, `spinal_pw_*`, `spinal_rowald_pulse_optimization`,
  `work/jgo/clean-up`, remote `work/jgo/GridSampling`.

**Role:** the earliest Dakota-wheel wrappers live here
(`utils/dakota_object.py`, `utils/funs_create_dakota_conf.py`,
`utils/funs_evaluate.py` — the latter only 81 lines), plus 2025 experiments in
`runs/dakota_*`. Its lineage flows into mmux_vite/flaskapi, and therefore into
itis-sumo. **Ported:** nothing directly (no files copied from here).
**Not ported:** spinal/optistim pulse-optimization apps, GAF kernels, nih
visualization, GridSampling, flask apps. **More to port:** none (superseded by
the mmux_vite core that was ported).

---

## 5. itis-dakota fork — engine, REFERENCE ONLY

Local `itis-dakota/` is a **nested git checkout** of the IT'IS Dakota fork
(current HEAD `d24ec38`, 2026-05-04), kept **for reference only**.

**This repository does NOT build itis-dakota.** The runtime engine is the
pre-built wheel from PyPI: `itis-dakota==1.5.9` (Dakota 6.20, parity with
mmux/vite flaskapi). The in-repo checkout exists so that:
- the wheel's surface (`dakota.environment.study`, `dakenv`, stubs, `dakota.json`)
  can be inspected without leaving the repo;
- the 6.23/6.24 sources needed for the T16mo modernization ladder are at hand
  (incl. `src_patches_v6xx/`).

See `DAKOTA-STUBS.md` for the wheel build process, the 6.23+
`Interface::interface_cache` regression, and the upgrade ladder. The fork's
uncommitted local mods (cmake, surfpack, `src/dakota_python.cpp`) are the fork's
own working-tree state and are unrelated to this package.

---

## Status accuracy note

SPEC §T reflects the **actual** port state (main SPEC now matches the E1 branch's
SPEC). E1 T12ze/T13uv/T14qa = ✓ on branch `feat/sumo-model-export-import`
(implementation + real round-trip test exist in that worktree); T15mn = `.` —
explicitly a separate mmux/vite PR, not this repo. The branch's E1 changes are
**committed as `cad02c5` on top of `955ec99`**, but that commit is NOT yet in
`main` — nothing E1-related is in `main`'s history yet.
