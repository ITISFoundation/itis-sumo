# SPEC — itis-sumo

Caveman-encoded (drop articles/filler; `→` becomes, `!` must, `?` may/uncertain, `⊥` never). Standalone surrogate-modeling library aggregating prior MetaModeling trials.

## LINKS
- engine ! PyPI wheel `itis-dakota==1.5.9` (Dakota 6.20, parity w/ mmux/vite). In-repo `itis-dakota/` = fork checkout, **REFERENCE-ONLY**: pkg consumes pre-built wheels, ⊥ builds fork (6.24 checkout kept for T16mo ladder)
- feeding repos (per-branch/commit inventory + more-to-port → `PORTING.md`):
  - `../mmux_vite` (GiulianaRenc/mmux_vite): v1 port source = branch `jgo/sobol-indices`@`22685f7` → `flaskapi/src/mmux_flaskapi/{dakota,data_preprocessor}/`; E1 inspiration = branch `jgo/sumo-model-export-import`@`d5150b9` → re-implemented in-repo on `feat/sumo-model-export-import` (T12ze-T14qa ✓)
  - `../itis_dakota_projects` (ITISFoundation/itis-a-dakota-project) + `../itis_dakota_projects_clean` (ITISFoundation/itis_dakota_projects): recycled `validate_dakota_installation`/`get_dakota_version` + `validate` CLI
  - `../mmux_python` (ITISFoundation/mmux_gui): ancestor lineage only (⊥ direct port source)
- future consumer → `../mmux_vite/flaskapi` (swaps in-tree core → this pkg)
- research → `DAKOTA-STUBS.md` (stubs + JSON→pydantic POC)

## §G
Standalone pkg `itis-sumo` (SuMo = Surrogate Model): aggregate MetaModeling core from mmux/vite + prior trials (mmux_python, itis_dakota_projects) into one wheel-driven library, usable BOTH inside mmux/vite AND headless (notebook/scripts) → build/evaluate/cross-validate surrogates + UQ (Sobol/correlation) + sampling (LHS/grid) + MOGA. First expansion (E1): SuMo model export/import persistence.

## §C
- engine ! itis-dakota wheel `dakota.environment.study(input_string=...)`; ⊥ subprocess `dakota` bin, ⊥ vendor surfpack; in-repo `itis-dakota/` fork ! REFERENCE-ONLY (⊥ build; runtime = PyPI wheel)
- v1 config ! NIDR string composers (proven); JSON/pydantic seam later (DAKOTA-STUBS POC, Dakota 6.24 experimental)
- port ! near-verbatim from mmux_flaskapi `dakota/`+`data_preprocessor/`; re-root `mmux_flaskapi.`→`itis_sumo.`; prune known-dead (`process_json_file`, `add_interface_s4l`, `create_function_sampling_conffile`, `create_moga_iterative_optimization_conffile`); de-web `JobVariableSelection`/`required_completed_jobs` (move from blueprints into pkg)
- recycle itis_dakota_projects: `validate_dakota_installation()`, `get_dakota_version()`, `validate` CLI, config sanity-guard; ⊥ subprocess `DakotaProject`/factory/generator/parser/activity-logger
- deps ! itis-dakota==1.5.9 (Dakota 6.20, parity w/ mmux/vite; stay until T16mo resolves the 6.23+ `interface_cache` regression — R5), numpy, pandas, scipy; scikit-learn ⊥ KFold only
- py.typed; `src/` layout; docs MkDocs + headless notebook; tests standalone ⊥ flask/osparc
- versioning single-source; uv + pip-compile pins; Conventional Commits
- E1: artifacts keyed server `sumo_model_id` (uuid); metadata sidecar `{id}.metadata.json`; ⊥ user-supplied path/prefix keys (traversal)
- py ! 3.11 (match mmux/vite; 1.5.9 ships no cp313 wheel); 3.13 only via T16mo rung 1 (1.5.11 cp313) or rung 2 (6.24)
- cross-repo call surface ! narrow, versioned, "deep" entrypoints — one stitched public function per feature (e.g. `analyze_dataset()`), ⊥ flaskapi orchestrating several itis-sumo internals itself; version-pinning itis-sumo is fine (same pattern as the itis-dakota engine pin, T16mo) — the coupling risk is a *wide* call surface, not the pin

## §I
- wheel: `dakota.environment.study`, `study.execute()` (⊖ `dakota.surrogates` until wheel bump, R4)
- CLI: `itis-sumo validate`
- python: `itis_sumo.{core,config,data,sampling,evaluate,preprocess,utils}` public funcs (+ forthcoming `analyze_dataset` narrow diagnostics entrypoint, T18ry)
- artifacts: run_dir `{dakota_stdout.txt,dakota_stderr.txt,*.dat,*.sps/.alg}`; models dir `{id}.metadata.json` (E1)

## §V
V1pm: `DakotaObject.run` ! execute wheel in ProcessPoolExecutor; ⊥ subprocess `dakota` bin
V2qw: run dirs ! explicit paths; process-cwd mutation confined to Dakota worker (`os.chdir` guard)
V3er: `create_manual_uq_samples` ! draw ∀ dist via seeded `np.random.Generator` (`random_state=`); ⊥ scipy global state
V4ty: core modules ! zero flask/osparc imports (test-enforced)
V5ui: NIDR config ! pure string composers; seam for JSON/pydantic layer later
V6op: `sanitize_varnames` ! preserve literal `-` in var names
V7as: data processing ! heal-or-drop malformed rows w/ trace; ⊥ silent wipe
V8df: `{output}_std_hat` availability ! from surrogate results post-evaluate_sumo, ⊥ raw job outputs
V9gh: Sobol ! scipy-based (Saltelli A/B/AB + closed-form 2nd-order), validated vs Ishigami analytical
V10jk: E1 export ! `{id}.metadata.json` sidecar w/ verbatim conf block + ordered input descriptors + output descriptor + format + a copy of the real training-data file (`{id}.processed_training.dat`, kept for reference/debuggability per user request — NOT technically required, archive alone is sufficient for reload, R9)
V11lz: E1 import ! conf block identical to export block bar export↔import swap; `import_build_points_file` satisfied by staging back the stored real training-data copy (V10); if that copy is missing (legacy model / deleted out-of-band) ⇒ loud warning log + fall back to a header-only placeholder synthesized from the sidecar descriptors (R9 proves this is safe — Dakota never reads row values back), avoiding a crash
V12xc: E1 keying ! server `sumo_model_id` uuid; ⊥ user-supplied path keys
V13vb: E1 models dir ! single env-overridable source (⊥ branch's dual-strategy split)
V14nm: ⊥ standalone surfpack dep; future standalone-eval → bump itis-dakota (`dakota.surrogates`)
V15zx (proposed, T17bq): `add_surrogate_model` training-file header layout ! explicit `has_eval_id_column: bool` param supplied by caller (whoever wrote/staged the file knows its structure); ⊥ infer from `"processed"` substring in filename — closes B2's root cause (workaround only renamed the staged file to preserve the substring)
V16qf: itis-sumo's cross-repo-facing API (flaskapi/mmux_vite) ! consumed only via §I's documented top-level entrypoints; ⊥ consumer reaches into internal submodules/functions directly — keeps refactors inside itis-sumo from forcing coordinated cross-repo changes; grows w/ each new feature surfaced (starts w/ `analyze_dataset`, T18ry)
V17ab: ∀ `.py` file in repo ! `ruff check` + `ruff format --check` clean (CI-enforced via `prek` on changed files; catches unsorted imports, unformatted code, executable-bit-without-shebang at review time — prevents B3-class regressions)
V18rs: ∀ `.github/workflows/**` change ! at least one validation job executes the affected workflow path; shared CI workflow changes ⊥ leave the validation matrix entirely skipped

## §R
R1: `export_model`/`import_model` child keywords; formats `text_archive`(.sps)/`binary_archive`(.bsps)/`algebraic_file`(.alg); naming `{prefix}.{resp}.{ext}` | branch R2
R2: `import_model` needs conf block identical to export block bar keyword swap; ? whether training VALUES must match — empirical round-trip test | branch R3 — RESOLVED T14qa: values need not match by literal path/identity, only content+structure — a byte-identical copy of the export-time training file at a DIFFERENT path (fresh run_dir, simulated separate session) reproduces export-time predictions exactly (atol 1e-6); real Dakota round-trip, not mocked
R3: .sps archive self-describing vs positional UNVERIFIED `?` → sidecar regardless | branch R8
R4: `dakota.surrogates` ⊖ in wheel 1.5.9; upstream 6.24 adds it — wheel bump over new dep | branch R5-R7
R5: 6.23+ regression — `study()` ctor → `Interface::interface_cache(problem_db)` static map (DakotaInterface.hpp:71), populated only via SimulationModel/NestedModel ctor (`Interface::get_interface`); pure data-fit surrogate confs (`import_build_points_file`, no interface) never populate → `DakotaInterface.cpp:77` `.at()` throws → `IndexError: map::at` before `execute()`. 6.20 ctor used `problem_description_db().interface_list()` (no cache → no crash). Upstream fix candidate = get-or-create (`operator[]`)
R8: resolves R3's `?` — variable-label matching on import happens between the staged points file's header and the Dakota study's OWN declared `variables` block descriptors (our conf, sourced from the metadata sidecar) — NOT by introspecting the `.sps`/`.bsps` archive itself. Verified (`dakota/src/SurfpackApproximation.cpp` in the itis-dakota fork checkout + live Dakota runs): reordering the points-file header (`x2 x1 y1` vs `x1 x2 y1`) still round-trips correctly — Dakota permutes to match by descriptor, per doc `DUPLICATE-import_model` ("matched by descriptor... allows order... to change"); supplying wrong (`foo bar baz`) or incomplete (`x1 y1`, dropping x2) descriptors fails loudly at Dakota's own `import_build_points` check ("not a permutation of expected variable labels") rather than silently mis-mapping columns. Confirms the archive is not introspectable from outside Dakota — the metadata sidecar's ordered descriptors remain the sole source of truth for what to declare (V10's sidecar design was necessary, not just defensive)
R9: `import_build_points_file`'s row VALUES are never read back into the reloaded surrogate — `SurfpackApproximation::import_model()` (`dakota/src/SurfpackApproximation.cpp:837-856`) does `spModel.reset(SurfpackInterface::LoadModel(filename))` straight from the archive; `approxData`/the points file's rows are never touched by that path. Empirically confirmed: importing with a header-only placeholder (0 data rows), 1 row of out-of-range dummy values (999.0 × N), and 2 rows of zeros ALL reproduce export-time predictions exactly (atol 1e-6) — only the header's descriptor SET matters (R8), not row count or values. → real training data is persisted by default anyway, for reference/debuggability (user preference, not a Dakota requirement); `stage_model_for_import` stages that stored copy back, falling back to a synthesized header-only placeholder + loud warning only if it's missing, e.g. a model exported before this persistence was added (T17bc)
R10: format support, per Dakota docs (`dakota/docs/keywords/DUPLICATE-export_model`, `DUPLICATE-import_model`) + source (`SurfpackApproximation::export_model`, `.cpp:467-516`) — only `text_archive`(.sps)/`binary_archive`(.bsps) call `SurfpackInterface::Save()` and are re-loadable via `import_model`; `algebraic_file`(.alg)/`algebraic_console` just dump `spModel->asString()` (one-way; docs state explicitly "not compatible with Dakota or the surfpack executable") — NOT valid for round-trip, Dakota's NIDR parser rejects them as `import_model` children outright (fails loud, not silent). Both `text_archive` and `binary_archive` verified round-trip exactly (atol 1e-6); no documented/observed precision difference — recommend `text_archive` (current default, human-inspectable/debuggable) with `binary_archive` as an opt-in for storage-sensitive cases

## §T
id|status|task|cites
T1pw|✓|scaffold `pyproject.toml` ([project], uv, py.typed, pip-compile pins), `src/` layout, CLI entry|§C
T2cf|✓|port `core/dakota_object.py` + `core/wiofiles.py`|V1,V2
T3qd|✓|port `config/` NIDR composers (incl. existing sumo_export/import params)|V5
T4gs|✓|port `data/` + `sampling/` (lhs, manual-uq, axis/grid, pareto, correlation)|V3,V6,V7
T5as|✓|port `evaluate/` (sumo eval, CV+metrics/ttest/convergence, MOGA, Sobol)|V8,V9
T6hy|✓|port `preprocess/` DataPreprocessor + integration; move pydantic models in; de-web|V4
T7nv|✓|recycle utils: `validate_dakota_installation`, `get_dakota_version`, `validate` CLI, config guard|§C
T8bx|✓|prune dead paths in ported modules|§C
T9kz|✓|port pure-core tests (test_dakota_*, cv-stats, preprocessor, sobol, correlation) + headless smoke|V4
T10le|✓|docs README + MkDocs + headless notebook — README ✓, MkDocs site ✓ (`mkdocs.yml`, algorithm pages + live V&V report w/ real pytest results, served via `mkdocs serve`), headless notebook explicitly deferred past alpha|§G
T11rt|✓|verify: standalone pytest green (⊖ flask), clean-venv install, `itis-sumo validate`, headless surrogate→CV→Sobol|§C
T12ze|✓|E1: `core/sumo_model_store.py` (models dir single env-overridable source `ITIS_SUMO_MODELS_DIR`, uuid keying, metadata sidecar) + `evaluate/funs_evaluate.py::export_sumo_model`/`import_sumo_model` public API|V10,V12,V13
T13uv|✓|E1: capture verbatim conf block for sidecar (close branch placeholder gap)|V10
T14qa|✓|E1: empirical export→import round-trip test (real, unmocked Dakota run — `tests/test_sumo_model_store.py`); resolves R2 `?`|R2,V11
T15mn|.|E1: wire mmux/vite flaskapi endpoints to pkg export/import — separate PR in `../mmux_vite` (jgo/sumo-model-export-import branch T24, itself still `.` there); NOT part of this repo|§I,R1-R4
T16mo|.|stepwise engine modernization: rung 1 `1.5.9→1.5.11` (packaging-only: drop py3.8, add cp313; same 6.20 engine — behavior-identical; re-run smoke); rung 2 `1.5.11→6.24.x` co-shipped w/ JSON input seam (R5 regression fixed upstream first, re-align py 3.13, re-run full smoke)|R4,R5
T17bq|✓|`add_surrogate_model` (`config/funs_create_dakota_conf.py`) ! replaced filename-substring sniffing w/ explicit `has_eval_id_column` param (infer-or-override idiom); all 5 `create_sumo_*_conffile` call sites updated (landed via `feat/vv-port-tests-and-docs`, `a6d694e`, ahead of this merge)|V15,B2
T17bc|✓|E1: persist real training data on export for reference (`{id}.processed_training.dat` sidecar copy, user preference over the leaner archive-only design); `stage_model_for_import` stages it back, falling back to a synthesized header-only placeholder + loud warning log only when that stored copy is missing (legacy model / deleted out-of-band) — fallback verified safe vs Dakota source + fake-points empirical tests (header reorder, wrong/missing descriptors, 0/1/2-row placeholders)|R8,R9,V10,V11
T18ry|.|design+implement `analyze_dataset(df, input_cols, output_cols, alpha=0.05, include_detail=False) -> DatasetDiagnostics` narrow entrypoint (scale/distribution auto-selection + outlier surfacing, plain dataclasses, JSON-serializable via `dataclasses.asdict()`) as the sole flaskapi-facing dataset-diagnostics API, replacing any per-function flaskapi orchestration; BLOCKED — building blocks `select_variable_scale`/`auto_select_distributions` (+ a new raw-value outlier detector, reusing `_tukey_outlier_mask`'s IQR technique) currently exist only on confidential incubator branch `feat/nih-in-silico-example`, not `develop` — needs individual promotion first, same promotion rule as T15mn/E1|§C,V16qf,I
T19kp|.|headless notebook, post-alpha — add runnable notebook counterpart to docs getting-started flow once alpha docs publish is stable|T10le
T20hm|.|CI workflow changes ! activate shared validation jobs and regression-check detector classification|V18rs

## §B
id|date|cause|fix
B1|2026-08-07|itis-dakota 6.23+ `Interface::interface_cache()` throws `IndexError: map::at` on interface-less surrogate confs (`study()` ctor, before `execute()`) — pure data-fit surrogates never instantiate an interface → static map missing entry (R5)|pin `==1.5.9` (Dakota 6.20, no cache) for flaskapi parity; ladder T16mo; upstream get-or-create fix
B2|2026-08-07|E1 import round-trip: `add_surrogate_model` (`config/funs_create_dakota_conf.py`) decided whether the training file has a leading `eval_id` column purely from whether the substring `"processed"` appears in the training filename, not an explicit flag — staging the re-import training file as `{id}.training.dat` (T12ze) desynced Dakota's `use_variable_labels` column expectation from the actual CSV, surfacing as `Cannot reorder variables ... not a permutation of expected variable labels`|immediate: `sumo_model_store.py::_training_file_name` names both the stored and staged file `{id}.processed_training.dat`, preserving the substring; root-cause: `add_surrogate_model` heuristic replaced by explicit `has_eval_id_column` param, T17bq/V15zx (now ✓)
B3|2026-08-18|CI `prek` job fails on PRs that touch unformatted `.py` files — ruff I001 (unsorted imports) + EXE002 (executable bit, no shebang) + ruff format violations in `funs_evaluate.py`, `funs_create_dakota_conf.py`, `test_property_invariants.py`, `generate_docs_figures.py`, `test_metamodeling_analytical.py`, `test_unit_solver.py` — pre-commit config dropped but ruff formatting not enforced repo-wide first|fix: `ruff check --fix` + `ruff format` all touched files + `chmod -x` executable bits; preventive: §V17ab invariant — ∀ `.py` file must pass `ruff check` + `ruff format --check`
B4kt|2026-08-18|CI workflow-only changes were absent from `detect_changes`, so all downstream jobs could be skipped and an invalid `astral-sh/setup-uv@v10` reference remained unresolved behind an apparently green PR|classify `.github/workflows/**` as CI changes, run shared validation matrix, replace nonexistent action tag; V18rs
