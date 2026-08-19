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

## VOCAB
Canonical nouns for ∀ public API, docs, tests, commit messages. One word per concept, ⊥ synonyms drifting back in.
- **sample** → one row of tabular data (an observation). ⊥ "job", ⊥ "point", ⊥ "record"
- **variable** = **parameter** → one INPUT column
- **response** = **quantity-of-interest (QoI)** → one OUTPUT column
- variables vs responses ! stay distinguished — they get different downstream treatment (normalization, sign, sampling, Sobol roles)
- **domain** → where we may draw/explore a variable (bounds + scale). Property of the design space, auto-inferable from existing samples
- **distribution** → real-world shape of a variable (a claim about the world). ⊥ inferable from a training design
- **surrogate** = **SuMo** → the trained metamodel
- oSPARC's `FunctionJob` is ⊥ itis-sumo vocabulary; it lives on the consumer side of the boundary

## §G
Standalone pkg `itis-sumo` (SuMo = Surrogate Model): aggregate MetaModeling core from mmux/vite + prior trials (mmux_python, itis_dakota_projects) into one wheel-driven library, usable BOTH inside mmux/vite AND headless (notebook/scripts) → build/evaluate/cross-validate surrogates + UQ (Sobol/correlation) + sampling (LHS/grid) + MOGA. First expansion (E1): SuMo model export/import persistence.

Consumer contract (grill 2026-08-18): itis-sumo owns ALL surrogate machinery end-to-end — training-file creation, preprocessing, Dakota config, run dirs, inverse transforms. Consumers (flaskapi OR a direct python user) pass ONLY tabular samples + configuration, and receive typed results in original units.

## §C
- engine ! itis-dakota wheel `dakota.environment.study(input_string=...)`; ⊥ subprocess `dakota` bin, ⊥ vendor surfpack; in-repo `itis-dakota/` fork ! REFERENCE-ONLY (⊥ build; runtime = PyPI wheel)
- v1 config ! NIDR string composers (proven); JSON/pydantic seam later (DAKOTA-STUBS POC, Dakota 6.24 experimental)
- port ! near-verbatim from mmux_flaskapi `dakota/`+`data_preprocessor/`; re-root `mmux_flaskapi.`→`itis_sumo.`; prune known-dead (`process_json_file`, `add_interface_s4l`, `create_function_sampling_conffile`, `create_moga_iterative_optimization_conffile`); de-web `JobVariableSelection`/`required_completed_jobs` (moved from blueprints into pkg) — SUPERSEDED in part by the tabular-only boundary below: `required_completed_jobs`' Dakota rule stays, the job-shaped models go back to the consumer (T24cm)
- recycle itis_dakota_projects: `validate_dakota_installation()`, `get_dakota_version()`, `validate` CLI, config sanity-guard; ⊥ subprocess `DakotaProject`/factory/generator/parser/activity-logger
- deps ! itis-dakota==1.5.9 (Dakota 6.20, parity w/ mmux/vite; stay until T16mo resolves the 6.23+ `interface_cache` regression — R5), numpy, pandas, scipy; scikit-learn ⊥ KFold only
- py.typed; `src/` layout; docs MkDocs + headless notebook; tests standalone ⊥ flask/osparc
- versioning single-source in `[project].version`; uv/uv_build + pip-compile pins; Conventional Commits
- release channel policy: feature branches → `.devN` (manual alpha/dev prereleases); `develop` → `aN`; `main` → `bN` until `1.0.0` graduation; version bump ! enforced by a PR-time CI check (target-branch suffix, PEP 440 order, ⊥ reuse/regression); ⊥ bot commit on `develop`/`main` (both branch-protected, no bypass granted); merge-time job ! push `v<version>` tag only (tags fall outside 'require PR' branch protection); `develop` may later rename `staging`
- real PyPI publish + GitHub Release ! manual `workflow_dispatch` only (for now); CI builds/tests alpha/beta/rc versions; feature `.devN` versions publish locally via `make publish-testpypi-dev` + `.env` token, ⊥ CI/TestPyPI tag cascade
- license ! itis-sumo currently private/pending quality review; adopt itis-dakota's license (IT'IS Foundation - All Rights Reserved) until an explicit public/OSS decision is made — MIT retracted as premature given itis-dakota (a required dependency) is itself proprietary
- TestPyPI publish token ! local `.env` (gitignored) `TESTPYPI_TOKEN`; Make injects `-t TESTPYPI_TOKEN`; ⊥ committed, ⊥ required as a pre-exported shell var
- feature-branch TestPyPI local path ! `make publish-testpypi-dev` computes/writes next `.devN`, builds/checks/uploads via local `.env` token; ⊥ commit/tag/CI publication
- E1: artifacts keyed server `sumo_model_id` (uuid); metadata sidecar `{id}.metadata.json`; ⊥ user-supplied path/prefix keys (traversal)
- py ! 3.11 (match mmux/vite; 1.5.9 ships no cp313 wheel); 3.13 only via T16mo rung 1 (1.5.11 cp313) or rung 2 (6.24)
- cross-repo call surface ! narrow, versioned, "deep" entrypoints — one stitched public function per feature (e.g. `analyze_dataset()`), ⊥ flaskapi orchestrating several itis-sumo internals itself; version-pinning itis-sumo is fine (same pattern as the itis-dakota engine pin, T16mo) — the coupling risk is a *wide* call surface, not the pin
- data in ! plain tabular (`pd.DataFrame` or an itis-sumo dataclass trivially convertible to one); ⊥ `FunctionJob`/oSPARC job concepts anywhere in itis_sumo. Ownership split: consumer converts jobs→table AND filters un-completed samples (warning); itis-sumo rejects insufficient/invalid data (raises; consumer catches + surfaces). Dakota sufficiency rule `max(5, n_variables + 1)` stays in itis-sumo
- preprocessing ! auto-defaulted (novice/normal users pass nothing); advanced override allowed but expressed in DOMAIN vocabulary only (`scale`, `direction`); ⊥ transform vocabulary in any public signature (`sign_switch`, `normalization="zscore"`, `mapped_name`, `_hat`/`_std_hat`). Effective config → INSPECTABLE in results, ⊥ SETTABLE in transform terms. MOGA `direction` defaults `"minimize"` (optimization-field convention)
- `domain` ⊥ conflated w/ `distribution` (see VOCAB): domain auto-inferable from samples (observed bounds + detected scale) → Sobol box + MOGA search space; distribution ! stated by the modeller → UQ propagation + correlation. Retires the `mean ± 3σ` fallback in `_bounds_from_distributions` (a conflation artifact — its own docstring admits the two uses differ). Lands in the handle transformation (T27fr), ⊥ the port
- architecture ! handle-primary is the TARGET (`fit(samples, config)` → `model.along_axes/grid/sobol/propagate_uq/...`, `model.save() -> sumo_model_id`); the mmux/vite port ships the current ONE-SHOT shape (no frontend change). One-shot entrypoints ! implemented internally as fit-then-query from day one → handle extraction later = re-export, ⊥ second port
- run artifacts ! ephemeral on success (⊥ unbounded growth in a web service), PRESERVED on failure w/ `run_dir` + captured stderr tail attached to the raised error (B1/B2/`Unknown error 250` are all Dakota-opacity bugs — the run dir is the only tractable evidence); `workspace=` opt-in for deliberate debugging. E1 model store (`ITIS_SUMO_MODELS_DIR`) = separate, always-persistent concern
- reproducibility ! fixed default seed `42`, overridable; effective seed returned in the result; ⊥ require caller to supply one
- errors ! stable taxonomy `SumoError` → `SumoInputError`(→400) / `SumoResultError`(→422, predictions missing) / `SumoEngineError`(→500, carries run_dir + stderr tail); ⊥ consumer classifying by string/regex match (retires the MOGA `re.search` branch)
- results ! typed dataclasses in ORIGINAL units + ORIGINAL names, JSON-serializable via `dataclasses.asdict()` (same convention as `DatasetDiagnostics`); ⊥ magic suffix keys in the public shape
- `?` should **distribution** get an auto-generated default the way domain does? User: "probably a good idea but should be explicitly discussed, not a given" → POST-PORT decision (T28gs), ⊥ guess
- `?` `get_sumo_cv_accuracy_metrics` is the ONE workflow bypassing DataPreprocessor entirely (raw var names + `process_input_file`) — normalize it (changes numbers ⇒ compat decision) or preserve + document?
- `?` public name/shape of the tabular dataclass vs raw `pd.DataFrame`

## §I
- wheel: `dakota.environment.study`, `study.execute()` (⊖ `dakota.surrogates` until wheel bump, R4)
- CLI: `itis-sumo validate`
- python (consumer-facing) → `itis_sumo.api` = THE module flaskapi/mmux_vite imports; nothing else. Tabular in, typed results out, taxonomy errors (T22ax)
- python (in-package/headless) → `itis_sumo.{core,config,data,sampling,evaluate,preprocess,utils}` public funcs; `core` now re-exports the E1 model store, `evaluate` now re-exports `export_sumo_model`/`import_sumo_model` (were reachable only via `funs_evaluate`, breaking V16qf)
- python (forthcoming) → `analyze_dataset` narrow diagnostics entrypoint (T18ry) — its output IS the override payload for the preprocessing defaults AND the auto-`domain` source, ⊥ a side feature
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
V19cn: public API ! use VOCAB nouns (sample/variable=parameter/response=QoI); ⊥ "job" in any itis_sumo signature, docstring, or result field
V20dm: consumer-facing entrypoints ! accept plain tabular data; ⊥ `FunctionJob`/`JobVariableSelection`/oSPARC status strings inside itis_sumo (test-enforced, sibling of V4ty)
V21pf: preprocessing ! auto-defaulted + overridable in domain vocabulary only; ⊥ transform vocabulary reachable from a public signature; effective config inspectable in the result
V22rs: results ! typed dataclass, original units + original names, `dataclasses.asdict()`-serializable; ⊥ `_hat`/`_std_hat` suffix keys in the public shape
V23er: ∀ error escaping `itis_sumo.api` ! be a `SumoError` subclass; ⊥ raw `KeyError`/`IndexError`/`ValueError` crossing the boundary
V24af: run dir ! discarded on success, PRESERVED on failure w/ its path + stderr tail attached to the raised `SumoEngineError`
V25sd: ∀ stochastic entrypoint ! default `seed=42`, overridable, effective seed echoed in the result
V26dd: `domain` and `distribution` ! remain distinct config objects; ⊥ derive a box from a distribution's `mean ± 3σ`
V27fq: one-shot entrypoints ! implemented internally as fit-then-query; ⊥ monolithic procedure that would need rewriting to expose a handle
V28tz: CI ! block a PR into `develop`/`main` whose `pyproject.toml` version lacks target branch's required prerelease suffix (`aN` develop, `bN` main) or regresses/duplicates PEP 440 ordering vs existing tags
V29yn: tag-on-merge job (develop/main) ! push only a `v<version>` tag, ⊥ version-bump commit; skip docs-only diff; existing-tag collision ! hard failure, ⊥ force-overwrite
V30wk: itis-sumo LICENSE + `pyproject.toml` license/classifiers ! match itis-dakota's (all-rights-reserved) until an explicit public/OSS decision is made; ⊥ an OSI classifier claiming otherwise
V31vp: any CI job that installs a built artifact and runs pytest against it ! checkout source first (`tests/` isn't shipped in the wheel) — ⊥ silently "pass" via pytest's 0-collected exit code
V32bb: PyPI publish + GitHub Release jobs (`publish`/`release` in `publish.yml`) ! gated behind explicit `workflow_dispatch`; ⊥ automatic cascade from a tag push. Feature `.devN` TestPyPI upload ! local Make target + `.env` token; ⊥ CI publication
V33tz: version check/tag job ! compare candidate version against highest existing tag via PEP 440 ordering; ⊥ emit version sorting lower than prior release channel/version
V34yn: merge-time tag job ! re-check live `origin` tags; existing `v<version>` collision → hard failure; ⊥ force-overwrite
V35wk: `workflow_dispatch` publish ! validate input `tag` matches `v[0-9]*.[0-9]*.[0-9]*` and built artifact version equals tag version before real PyPI upload
V36vp: itis-sumo LICENSE ! own IT'IS copyright/author notice; ⊥ copy itis-dakota attribution or Dakota-source LGPL paragraph
V37bb: `.env` ! gitignored before token-based publish target lands; token never committed
V38cc: `publish-testpypi-dev` ! clean worktree → compute/write next `.devN` → build/check/upload via `.env`; ⊥ commit/tag/CI publication
V39qf: dev version selection ! max(existing git-tag `.devN`, published TestPyPI `.devN`) + 1; uv upload auth ! username `__token__` + password token; ⊥ reuse published version or pass username + token together

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
T21vk|✓|VOCAB section (above) + `docs/reference/glossary.md` (nav-registered, strict build green); vocabulary enforced on the api layer by `tests/test_api_contract.py::TestPublicSurface`. Sweep of the OLDER modules' docstrings still outstanding → folded into T24cm|V19cn
T22ax|✓|`itis_sumo/api` spine ( `errors.py` taxonomy, `types.py` config+results, `_session.py` fit-then-query + run-dir lifetime, `workflows.py` public funcs, `__init__.py` surface): tabular input type, `PreprocessingSpec` (domain vocabulary), `SumoError` taxonomy, typed result dataclasses, seed policy, artifact policy|V20dm,V21pf,V22rs,V23er,V24af,V25sd
T23bn|✓|`itis_sumo.api`: `cross_validate()` + `evaluate_along_axes()` — internally fit-then-query; 34 contract tests + 10 real-Dakota end-to-end tests green. Equivalence-vs-flaskapi-glue pinning ⊥ done here (needs the mmux_vite checkout) → moves to the consumer PR|V27fq
T24cm|.|remove `preprocess/models.py::{FunctionJob,JobVariableSelection}`; re-express the Dakota sufficiency rule over tabular data; jobs→table adapter moves to flaskapi (this port DELETES itis-sumo code)|V20dm
T25dp|.|`itis_sumo.api`: remaining 6 workflows (UQ-w-uncertainty incl. the ~120-line erfinv/histogram block, correlation, Sobol, grid, MOGA, cv-accuracy-metrics) + E1 `export_model`/`evaluate_stored_model` facade|V22rs,R1-R4
T26eq|.|POST-PORT: extract the fitted-model handle (`fit()` → methods → `save()`/`load()`); carries the fitted preprocessing config ⇒ closes the E1 gap (model store persists archive+metadata+training copy but ⊥ preprocessor config, so a reloaded model cannot inverse-transform to original units)|V27fq,V10jk
T27fr|.|POST-PORT: split `domain` vs `distribution` config + consumer migration; absorb the mmux_vite `jgo/fullstack-logscale` work|V26dd
T28gs|.|POST-PORT `?`: decide whether `distribution` gets an auto-generated default — explicit discussion required, ⊥ silently defaulted|§C `?`
T29hw|~|`publish.yml` tag trigger accepts `v`-prefixed PEP 440 prereleases ✓; tagging `v0.1.0a1` BLOCKED on the one-time PyPI Trusted Publisher config (user action), then clean-venv install + `itis-sumo validate` + headless smoke|T1pw
T30qa|✓|release/CI workflow refresh: build→TestPyPI automatic on tag push (✓), PyPI+Release gated manual (✓, T35cc); auto-tag-on-branch redesigned to PR-time check + tag-only-at-merge (no bot commit) — see T31xx/T32yy; keep git-cliff release notes, dependency-review + concurrency (✓); skip weekly cron/healthchecks for now|§C,V17rt
T31xx|✓|CI: PR-time check job (feature branch→`.devN`, target `develop`→`aN`, target `main`→`bN`) blocking merge if `pyproject.toml` version missing, regresses, or duplicates PEP 440 order vs existing tags|V28tz,V33tz
T32yy|✓|replace `auto-tag.yml`'s commit-based bump w/ tag-only-at-merge (read already-bumped version from merged commit, re-check live tags, push `v<version>`, ⊥ commit) — removes need for branch-protection bypass|V29yn,V34yn
T33zz|✓|swap itis-sumo LICENSE + `pyproject.toml` `license`/classifiers to match itis-dakota (IT'IS Foundation - All Rights Reserved); drop MIT classifier|V30wk
T34aa|✓|`make publish-testpypi-dev` ! source `TESTPYPI_TOKEN` from local `.env` (gitignored) instead of requiring a pre-exported shell var|§C
T35cc|✓|gate `publish`/`release` jobs behind explicit `workflow_dispatch`; `build`+`verify` stay CI-only for release tags; feature `.devN` uploads move to local Make target|V31vp,V32bb
T36dd|✓|add `.env` to `.gitignore`; make `publish-testpypi` source `.env` without printing token, then build/check/publish|V37bb
T37ef|✓|validate manual publish tag + artifact version before PyPI upload|V35wk
T38dd|✓|make `publish-testpypi-dev` auto-compute/write `.devN`, build/check/upload directly to TestPyPI; CI verifies alpha/beta/rc before real PyPI; no dev tag cascade|V38cc,V39qf

## §B
id|date|cause|fix
B10rf|2026-08-19|local TestPyPI publish passed username + token to uv, which rejects that combination; dev counter scanned git tags only, so a previously uploaded `.dev1` was selected again|use `__token__` + password token and include TestPyPI JSON releases in next-version calculation|V39qf
B1|2026-08-07|itis-dakota 6.23+ `Interface::interface_cache()` throws `IndexError: map::at` on interface-less surrogate confs (`study()` ctor, before `execute()`) — pure data-fit surrogates never instantiate an interface → static map missing entry (R5)|pin `==1.5.9` (Dakota 6.20, no cache) for flaskapi parity; ladder T16mo; upstream get-or-create fix
B2|2026-08-07|E1 import round-trip: `add_surrogate_model` (`config/funs_create_dakota_conf.py`) decided whether the training file has a leading `eval_id` column purely from whether the substring `"processed"` appears in the training filename, not an explicit flag — staging the re-import training file as `{id}.training.dat` (T12ze) desynced Dakota's `use_variable_labels` column expectation from the actual CSV, surfacing as `Cannot reorder variables ... not a permutation of expected variable labels`|immediate: `sumo_model_store.py::_training_file_name` names both the stored and staged file `{id}.processed_training.dat`, preserving the substring; root-cause: `add_surrogate_model` heuristic replaced by explicit `has_eval_id_column` param, T17bq/V15zx (now ✓)
B3|2026-08-18|CI `prek` job fails on PRs that touch unformatted `.py` files — ruff I001 (unsorted imports) + EXE002 (executable bit, no shebang) + ruff format violations in `funs_evaluate.py`, `funs_create_dakota_conf.py`, `test_property_invariants.py`, `generate_docs_figures.py`, `test_metamodeling_analytical.py`, `test_unit_solver.py` — pre-commit config dropped but ruff formatting not enforced repo-wide first|fix: `ruff check --fix` + `ruff format` all touched files + `chmod -x` executable bits; preventive: §V17ab invariant — ∀ `.py` file must pass `ruff check` + `ruff format --check`
B4kt|2026-08-18|CI workflow-only changes were absent from `detect_changes`, so all downstream jobs could be skipped and an invalid `astral-sh/setup-uv@v10` reference remained unresolved behind an apparently green PR|classify `.github/workflows/**` as CI changes, run shared validation matrix, replace nonexistent action tag; V18rs
B6nr|2026-08-18|the workflow regression test hard-coded a Dependabot-managed action version and failed on a valid setup-uv bump; CI already validates the workflow itself|remove tests/test_ci_workflow.py and rely on CI validation

B7pv|2026-08-18|prek passed deleted Python paths from the workflow diff to Ruff, causing E902 after removing a test file|exclude deleted paths with `--diff-filter=d` before xargs

B8lf|2026-08-18|the develop merge retained executable mode on changed `funs_evaluate.py`, so prek Ruff raised EXE002 for a Python file without a shebang|clear executable bit; existing V17ab catches changed-file Ruff violations

B9dw|2026-08-18|after rewriting develop history, GitHub push events reported the unreachable pre-rewrite `before` SHA and detect_changes aborted with `bad object`|fall back to `git diff-tree` when the event base commit is unavailable; no code invariant added because this is CI history topology
B10cs|2026-08-18|`itis_sumo.api.evaluate_along_axes(at={...})` raised `SumoEngineError: 'x1'` for a PARTIAL held-value mapping — `create_samples_along_axes` reads `[cut_values[var] for var in input_vars]`, so an incomplete dict is a `KeyError`, not a documented default. Never surfaced in flaskapi because its frontend always sends every slider|`_map_held_values` completes the mapping from the sample means (the same value the no-`at` path uses) before translating; covered by `tests/test_api_workflows.py::TestAlongAxes::test_honours_the_values_the_caller_holds_fixed`. ⊥ new §V invariant: V23er already requires taxonomy errors, and the real lesson is that a caller-facing default must be materialised by the API layer rather than assumed by an internal helper
B11xy|2026-08-19|`publish.yml`'s `testpypi` job ran pytest against the installed wheel w/o a checkout step — `tests/` (not shipped in the wheel) was absent, pytest collected 0 items and exited 5 before the job ever reached the TestPyPI publish step (caught via `gh run view` on the `v0.1.0a1` tag push, which failed harmlessly — no upload occurred anywhere)|added `actions/checkout` before `download-artifact` (ordered first — checkout's default `git clean` would otherwise wipe an already-downloaded `dist/`); V31vp
