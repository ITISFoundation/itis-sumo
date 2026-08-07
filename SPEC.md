# SPEC — itis-sumo

Caveman-encoded (drop articles/filler; `→` becomes, `!` must, `?` may/uncertain, `⊥` never). Standalone surrogate-modeling library aggregating prior MetaModeling trials.

## LINKS
- engine wheel → `itis-dakota/` (ITISFoundation/itis-dakota fork, pybind11); pin `==1.5.9` (Dakota 6.20, parity w/ mmux/vite); fork checkout = 6.24 for T16mo ladder
- v1 port source → `../mmux_vite/flaskapi/src/mmux_flaskapi/{dakota,data_preprocessor}/`
- E1 source → `../mmux_vite` branch `jgo/sumo-model-export-import` (flaskapi/SPEC.md R2-R8, V36-V38, T24)
- future consumer → `../mmux_vite/flaskapi` (swaps in-tree core → this pkg)
- research → `DAKOTA-STUBS.md` (stubs + JSON→pydantic POC)

## §G
Standalone pkg `itis-sumo` (SuMo = Surrogate Model): aggregate MetaModeling core from mmux/vite + prior trials (mmux_python, itis_dakota_projects) into one wheel-driven library, usable BOTH inside mmux/vite AND headless (notebook/scripts) → build/evaluate/cross-validate surrogates + UQ (Sobol/correlation) + sampling (LHS/grid) + MOGA. First expansion (E1): SuMo model export/import persistence.

## §C
- engine ! itis-dakota wheel `dakota.environment.study(input_string=...)`; ⊥ subprocess `dakota` bin, ⊥ vendor surfpack
- v1 config ! NIDR string composers (proven); JSON/pydantic seam later (DAKOTA-STUBS POC, Dakota 6.24 experimental)
- port ! near-verbatim from mmux_flaskapi `dakota/`+`data_preprocessor/`; re-root `mmux_flaskapi.`→`itis_sumo.`; prune known-dead (`_parse_json_dict`/`process_json_file`, `add_interface_s4l`, `create_function_sampling_conffile`, `create_moga_iterative_optimization_conffile`); de-web `JobVariableSelection`/`required_completed_jobs` (move from blueprints into pkg)
- recycle itis_dakota_projects: `validate_dakota_installation()`, `get_dakota_version()`, `validate` CLI, config sanity-guard; ⊥ subprocess `DakotaProject`/factory/generator/parser/activity-logger
- deps ! itis-dakota==1.5.9 (Dakota 6.20, parity w/ mmux/vite; stay until T16mo resolves the 6.23+ `interface_cache` regression — R5), numpy, pandas, scipy; scikit-learn ⊥ KFold only
- py.typed; `src/` layout; docs MkDocs + headless notebook; tests standalone ⊥ flask/osparc
- versioning single-source; uv + pip-compile pins; Conventional Commits
- E1: artifacts keyed server `sumo_model_id` (uuid); metadata sidecar `{id}.metadata.json`; ⊥ user-supplied path/prefix keys (traversal)
- py ! 3.11 (match mmux/vite; 1.5.9 ships no cp313 wheel); 3.13 only via T16mo rung 1 (1.5.11 cp313) or rung 2 (6.24)

## §I
- wheel: `dakota.environment.study`, `study.execute()` (⊖ `dakota.surrogates` until wheel bump, R4)
- CLI: `itis-sumo validate`
- python: `itis_sumo.{core,config,data,sampling,evaluate,preprocess,utils}` public funcs
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
V10jk: E1 export ! `{id}.metadata.json` sidecar w/ verbatim conf block + ordered input descriptors + output descriptor + format
V11lz: E1 import ! conf block identical to export block bar export↔import swap
V12xc: E1 keying ! server `sumo_model_id` uuid; ⊥ user-supplied path keys
V13vb: E1 models dir ! single env-overridable source (⊥ branch's dual-strategy split)
V14nm: ⊥ standalone surfpack dep; future standalone-eval → bump itis-dakota (`dakota.surrogates`)

## §R
R1: `export_model`/`import_model` child keywords; formats `text_archive`(.sps)/`binary_archive`(.bsps)/`algebraic_file`(.alg); naming `{prefix}.{resp}.{ext}` | branch R2
R2: `import_model` needs conf block identical to export block bar keyword swap; ? whether training VALUES must match — empirical round-trip test | branch R3
R3: .sps archive self-describing vs positional UNVERIFIED `?` → sidecar regardless | branch R8
R4: `dakota.surrogates` ⊖ in wheel 1.5.9; upstream 6.24 adds it — wheel bump over new dep | branch R5-R7
R5: 6.23+ regression — `study()` ctor → `Interface::interface_cache(problem_db)` static map (DakotaInterface.hpp:71), populated only via SimulationModel/NestedModel ctor (`Interface::get_interface`); pure data-fit surrogate confs (`import_build_points_file`, no interface) never populate → `DakotaInterface.cpp:77` `.at()` throws → `IndexError: map::at` before `execute()`. 6.20 ctor used `problem_description_db().interface_list()` (no cache → no crash). Upstream fix candidate = get-or-create (`operator[]`)

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
T10le|.|docs README + MkDocs + headless notebook|§G
T11rt|✓|verify: standalone pytest green (⊖ flask), clean-venv install, `itis-sumo validate`, headless surrogate→CV→Sobol|§C
T12ze|✓|E1: port `sumo_model_store` + export/import API into pkg; unify models-dir strategy|V10,V12,V13
T13uv|✓|E1: capture verbatim conf block for sidecar (close branch placeholder gap)|V10
T14qa|✓|E1: empirical export→import round-trip test; resolve R2 `?` (training VALUES)|R2,V11
T15mn|✓|E1: wire mmux/vite flaskapi endpoints to pkg export/import (post-v1, separate PR)|§I,R1-R4
T16mo|.|stepwise engine modernization: rung 1 `1.5.9→1.5.11` (packaging-only: drop py3.8, add cp313; same 6.20 engine — behavior-identical; re-run smoke); rung 2 `1.5.11→6.24.x` co-shipped w/ JSON input seam (R5 regression fixed upstream first, re-align py 3.13, re-run full smoke)|R4,R5

## §B
id|date|cause|fix
B1|2026-08-07|itis-dakota 6.23+ `Interface::interface_cache()` throws `IndexError: map::at` on interface-less surrogate confs (`study()` ctor, before `execute()`) — pure data-fit surrogates never instantiate an interface → static map missing entry (R5)|pin `==1.5.9` (Dakota 6.20, no cache) for flaskapi parity; ladder T16mo; upstream get-or-create fix
