# Ponytail backlog — repo audit + diff review (2026-09-29)

Scanned @ `feat/api-log-scale` (post T46ls/T47mt). Two passes, nothing applied.
Scope: over-engineering only — correctness/perf findings go through normal review.
Tags: `delete:` dead → nothing · `stdlib:` stdlib ships it · `native:` platform/dep does it ·
`yagni:` unused flexibility · `shrink:` same logic, fewer lines.

## A. Repo-wide audit (biggest cut first)

| # | Tag | What to cut | Replacement | Where | ~ln |
|---|-----|-------------|-------------|-------|-----|
| A1 | delete | `DataPreprocessor` fluent filter API: `filter_variables` / `filter_by_names` / `filter_by_patterns` / `filter_normalized_only` / `filter_non_normalized_only` / `get_filtered_variable_names` — zero callers tree-wide (the only `filter_variables` callers are the other dead methods in the same block) | nothing — product fits+transforms in-memory | `preprocess/data_preprocessor.py:637-786` | 150 |
| A2 | delete | Dakota-native CV chain: `evaluate_sumo_crossvalidation` + `_parse_crossvalidation_outputlogs` + `create_sumo_crossvalidation_conffile` + re-exports + tests — api derives all metrics from manual-CV predictions (`compute_cv_accuracy_metrics`); V&V "Known limitations" already declares the stdout-regex path non-canonical | `compute_cv_accuracy_metrics` (already live) | `evaluate/funs_evaluate.py:383-460`, `config/funs_create_dakota_conf.py:427-466` | 175 |
| A3 | yagni | `lhs()` five pyDOE methods (`center`/`maximin`/`centermaximin`/`correlation`/`lhsmu`) + `iter` + dispatch — sole product caller `_lhs(n, k, seed)` = classic | keep `_lhsclassic`; drop `_lhscentered`/`_lhsmaximin`/`_lhscorrelate`/`_lhsmu` + their method tests | `sampling/lhs.py:87-233` | 175 |
| A4 | delete | `propagate_uq` + `create_uq_propagation_conffile` — Dakota-native normal-only UQ, zero src callers; V&V: "not the pathway production traffic exercises" | `propagate_manual_uq_with_uncertainty` (live) | `evaluate/funs_evaluate.py:125-165`, `config/funs_create_dakota_conf.py:386-426` | 110 |
| A5 | delete | `get_non_dominated_indices` / `is_dominated` — non-dominance lives in Dakota's MOGA study; no src caller, tests only | nothing | `data/funs_data_processing.py:491-529` | 100 |
| A6 | delete | `DataPreprocessor.save_config` / `load_config` (+ `convert_numpy_types`) — flaskapi-era cross-request persistence; `SumoSession` holds the fitted preprocessor in memory; zero src callers | in-memory session (already live) | `preprocess/data_preprocessor.py:580-633` | 85 |
| A7 | delete | `retrieve_csv_result` — no src caller anywhere | nothing | `evaluate/funs_evaluate.py:35-63` | 43 |
| A8 | delete | `create_grid_samples` kwargs `downscaling_factor` + `gridpoints_file_name` — no call site ever passes either | 7 required args + fixed filename | `data/funs_data_processing.py:404-425` | 12 |
| A9 | delete | `DataPreprocessor.get_variable_mapping` — mirror of the used `get_inverse_mapping`; 1 test, 0 src callers | `get_inverse_mapping` + reversed lookup | `preprocess/data_preprocessor.py:564-570` | 10 |
| A10 | native | `scikit-learn` imported in exactly one file for one symbol (`KFold`) | ~8-line `rng.permutation` column-splits — **verify 42-seed fold parity first (tests pin it)** | `pyproject.toml`, `evaluate/funs_evaluate.py` | -1 dep |
| A11 | native | `pydantic` imported in exactly one file for one `BaseModel` (`SumoModelMetadata`) | `dataclasses` + `json` round-trip; counter: `model_validate_json` gives free disk-read validation, keep if sidecar forgiveness matters | `core/sumo_model_store.py` | -1 dep |
| A12 | yagni | `funs_dataset_diagnostics` / `analyze_dataset` exported with zero product callers — **spec-guarded**: SPEC §I pins it as the forthcoming T18ry entrypoint (BLOCKED on incubator promotion). Prune only via a spec decision, not silently | — | `data/funs_dataset_diagnostics.py` | 83 (deferred) |

Audit subtotal: **~-790 lines, -2 deps possible.**

## B. Diff / repo review (2026-09-29 pass)

| # | Tag | What to cut | Replacement | Where | ~ln |
|---|-----|-------------|-------------|-------|-----|
| B1 | shrink | "Distributions must cover variables exactly" guard written VERBATIM 3× (`uncertainty()`, `correlations()`, Sobol path) | one `_require_exact_distributions(distributions)` helper, called 3× | `api/_session.py:425-440, 466-480, 520-535` | -12 |
| B2 | delete | `wiofiles` demo: module-scope `pylibc = ctypes.PyDLL(None)` (runs at every import!) + `main()` smoke script with its capture/read block copy-pasted twice | nothing — production surface is `capture_to_file`; move demo to `scripts/` if wanted | `core/wiofiles.py:66-105` | -35 |
| B3 | shrink | `get_dakota_version`: `hasattr(itis_dakota, "__version__")` branch + nested importlib fallback | one try/except on `importlib.metadata.version("itis-dakota")` → `None` | `utils/helpers.py:68-90` | -6 |
| B4 | shrink | `generate_grid_samples` log block: `log_names` set + `build_bounds` loop + separate exp-back loop | one dict-comp `{name: ln-bounds or raw-bounds}`; exp-back stays | `api/workflows.py:455-480` | -6 |
| B5 | shrink | `_parse_data` pre-initializes `data = []` never used | `return [line.split() for line in Path(file).read_text().splitlines()]`, 2 lines | `data/funs_data_processing.py` | -2 |

Review subtotal: **net: -61 lines possible.**

## Deliberately NOT flagged (load-bearing)

- Scale tripwires (`scale_distribution`/`resolve_log_scale`/required `input_scales`/`output_scale`) — V45ls enforcement, not ceremony.
- Flip matrix + behaviour tests (~hundreds of lines) — evidence, ponytail minimum.
- `capture_to_file`'s dup2 + `libc.fflush` hand-roll — captures the Dakota wheel's C-level stdio; stdlib `contextlib.redirect_stdout` cannot.
- `mkdtemp` + conditional `rmtree` (3 sites) — "discard on success, keep on failure" is the documented `workspace` contract; `TemporaryDirectory` would delete evidence.
- `data/config/evaluate/__init__` re-export pattern — V16qf §I convention.
- 2× duplication of the MC-samples prelude (`correlate_manual_uq_samples` vs `propagate_manual_uq_with_uncertainty`) — under the rule of three; extraction return-shape would cost more than the copy.

## Notes for whoever picks these up

- Each cut = its own commit; run full suite + repo-wide `ty check` (V41hp) after every one.
- A2/A4 also shed config-surface — re-run the analytical tier (real Dakota), not just unit tier.
- A10 is the only parity-risk cut (fold layout is test-pinned).
- A12 needs a spec decision (T18ry) — do not delete unilaterally.
- Removing dead code ALSO removes its tests; suite count drops are expected, update the V&V report counts.
