# Architecture review — deepening opportunities (2026-09-30)

Run: `/improve-codebase-architecture` + `codebase-design` vocabulary over `feat/api-log-scale`
(92ee066), via a very-thorough structural walk + session knowledge from the log-scale work.
MD replaces the skill's usual HTML report at user request (durable for a later work session).
Companions: [ponytail-backlog](ponytail-backlog-2026-09.md) (what to DELETE — do those FIRST,
they shrink the surface these candidates touch), [ponytail-review](ponytail-review-2026-09.md).
All three files are intentionally **untracked** on the feature branch — land them on `develop`.

Prior art: no `CONTEXT.md`, no `docs/adr/`. SPEC.md §G/§I + `docs/VERIFICATION_VALIDATION_PLAN.md`
are the governing notes. Constraints they record that these proposals must respect:
- **V16qf** — `itis_sumo.api` is the only consumer-facing layer; engine import direction never inverts.
- **T26eq** — the api's fit-step/query-step split is *deliberate*: a future "hold a fitted surrogate"
  handle reuses session methods verbatim. Candidates below advance that, they don't fight it.

Shape of the code (src LOC): `api/` 1,358 (types+errors+session+workflows+re-exports) →
engine ~4,700 (`funs_evaluate` 1405 · `funs_data_processing` 816 · `data_preprocessor` 786 ·
`funs_create_dakota_conf` 569 · lhs/store/etc) → `itis-dakota` wheel. Deletion tests were run
per candidate: **CONCENTRATES** = real module worth deepening; **MOVE** = pass-through, only
worth doing when fused with something that concentrates.

---

## C1 · `run_surrogate(fit, samples) -> predictions` — engine-internal seam ★ TOP PICK · strong

**Files:** `evaluate/funs_evaluate.py:173-255` (`propagate_manual_uq_with_uncertainty`),
`:256-324` (`correlate_manual_uq_samples`), `:~1205-1235` (Sobol prelude),
parse copies at `:753-756`/`:800-803`/`:881-884` (+ B22-tolerant variant `:518-527`, grid shape
`:923-935`), variance-sqrt copies `:757-759`/`:805-806`/`:885-886` + `data/funs_data_processing.py:371-374,473-477`,
duplicated fit `_session.py:245-265` vs `:771-788`.

**Problem:** Eight near-copies of one handshake — write samples → `preprocessor.transform` →
write space-delimited processed file → `DakotaObject` evaluate → parse `predictions.dat` →
optional `np.sqrt(variances.dat)` — scattered across the engine with undocumented per-copy
divergence (extension, eval_id handling, B22 tolerance, column order). The `preprocessor` param
is **untyped** at all three engine sites (`funs_evaluate.py:179,:262,:1062`) and callers must
secretly know the name-space rules (transform wants ORIGINAL names; the file carries MAPPED
names; the DataFrame must keep original keys). Two production maths (erfinv spread `:243-247`,
Sobol estimator `:1154-1401`) are so entangled with this dance that the "validation" tests
**re-implement the maths instead of calling it** (`test_metamodeling_analytical.py:70-103`,
`test_sobol_indices.py:76-153`) — a shipped-copy vs tested-copy drift hazard (B18mt precedent).

**Fix:** one engine-internal module: `fit_surrogate(...) -> SurrogateFit` +
`run_surrogate(fit, samples_df) -> predictions_df` (original-name indexed; `<output>_mean` /
`<output>_std` columns; sqrt + optional-variances + B22-heal + eval_id + per-fold scatter all
behind it). Second adapter = an in-test `FakeSurrogate` (the repo already mocks at this shape —
`test_dakota_object.py:17-45` fabricates model-metadata).

**Wins:**
- **Leverage/depth:** 1 small interface hides ~800 ln of 8-variant plumbing; deleting it makes
  the 3 preambles + parse copies re-condense → CONCENTRATES, real module.
- **Locality:** a Dakota file-format change today touches ~10 sites / 5 files
  (`_session.py:264-265,787-788`; `data:252-255`; `funs_evaluate.py:217-218,291-292,1213-1214`;
  `sumo_model_store.py:69`; `config:141,199`) → 1 file.
- **Test economics:** erfinv spread, fold scatter (B22), 2-D grid reshape flip from
  analytical/integration-only to fast fake-surrogate unit tests; ~200 ln of in-test
  re-implementation deleted.
- Removes untyped `preprocessor` from 3 signatures; its 10 reach-ins (`mapped_name`,
  `transform`, `get_inverse_mapping`, `inverse_transform_output_std`, …) → 0.
- The real+fake adapter pair makes this a *real* seam (one adapter = hypothetical; two = real),
  so it is ponytail-safe — it is not speculative abstraction, its second implementation is the
  test surface the repo is missing.

**Before:** `funs = _setup_process_files(...); write_csv(df, pfile); write header; run_surrogate_dakota(fids, pfile, df); loadtxt×3; sqrt×2` ×8 variants, 3 untyped params, hidden name-space rules.
**After:** `preds = run_surrogate(fit, samples)` anywhere; format policy lives once; fakes allowed.

**Sequencing:** after PR #47 merges (same files in flight); take ponytail A4/A7 (dead
`propagate_uq`, `retrieve_csv_result`) first so fewer copies are absorbed.

---

## C2 · Typed session→engine seam · worth exploring (do after C1)

**Files:** `api/_session.py:271-548` (7 query methods), `_run_engine:701-711`, arg-order facts
at `:275-285,:323-332,:386-396,:437-447,:478-490,:533-548`.

**Problem:** `_run_engine(description, function, *args, **kwargs)` is untyped positional; each
query hand-rolls the engine's arg order, renames kwargs on the way through
(`folds→N_CROSS_VALIDATION`, `points_per_variable→NSAMPLESPERVAR`, `cut_values`), and mmux-era
engine names leak into api call sites. `_run_engine` already concentrates error-mapping +
workspace/stderr plumbing — deletion test: removing it pushes that into 7 sites → CONCENTRATES;
the seam exists but its *interface* is untyped.

**Fix:** per-query typed adapter functions (thin, named args, private) that own run-dir +
V24af-evidence + error mapping, so each session method is one line. This is exactly the shape
**T39pk/T26eq** (`SPEC.md:153,168`) need for the future fitted-model handle to be a re-export.

**Wins:** positional-order facts deleted; kwarg-rename noise gone; queries uniform; the
`SumoSession` doc's "internally already split" promise gets a typed spine.

---

## C3 · Split `funs_*` by concept ONLY if pure cores come out · worth exploring

**Files:** `evaluate/funs_evaluate.py` (1405 ln), `data/funs_data_processing.py` (816 ln) +
their `__init__` facades.

**Problem:** `funs_*` names state zero concepts (mmux heritage — `test_dakota_funs_evaluate.py:1`
still says "mmux_flaskapi"). `funs_data_processing` holds correlation math AND Pareto dominance
AND scale math; `funs_evaluate` holds CV statistics AND file-format healing. Import graph is
clean (data = bottom layer, no cycles) so splitting is *safe* — which is also the warning:
**pure re-filing is MOVE** (adds seams, concentrates nothing).

**Fix (only the concentrating part):** three callable pure cores —
`sobol_from_evaluations` (from `:1154-1401`), `inject_predictive_std` (kills the 5× sqrt copies),
`_scatter_fold_predictions` (`:542-551,:570-581`, shared with C1) — plus concept renames at the
facade level (`__init__` re-exports make renames cheap). Fold `test_sobol_indices.py`'s three
genuinely-pure tests off the module-wide analytical mark while there.

**Ponytail boundary:** stop at ~5 concept modules; never below one concept per file; keep the
`data/config/evaluate` `__init__` facade convention (V16qf §I).

---

## C4 · Fold `workflows.py` pass-throughs into the session · worth exploring

**Files:** `api/workflows.py`, `api/_session.py`.

**Problem:** deletion test verdict = **MOVE**: 6 of 7 non-MOGA wrappers add docstring + one
relay line; MOGA (`:367-398`) adds the only real glue (flat CSV → nested distributions). The
session currently does NOT re-validate its args, so deleting workflows outright would leak
`ValueError` through the taxonomy — that's the guard-relocation work, not a reason for the layer.

**Fix:** relocate the six guards (`folds<2 :89-90`, axes `:125-126`, points-per-var `:160-161`,
grid bounds `:416-417`, empty-samples `:194-195` + `:263-264`) into the session's entry points,
then workflows becomes an import/re-export table keeping the user-facing docstrings. One
implementer per behaviour; T26eq gets a single surface to grow the fitted-model handle on.

---

## C5 · Processed-file format voodoo — mostly absorbed by C1 · strong (as C1 acceptance)

**Files:** `config/funs_create_dakota_conf.py:134-141` (+ kwarg threading `:154,:361,:396,:433,:474,:542`),
`core/sumo_model_store.py:65-69`, `data/funs_data_processing.py:181-182,242-243,252-255`,
`funs_evaluate.py:217-218,291-292,1213-1214`.

**Problem:** whether a processed file carries the `%eval_id` annotation is inferred from the
filename *substring* "processed"; the model store is forced to preserve that substring for
correctness; one format masquerades under `.csv` (space-delimited!), `.txt`, and `.dat`, while
`load_data` parses `.csv` as comma — a feedback trap if predictions are ever re-fed as input.
Plus a dead `%eval_id` branch (`data:242-243`, sanitizer already rewrote `%`).

**Fix:** canonical `(name, format, annotation)` owned inside C1's seam; store and conf
creators consume the constant, substring inference deleted. Track inside C1 rather than
separately.

---

## C6 · Small severances (each its own commit, each its own grep)

- `funs_evaluate.py:365-380` → `_session.py:492-510`: stringly-typed dict crossed into a
  16-key hand-copied `UncertaintyResult`; typo = runtime KeyError → engine returns the typed
  result (or a narrow TypedDict) at the seam.
- `_session.py:407-414`: if/else whose two branches are *identical* → collapse.
- `sumo_model_store.py:117-125` vs `:156-179`: existence + header-fallback knowledge split
  across two idioms → one (deletion test: the load/stage pair itself is real, CONCENTRATES R2/R9).
- `utils/` is a grab-bag: `config_guard.py` (NIDR text sanity, sole consumer `cli.py`) is
  config-layer domain; `helpers.py` = install introspection + `create_run_dir`.
- Library `print()`s at `funs_evaluate.py:552-553`; dead params `xscale/yscale/label_converter/MAKEPLOT`
  `:87-90` + commented-out params `:899-905`; misplaced docstring `:734-739`; "Why???" grid
  reshape `:963-983` (C1 territory); `get_variable_names` exported, zero callers.
- Duplicate "cover exactly" guard ×3 → backlog **B1** (same fix, logged there).
- Overlaps with ponytail backlog A-items are intentional — one ledger each, no double-cutting.

---

## Not flagged (deliberately — load-bearing as-is)

- `api/types.py` + `api/errors.py` — deep, small; the whole V16qf thinness is *chosen*, not accidental.
- `core/wiofiles.py:capture_to_file` — ~35 ln hiding dup2/fflush/libc quirks; CONCENTRATES hard; keep.
- `DataPreprocessor`'s fit/transform core — cohesive engine of the session (its dead filter
  family is a backlog delete, not an architecture problem).
- The MC-prelude 2× duplication beyond C1's absorption — rule-of-three until C1 lands.
- Flip-matrix/scale tests — evidence, ponytail minimum.

## Constraints for any pick

- Behavior sacred; linear defaults bit-identical; V&V test counts updated with every cut.
- Full suite + repo-wide `ty check` (V41hp) after every commit.
- SPEC §I/§V/§T edits go through the **spec skill** (sole mutator) — none proposed here
  except in [deepen-proposal-2026-09.md](deepen-proposal-2026-09.md), which pre-drafts the C1 edits.
- Ordering: PR #47 merges → backlog A-deletes land → C1(+C5) → C2 → C3 cores → C4/C6 opportunistic.

---

## Glossary (shared design vocabulary — used above)

- **Module** — anything with an interface and implementation: function, class, package.
- **Interface** — all a caller must know: signature, invariants, ordering, error modes, config.
- **Depth** — behaviour hidden per unit of interface learned. **Deep** = much behind little.
- **Shallow** — interface nearly as complex as the implementation.
- **Seam** — where a module is cut; where behaviour can change without editing there.
- **Adapter** — a concrete implementation at a seam. One = hypothetical; two = real.
- **Leverage** — what callers gain from depth. **Locality** — what maintainers gain: change in
  one place, bugs where the fix is.
- **Deletion test** — delete it: complexity **concentrates** somewhere real (keep/deepen) or
  just **moves** (pass-through).
