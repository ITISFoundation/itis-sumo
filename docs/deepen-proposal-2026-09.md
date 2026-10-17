# Deepen pass (proposed) — `run_surrogate`, the engine's surrogate-run seam · 2026-09-30

`/deepen` run. **Nothing applied** — this is the pick + diagnosis + researched design + SPEC
edits *pre-drafted* for the spec skill to write. Behavior sacred: tests green before AND after.
Context: `feat/api-log-scale` @ 92ee066, baseline suite **365 passed / 30 analytical**,
ruff + repo-wide `ty` clean. Companion: [architecture-review](architecture-review-2026-09.md) C1
(this is its full deepen form; C5 rides along as acceptance criteria).

## 1 · PICK THE SHALLOW

Engine MC-trio + predictions-parse family in `evaluate/funs_evaluate.py`. Ranked against the
session→engine seam (C2) and the grab-bag split (C3): worst interface-per-caller-knowledge,
worst change amplification, and the only candidate whose absence already forced tests to
re-implement production maths. One module; C2/C3 explicitly deferred.

## 2 · DIAGNOSE (defects, caveman, evidence)

- `funs_evaluate.py:173-255, 256-324, ~1205-1235: shallow. Same ~25-35 ln preamble 3× —
  setup_files, transform, write processed file + hand-built header, run, loadtxt — before the
  real work (erfinv spread / Pearson / Saltelli) begins. Unknown-unknowns: caller must know
  `transform()` wants ORIGINAL names (:199-200), output name is scalar (:200), the FILE carries
  MAPPED names (:201-209,217-218), the DataFrame must keep ORIGINAL keys (:203-210 NOTE).
  Get it wrong → wrong predictions, no error.`
- `funs_evaluate.py:179,:262,:1062: preprocessor param UNTYPED at all three engine sites.
  Required protocol is duck-typed folklore: input_variables[var].mapped_name, transform(),
  transform_independent_variables(), get_inverse_mapping(), inverse_transform_output_std().`
- `funs_evaluate.py:753-756, 800-803, 881-884 + data:370-373: predictions parse ×4 VERBATIM;
  manual-CV variant :518-527 the ONLY B22-tolerant one; grid shape :923-935 fifth. Divergence
  undocumented (extension, separator, eval_id).`
- `variance sqrt policy ×5: funs_evaluate.py:757-759, 805-806, 885-886 + data:371-374, 473-477
  (incl. is_file guard each time).`
- `_session.py:245-265 vs 771-788: fit duplicated (docstring even repeats itself).`
- `test_metamodeling_analytical.py:70-103: erfinv spread RE-IMPLEMENTED in test — production
  loop funs_evaluate.py:243-247 never called. test_sobol_indices.py:76-153: Saltelli+estimator
  RE-IMPLEMENTED — production :1154-1401 never called. Two copies of maths, one shipped =
  drift hazard (B18mt lesson: parallel paths that look right diverge).`
- `Change amplification proof: one processed-file format change → ~10 sites / 5 files
  (_session.py:264-265,787-788 · data:252-255 · funs_evaluate.py:217-218,291-292,1213-1214 ·
  sumo_model_store.py:69 · funs_create_dakota_conf.py:141,199). eval_id presence inferred from
  filename SUBSTRING "processed" (conf:134-141); store forced to preserve substring (:65-69).
  Format lies: space-delimited written under .csv names (funs_evaluate:217-218) while
  load_data parses .csv as COMMA (data:181-182) — feedback trap.`

Complexity is real — shows all three symptoms: change amplification ✓, high cognitive load
(name-space rules) ✓, unknown-unknown (filename voodoo) ✓.

## 3 · RESEARCH THE DEEPENING

Moves: **pull complexity down** + **place the seam with two adapters** ("the interface is the
test surface"; one adapter hypothetical, two real — here the second is the test fake, which is
precisely the point). Derived from the repo's own deeper modules: `wiofiles.capture_to_file`
(small interface, hides libc quirks) as shape, and the existing metadata mock precedent
`test_dakota_object.py:17-45` as fake seed.

Target shape (engine-internal, private module e.g. `evaluate/_run_surrogate.py`; NOT public
api — V16qf holds):

```python
@dataclass(frozen=True)
class SurrogateFit: ...            # run_dir, preprocessor, response, processed-file facts

def fit_surrogate(run_dir, samples, preprocessor, response) -> SurrogateFit
def run_surrogate(fit, samples: pd.DataFrame) -> pd.DataFrame
    # IN/OUT in ORIGINAL names (index + columns). Hides: mapped-name rewrite, header,
    # space-delimited write, eval_id annotation, predictions.dat + variances.dat parse
    # (B22 heal lives here ONCE), sqrt policy, optional per-fold _eval_id scatter.
    # columns: <response>_mean, <response>_std (absent variances -> no _std, per guard)
```

Callers rewired (8 variants → 1 interface): the MC trio `:173,:256,:1056`, manual CV `:453`
(per-fold call + scatter), grid `:892` + axes `:77` (via their readers), `evaluate_sumo:727`,
export/import parse sites `:764,:830` (sqrt copies fold in). Session `fit()` + MOGA fit collapse
to `fit_surrogate` (dup `_session.py:245-265/:771-788` deleted).

Fake adapter (the payoff): in-memory `FakeSurrogate` implementing the same interface from a
closed-form callable → erfinv spread, fold scatter/B22, 2-D grid reshape, Sobol estimator get
FAST unit tests; `test_metamodeling_analytical.py:70-103` and `test_sobol_indices.py:76-153`
in-test re-implementations replaced by calls to production code.

Considered & rejected: narrow `Protocol` for `preprocessor` alone — fixes the typing leak only;
the ×3 file dance and ×5 parse copies survive; `run_surrogate` subsumes it (preprocessor never
crosses the seam again).

## 4 · PROPOSE (SPEC.md edits — spec skill writes, not this file)

- **§I** amend E2/engine invariant: engine functions MUST NOT receive the fitted preprocessor;
  every surrogate evaluation flows through the single internal seam (`fit_surrogate` /
  `run_surrogate`), real + fake adapters.
- **§V** new (suggest `V46rs`): "Surrogate-run policy single-home — `predictions.dat`/
  `variances.dat` parsing, eval_id annotation, and sqrt policy exist in exactly one module
  (grep `loadtxt(` outside it = 0 hits on engine readers); fold-tolerance (B22) and spread maths
  are unit-testable with a fake surrogate; the erfinv spread and Sobol estimator are tested
  against production functions, not test copies." Verify: grep-budget check + fast-tier tests.
- **§T** new (suggest `T48rs`): extract seam → rewire 8 call sites + both fits → add
  `FakeSurrogate` unit tier → delete the two in-test maths re-implementations → update V&V
  counts. Predecessors: PR #47 merged; ponytail backlog A4 (`propagate_uq`) + A7
  (`retrieve_csv_result`) deleted first (shrinks absorption surface).
- **§B**: any name-space misrouting surfaced during extraction gets backprop'd, not patched
  silently (protocol: backprop skill).

## 5 · VERIFY (behavior-hold contract)

1. Full suite green before AND after (365 / 30 baseline) — no test edited except the two
   re-implementations swapped to production calls and any file-move churn.
2. Analytical tier numerically IDENTICAL pre/post (fixed seeds): compare one CV, one UQ, one
   Sobol, one grid, one MOGA result set byte-for-byte — the Dakota-facing files this seam
   writes must be unchanged (byte-diff the processed files in a kept workspace too).
3. Linear-mode bit-identity spot-check preserved (V21pf precedent: linear == today).
4. V24af workspace semantics untouched (keep-on-failure evidence path unchanged).
5. `ruff check` + repo-wide `ty check` clean (V41hp) after each commit.
6. Suite-count deltas logged in the V&V report (deleting test copies lowers count — expected).

## 6 · Net (estimates)

~120-150 production duplication lines absorbed · 3 untyped `preprocessor` params deleted
(reach-ins 10 → 0) · ~200 lines of in-test maths re-implementation replaced by fast fake tests ·
3 behaviours flip analytical-only → unit-fast · format change-amplification 10 sites/5 files → 1 file.

## 7 · Adjacent, deliberately NOT in this pass

C2 typed session seam (next pass, natural follow-on) · C3 pure cores beyond scatter/sqrt already
absorbed here · C4 workflows fold-in · C6 severances · ponytail A-deletes (land BEFORE, separate PR).
