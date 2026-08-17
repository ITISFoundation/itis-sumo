# BRANCH_CONSOLIDATION — branch/tooling sprawl diagnosis + incubator inventory

Working doc, not a spec (see `SPEC.md`/`DOCS_SPEC.md` for those). Two purposes:
1. Preserve the original branch-consolidation diagnosis/plan (2026-08-13/14) as a
   durable record, since it lived only in an ephemeral plan-mode file.
2. Track, over time, what's sitting on the confidential incubator branch
   (`feat/nih-in-silico-example`) and whether/when each piece is ready to be
   individually promoted into `develop` — the promotion rule `SPEC.md` §C and
   this repo's branching model both assume but don't themselves track.

---

## 1. Original diagnosis & plan (2026-08-13/14, as captured in plan mode)

### Context

Reviewed `itis-sumo` because things felt scattered: documentation split across
branches, and new UQ/meta-modeling tooling (scale selection, convergence
diagnostics, calibration/coverage) being developed *at the same time* as used
on real data (the NIH worked example, and a validation pass on Giuliana's
dataset). Building-while-using tends to produce shaky, under-tested tooling
that needs a few real uses before it "coalesces" — new-idea work should stay
visibly separate from the consolidated, trustworthy core.

### What the repo showed at the time

| Branch | Last commit | State |
|---|---|---|
| `main` | 2026-08-11 | Stale — predates the docs restructure and everything since |
| `feat/vv-port-tests-and-docs` | 2026-08-12 | MkDocs/Diátaxis restructure + V&V report. Self-contained, not merged to `main` |
| `feat/sumo-model-export-import` | 2026-08-07 | E1 (model export/import). SPEC marked T12–T14 `✓` — done, tested, not merged |
| `worktree-task-1.6-ci-release` | 2026-08-10 | CI/release process (Task 1.6). Self-contained, not merged |
| `feat/nih-in-silico-example` | 2026-08-13 | 17 commits ahead of `main`: docs restructure, then the NIH worked example, then a growing run of new tooling (T19ny→T29ny) |
| `feat/merck-vns-thermal-uq` | 2026-08-12 | Intentionally separate/confidential — split working as intended |

The "split documentation" feeling was real: the Diátaxis MkDocs restructure
only existed on `feat/vv-port-tests-and-docs` and `feat/nih-in-silico-example`
— `main` never got it.

The dogfooding-while-building pattern was visible directly in `SPEC.md`'s own
task ledger on the incubator branch (T19ny–T29ny): diagnostics code lands,
*then* tests/baselines get retrofitted a commit or two later.

Giuliana's dataset validation didn't show up anywhere in the repo — no
committed/referenced dataset, no SPEC status change — even though the SPEC
already anticipated it (`T18rc`, still `.`).

### The actual shape of the problem

Two different things were being conflated under "everything is a bit
everywhere":

1. **Unmerged-but-actually-done work.** `feat/vv-port-tests-and-docs`,
   `feat/sumo-model-export-import` (E1), and `worktree-task-1.6-ci-release`
   were all scoped, complete, SPEC-marked-`✓` features sitting unmerged. A
   merging backlog, not a "too green to consolidate" problem.
2. **Genuinely-still-forming tooling.** The scale-selection /
   convergence-diagnostics / calibration-coverage machinery on
   `feat/nih-in-silico-example` had only been exercised against one dataset
   (NIH) at the time. By the stated bar — prove it a few times before calling
   it consolidated — it wasn't ready to be called core yet.

### Recommended mental model

- **Core (`main` / the `itis_sumo` package):** only things proven across ≥1
  real independent use case, with tests, stable API, and docs.
- **Incubator (feature branches like `feat/nih-in-silico-example`):** where
  new tooling gets built *and* dogfooded together on purpose — expected to be
  shaky. The mistake isn't building-while-using, it's letting the incubator
  become the only place the docs and the "done" features live too.
- **Promotion rule:** a tool graduates from incubator to core once it's (a)
  been run against a second, independent dataset with results matching
  expectations, (b) has tests that predate its last behavior change rather
  than trailing it, and (c) has stable docs.

### Decisions made

- Adopt the `mmux_vite`-style branching model: `main` = stable/released,
  `develop` = staging buffer feature branches merge into before eventually
  reaching `main`. `main` stays untouched until `develop` itself is judged
  stable.
- All of `feat/nih-in-silico-example` (T19–T29) judged still-shaky as a whole
  — none of it merges into `develop` as a block. Once `develop` exists,
  individual pieces get cherry-picked / opened as their own PRs into `develop`
  one at a time as they prove out (§3 below is where that tracking lives).
- Longer-term direction: `itis-sumo` becomes the headless computation package
  that `../mmux_vite`'s Flask API consumes (Flask as a thin interface layer),
  rather than `mmux_vite` carrying its own in-tree core — already the
  trajectory `SPEC.md` describes (`future consumer`, `T15mn`). Development of
  new UQ/meta-modeling capability continues to happen *in* `itis-sumo`
  (settled — reverting to developing in `mmux_vite/flaskapi` would reopen the
  flask/osparc coupling `V4ty` exists to prevent). What's still open is *how*
  `mmux_vite` installs `itis-sumo` reliably (`T15mn`) — a packaging problem,
  deferred to its own session. See `SPEC.md` §C's narrow/versioned/deep
  entrypoint principle (added 2026-08-17) as the design answer for that
  integration once it's picked back up.

### Plan of action (as executed)

1. Preserve uncommitted docs-serving work (`Makefile` + `docker/Dockerfile.docs`) as its own commit on the incubator branch.
2. Create `develop` from `main`.
3. Merge `feat/vv-port-tests-and-docs` into `develop` first (docs restructure + V&V report). Run tests after.
4. Cherry-pick the docs-serving commit onto `develop`.
5. Verify docs serving on `develop` (local build/serve; Tailscale reachability checked by the user separately).
6. Once confirmed, branch a docs-focused branch off `develop` for further doc work.
7. Merge `feat/sumo-model-export-import` into `develop` (E1). Run tests after.
8. Merge `worktree-task-1.6-ci-release` into `develop` last, with a careful review pass (publish step flagged as under-curated, not blindly landed).
9. Leave `feat/nih-in-silico-example` and `feat/merck-vns-thermal-uq` exactly as they were — incubator/private, untouched by this consolidation.
10. Leave `main` untouched — no merge from `develop` back to `main` in this pass.

Explicitly deferred: merging any NIH tooling, updating `T18rc`, and the
`mmux_vite`/Flask packaging integration (`T15mn`).

---

## 2. Status update (2026-08-17)

Since the plan above was written:

- `develop` exists, is the current checked-out branch, and is ahead of
  `origin/develop` (steps 1–8 above are done: docs restructure, E1, and the
  CI/release branch are all merged in; `main`/`origin/main` remain untouched
  at `987b8cc`).
- `feat/merck-vns-thermal-uq` has since been **merged into
  `feat/nih-in-silico-example`** (`c2076a7`, 2026-08-15), not into `develop`.
  This raised the confidentiality bar for the *whole combined branch* —
  including the previously-not-confidential NIH portions — since Merck's
  stricter terms now cover everything sharing the branch. See memory
  `merck_vns_confidential_branch` for the full policy. The branch is still
  local-only, never pushed, and per that policy will **never** be merged into
  `develop`/`main` wholesale — only individually-proven pieces get ported out.
- `feat/nih-in-silico-example` picked up further tooling since the plan was
  written: `a0fb552` (mean-signed-error bias metric + response-scale
  isolation analysis), on top of T19df–T29ny.
- `T18rc` (second independent dataset as proof point) is **still `.`** in the
  incubator SPEC — Giuliana's dataset pass was never written back into the
  repo. However, per §3 below, the *scale/distribution* piece specifically
  has effectively gotten its second proof point already, via the Merck
  dataset (a different, independent use of `auto_select_distributions`) —
  just not captured as a formal `T18rc` close-out, and not a repeatable
  automated check.
- `SPEC.md` §C/§V/§T amended (`a837401`, 2026-08-17) to document the narrow,
  versioned, "deep" cross-repo entrypoint principle, with `T18ry` tracking a
  forthcoming `analyze_dataset()` entrypoint — explicitly `BLOCKED` on
  promoting the diagnostics logic inventoried below. Decision: scaffold the
  entrypoint's shape (dataclasses + function signature) on `develop` now,
  without wiring in real detection logic, and revisit full promotion once
  §3's readiness gaps close.

---

## 3. Incubator inventory — `feat/nih-in-silico-example`

Generic-tooling inventory only — dataset/report specifics stay off this file
per the confidentiality policy; where a dataset is named it uses the same
generic framing already committed on that branch (no internal project codes,
no co-author names).

| Piece | Where (branch-only, not on `develop`) | Introduced | Unit tests | Cross-dataset proof | Docs | Promotion readiness |
|---|---|---|---|---|---|---|
| `select_variable_scale` / `auto_select_distributions` | `src/itis_sumo/data/funs_data_processing.py` | `a15ce23` | 8 tests, `tests/test_dakota_funs_data_processing.py` — synthetic (`rng.normal/uniform/lognormal`), not tied to any confidential dataset | **Yes** — used independently on the NIH SPARC in-silico dataset (`report_scale_selection`, all 19 outputs, `6287338`) and on the Merck Healthcare KGaA VNS thermal-safety dataset (Phase A) | `docs/how-to/select-distribution-scale.md` | **Closest to ready.** Has synthetic unit coverage + two independent real-dataset uses, satisfying the plan's "second proof point" bar in spirit. Missing: a formal `T18rc`-style close-out and the raw-value outlier detector (see next row) it would need to fully back `analyze_dataset()`'s outlier-surfacing half. |
| Raw-value outlier detection (input/output columns directly) | **Does not exist yet**, anywhere | — | — | — | — | Only `_tukey_outlier_mask` (private, `src/itis_sumo/evaluate/funs_evaluate.py`) exists, and it operates on CV *residuals*, not raw input/output values (`V17kb`: unfiltered MAE stays primary metric). Reusing its IQR technique on raw columns is new work, not a port. |
| `compute_cv_diagnostics` (rmse/mae/paired-ttest/cohens_d/tukey-flags) | `src/itis_sumo/evaluate/funs_evaluate.py` | `5a15419` (T19df) | Yes, `tests/test_funs_evaluate_cv_stats.py::TestComputeCvDiagnostics` — synthetic | NIH dataset only; not exercised against Merck data | `docs/explanation/convergence-diagnostics.md` | Tests are synthetic/generic and solid, but only one real-dataset use so far — same "one proof point" gap the original plan flagged. Not part of the `analyze_dataset()` scope directly (it's CV/convergence, not input-scale diagnostics), but shares the promotion-rule question. |
| `compute_coverage` (nominal-vs-empirical PI coverage) | `src/itis_sumo/evaluate/funs_evaluate.py` | `95c5ac4` (T24bn) | Yes, `TestComputeCoverage` — synthetic | NIH dataset only | `docs/explanation/convergence-diagnostics.md` (calibration section) | Same status as `compute_cv_diagnostics` above — solid tests, single-dataset real-world use. |
| `mean_signed_error` bias metric + response-scale isolation analysis | `src/itis_sumo/evaluate/funs_evaluate.py` | `a0fb552` | Not yet inventoried in detail | NIH dataset only | `docs/explanation/convergence-diagnostics.md` | Newest addition (2026-08-14) — least proven of the set, needs its own look before any promotion decision. |

### Immediate implication for `T18ry` (`analyze_dataset()`)

The scale/distribution half of the proposed entrypoint has real, if informal,
two-dataset backing — the outlier-surfacing half does not exist yet in any
form that operates on raw values. Scaffolding `analyze_dataset()`'s shape now
on `develop` (per the 2026-08-17 decision above) is safe; wiring in real
detection logic means either porting `select_variable_scale`/
`auto_select_distributions` as their own clean commit (closest analogue:
E1's promotion via `feat/sumo-model-export-import` → `develop`) or writing a
fresh raw-value outlier detector from scratch, since none exists to port yet.
