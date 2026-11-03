# Ponytail review log — 2026-09-29 pass

Companion to [`ponytail-backlog-2026-09.md`](ponytail-backlog-2026-09.md) (which holds the
full repo-wide audit A1–A12 + this review as B1–B5 with line estimates).
This file logs the review exactly as run, for later attention.

Scope run: `/ponytail-review` over the `feat/api-log-scale` branch diff
(`origin/develop..HEAD`, ~631 src lines added: T46ls scale-everywhere + T47mt
`evaluate_correlations`) + targeted repo hot spots (parsers, helpers, tempdir sites).
Tags: `delete:` dead → nothing · `stdlib:` stdlib ships it · `native:` platform/dep does it ·
`yagni:` unused flexibility · `shrink:` same logic, fewer lines. Nothing applied.

## Findings (as delivered)

- `api/_session.py:425-535: shrink: "Distributions must cover variables exactly" guard written VERBATIM three times (uncertainty, correlations, sobol). One `_require_exact_distributions()` helper, called 3×.` → **-12**
- `core/wiofiles.py:66-105: delete: module-scope pylibc = ctypes.PyDLL(None) (runs at EVERY import) + main() smoke demo with the same capture/read block pasted twice. Production surface is capture_to_file; demo → scripts/ or nothing.` → **-35**
- `utils/helpers.py:68-90: shrink: get_dakota_version's hasattr branch + nested metadata fallback. importlib.metadata.version("itis-dakota") in one try/except, same answer.` → **-6**
- `api/workflows.py:455-480: shrink: log_names set + build_bounds loop + separate exp-back pass in generate_grid_samples. One dict-comp over scales, exp-back stays.` → **-6**
- `data/funs_data_processing.py:_parse_data: shrink: `data = []` initialized then immediately rebound. return [line.split() for line in Path(file).read_text().splitlines()], 1 expr.` → **-2**

`net: -61 lines possible` (review pass)
on top of the audit's `~-790 lines, -2 deps possible` (backlog A1–A12).

## Checked and left alone (real complexity, not over-complexity)

- `capture_to_file`'s `dup2` + `libc.fflush` hand-roll — captures Dakota wheel C-level stdio; `contextlib.redirect_stdout` cannot.
- 3× `mkdtemp` + conditional `rmtree` — "discard on success, keep on failure" is the documented `workspace` contract; `TemporaryDirectory` would delete failure evidence.
- V45ls required-arg tripwires (`scale_distribution`, `resolve_log_scale`, `input_scales`/`output_scale`) — invariant enforcement, not ceremony.
- Flip matrix + behaviour tests — test evidence is the ponytail minimum; never flagged for deletion.
- 2× MC-prelude duplication (`correlate_manual_uq_samples` vs `propagate_manual_uq_with_uncertainty`) — under the rule of three; the extraction's return tuple would cost more than the copy.

## Follow-up routing

- Cut each B item as its own commit; full suite + repo-wide `ty check` (V41hp) after each.
- B2 note: moving `main()` out of `core/wiofiles.py` also removes the import-time `PyDLL(None)` side effect (that half is correctness-adjacent, worth taking regardless of the rest).
- Structural/architecture review (module depth, seams) is NOT in ponytail's scope → use the
  `improve-codebase-architecture` / `deepen` / `codebase-design` skills (see chat log 2026-09-30).
