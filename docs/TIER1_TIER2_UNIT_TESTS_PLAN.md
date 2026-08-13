# Tier 1, 2 & 3: Tests for itis-sumo

## Scope

Standard code-level tests for itis-sumo, organized around the actual metamodeling
workflow it implements — **LHS sampling → grid/axis sampling → surrogate model building →
cross-validation → uncertainty quantification → (MOGA, deferred)** — rather than by source file.

- **Tier 1 & 2** are pure/near-pure function tests and their mathematical invariants. They run fast,
  need no Dakota execution, and are tagged `@pytest.mark.unit`.
- **Tier 3** is new: it runs the real Dakota solver (not mocked) against small analytical functions
  and checks the outputs match known answers. Tagged `@pytest.mark.analytical`, run as a distinct CI
  step since each test spawns a real Dakota subprocess.

None of this is the V&V report (`docs/VERIFICATION_VALIDATION_PLAN.md`) — that remains a
later, separate deliverable (full analytical test-function suite F1–F8, pass/fail tables, plots,
MOGA). Tier 3 here is a narrow, report-free slice of that same idea, scoped to what's needed now.

---

## Tier 1: Direct Unit Tests (No Dakota Dependency)

### 1. LHS Sampling (`lhs.py`)

**`lhs(n, k, method, seed) -> np.ndarray`**
- Output shape: `(k, n)` — k samples, n dimensions
- Values in `[0, 1]`
- **Stratification**: each column has exactly one sample in each interval `[i/k, (i+1)/k]`
- **Reproducibility**: same seed → identical output
- **Maximin**: `_lhsmaximin` produces larger min pairwise distance than `_lhsclassic`
- **Correlation**: `_lhscorrelate` produces smaller max off-diagonal correlation than `_lhsclassic`
- **Centered**: all values at interval midpoints `(i + 0.5) / k`

### 2. Grid & Axis Sweep Sampling (`funs_data_processing.py`)

**`create_grid_samples(run_dir, grid_vars, input_vars, mins, cut_values, maxs, n_points_per_dimension) -> Path`**
- 2D grid, 3 points per dim → 9 total points
- 3D grid, 2 points per dim → 8 total points
- Non-grid variables fixed at cut values
- Output values within `[min, max]` bounds
- File is written and readable

**`create_samples_along_axes(run_dir, data, input_vars, NSAMPLESPERVAR, cut_values) -> Path`**
- 2 input vars, 10 samples per var → 20 total points
- Sweep variable varies linearly from min to max
- Non-sweep variables fixed at cut value
- Output file dimensions match expected count

### 3. Surrogate Model Building (`funs_create_dakota_conf.py`)

**`add_surrogate_model(training_samples_file) -> str`**
- Contains `gaussian_process surfpack`
- Contains `import_build_points_file` with correct path
- Contains `export_approx_points_file "predictions.dat"`
- Contains `export_approx_variance_file "variances.dat"`
- With `cross_validation_folds` set: contains `cross_validation folds = N` and `metrics`

**`create_sumo_evaluation_conffile(build_file, samples_file, input_variables, output_responses) -> str`**
- Contains all expected blocks: environment, model, method, variables, responses
- Variable count matches `input_variables`
- Response descriptor matches `output_responses`

### 4. Cross-Validation (`funs_create_dakota_conf.py`, `funs_evaluate.py`)

**`create_sumo_crossvalidation_conffile(...) -> str`**
- Contains `cross_validation` block with the requested fold count
- Contains the expected metrics list (`root_mean_squared`, `sum_abs`, `mean_abs`, `max_abs`)

**`create_sumo_manual_crossvalidation_conffile(...) -> str`**
- Contains `import_build_points_file` referencing the training file
- Correctly encodes the provided validation indices (fold-out points)

**`_parse_crossvalidation_outputlogs(log_output, N_CROSS_VALIDATION) -> dict`**
- Valid Dakota CV log output → correct RMSE, MAE, max_abs, sum_abs extraction
- Empty log output → returns empty dict or "No surrogate quality metrics found"
- Malformed log output → handled without crash
- Multiple variables in log → correct per-variable extraction

### 5. Uncertainty Quantification (`funs_data_processing.py`, `funs_create_dakota_conf.py`)

**`create_manual_uq_samples(input_vars, distributions, num_samples, seed) -> dict`**
- Normal distribution: output mean ≈ input mean, output std ≈ input std (large N)
- Uniform distribution: output range ≈ [min, max]
- Constant distribution: all values equal to the constant
- Mixed distributions: correct types per variable

**`get_bounds_uniform_distributions(input_vars, distributions) -> tuple[list, list]`**
- Correct lower/upper bounds from uniform distributions
- Non-uniform distribution → raises ValueError

**`get_bounds_uniform_distribution(var, dist) -> tuple[float, float]`**
- Returns (min, max) for uniform
- Non-uniform → raises ValueError

**`create_uq_propagation_conffile(...)`**
- Contains `normal_uncertain` with correct means and std_deviations
- Contains `lhs` sampling with correct sample count

### 6. Data Filtering / Preprocessing (`funs_data_processing.py`, `data_preprocessor/`)

Supporting utility used across the workflow above, not a workflow stage on its own.

**`_filter_data(df, keep_idxs, filter_N_samples, filter_highest_N) -> pd.DataFrame`**
- `keep_idxs=[0,2,4]` → keeps rows 0, 2, 4
- `filter_N_samples=3` → keeps first 3 rows
- `filter_highest_N=2` → drops the 2 rows with the highest value in the target column
- Invalid index (out of range) → handled gracefully
- Empty filter criteria → returns original

**`DataPreprocessor`** — roundtrip fidelity (see Tier 2, P3/P6)

---

## Tier 2: Property-Based / Invariant Tests

These verify mathematical properties that must hold regardless of specific inputs.

### P1. LHS Invariants
- **Stratification**: for each column, `np.histogram(col, bins=k)` produces exactly 1 count per bin
- **Bounds**: `0 ≤ lhs[i,j] ≤ 1` for all i, j
- **Permutation**: each column is a permutation of its stratified intervals (no two points in the same interval for the same variable)
- **Maximin monotonicity**: `_lhsmaximin` min distance ≥ `_lhsclassic` min distance (for same iter count)

### P2. Grid Sample Invariants
- Total points = `∏ n_points_per_dimension[i]` for grid variables
- All grid variable values within `[min, max]`
- Non-grid variable values exactly equal to cut values
- For 2D grid: `np.meshgrid` consistency (row/column ordering matches reshaped output)

### P3. Axis Sweep Invariants
- Total points = `NSAMPLESPERVAR × len(input_vars)`
- Sweep variable varies monotonically from min to max
- Non-sweep variables constant at cut value
- Sweep spacing is uniform (equal differences between consecutive points)

### P4. DataPreprocessor Roundtrip
- `inverse_transform(transform(df)) ≈ df` for any valid config (z-score or min-max)
- After z-score transform: column means ≈ 0, column stds ≈ 1
- After min-max transform: column ranges = [0, 1]
- With sign switching: roundtrip still exact

### P5. Normalization Invariants
- Z-score: `(x - mean) / std` produces mean ≈ 0, std ≈ 1 across a representative sample
- Min-max: `(x - min) / (max - min)` produces range [0, 1]
- Sign flip: negating a variable and then negating again returns original

---

## Tier 3: Analytical Integration Tests (Real Dakota)

Unlike Tiers 1–2, these run the **real** Dakota solver (`itis-dakota`, no mocking) end-to-end against
small analytical functions with known closed-form answers, and assert the outputs match. This is the
"give it inputs, check the outputs are what we expect" layer — a narrow, report-free precursor to the
full `VERIFICATION_VALIDATION_PLAN.md` (which remains the later, separate deliverable: F1–F8 test
functions, Categories A–H, pass/fail tables, plots, MOGA).

Location: `tests/test_metamodeling_analytical.py`. All tests tagged `@pytest.mark.analytical`
and excluded from the default `pytest -m "not analytical"` run (see Implementation) since each spawns a real
Dakota subprocess.

**Implementation note:** `add_surrogate_model` only omits the `eval_id` tabular column when the
training file path contains `"processed"` (`funs_create_dakota_conf.py:180`) — fixture files must be
named e.g. `train_processed.txt`, matching the convention `funs_data_processing.py` already uses.

Test IDs below mirror the categories/IDs in `VERIFICATION_VALIDATION_PLAN.md` (Categories B–F), so
that document's Test Functions table and this one stay traceable to each other. Category A (LHS) and
H (DataPreprocessor) are pure-Python and already covered by Tier 1/2; Category G (MOGA) is deferred
(see below).

**Category F implementation note:** `propagate_uq`/`create_uq_propagation_conffile` (Dakota-native
UQ, normal-uncertain only) is **not called from any Flask route** — `blueprints/dakota.py` only
imports `evaluate_sumo*` functions from `funs_evaluate.py` (historical context from the original
mmux_vite Flask app; itis-sumo has no Flask layer). The real, user-reachable UQ pathway is
`POST /manual_uq_propagation_with_uncertainty`, which composes `create_manual_uq_samples`
(`funs_data_processing.py` — supports `normal`/`uniform`/`constant` per-variable distributions) with
`evaluate_sumo` and an erfinv-based predictive-uncertainty injection. Category F below tests *that*
pathway, which is why F2 (uniform input) and F7 (mixed normal+uniform inputs) are included rather than
excluded — the earlier assumption that non-normal inputs were unsupported was based on the wrong
(unreachable) function.

**Category B — Surrogate Model Accuracy** (`evaluate_sumo`)

| ID | Case | Pass criteria |
|----|------|----------------|
| B1 | Exact interpolation at training points, `f(x) = 2x+1` | `y_hat ≈ y_train` (tight tolerance) |
| B2 | Linear `f(x) = 2x + 1` | RMSE < 1e-2, R² > 0.99 |
| B3 | Quadratic `f(x) = x²` | RMSE < 1e-2, R² > 0.99 |
| B4 | Sinusoidal `f(x) = sin(x)` on `[0, 2π]` | RMSE < 0.05, R² > 0.95 |
| B5 | Logarithmic `f(x) = ln(x)` on `[0.01, 2]` | RMSE < 0.01 (eval points kept away from the boundary singularity — see test docstring) |
| B6 | Rosenbrock (2D), `N=100` | finite, non-trivial predictive std reported; normalized RMSE documents it's harder than the smooth cases above |
| B7 | Convergence: `sin(x)` over 3 periods, `N = 6, 15, 60` | RMSE strictly decreasing |
| B8 | Predictive variance | ≈0 at training points; larger when extrapolating far outside the training domain |

**Category C — Axis Sweep** (`evaluate_sumo_along_axes`)

| ID | Case | Pass criteria |
|----|------|----------------|
| C1 | Linear `f(x,y) = 2x + 3y` | sweep-along-x slope ≈ 2, sweep-along-y slope ≈ 3 (within 5%) |
| C2 | `f(x,y) = sin(x) + y` | sweep along x (y fixed) RMSE < 0.1 vs analytical |
| C3 | Non-default cut value | sweep predictions match `f(x, cut_y)` elementwise |

**Category D — Grid Evaluation** (`evaluate_sumo_on_grid`)

| ID | Case | Pass criteria |
|----|------|----------------|
| D1 | Linear `f(x,y) = x + y` | sorted predicted values match sorted analytical values (see test docstring re: known grid reshape ordering quirk) |
| D2 | Quadratic `f(x,y) = x² + y²` | same sorted-set comparison, RMSE < 0.05 |
| D3 | Dimension consistency | `NSAMPLESPERVAR=10`, 2 grid vars → output shape `(10, 10)` |
| D4 | Fixed non-grid variable | 3 inputs, 2 grid + 1 fixed; fixed var equals its cut value in all evaluations |
| D5 | Asymmetric domain/function `f(x,y) = 2x + 5y` | elementwise match against `results[response][j][i] == f(x_lin[i], y_lin[j])` — resolves the reshape's own `# ... Why???` comment empirically; D1/D2's sorted-set comparisons can't catch a transpose bug because their symmetric domains/functions make it invisible |

**Category E — Cross-Validation** (`evaluate_sumo_crossvalidation`, `evaluate_sumo_manual_crossvalidation`)

| ID | Case | Pass criteria |
|----|------|----------------|
| E1 | Well-sampled linear `f(x) = 2x+1` | parsed RMSE < 0.1 |
| E2 | Convergence: `sin(x)`, `N = 20, 50, 100` | RMSE(100) < RMSE(20) |
| E3 | 95% prediction interval coverage (manual CV) | ≥ 70% of held-out points within `y_hat ± 1.96·std_hat` (generous lower bound — small-N regime) |
| E4 | Manual vs built-in CV agreement | both RMSE < 0.1 for the same well-sampled linear case |

**Category F — UQ Propagation** (`create_manual_uq_samples` + `evaluate_sumo` + erfinv injection,
mirroring `flask_manual_uq_propagation_with_uncertainty` — see implementation note above)

| ID | Case | Pass criteria |
|----|------|----------------|
| F1 | `f(x) = 2x+1`, `x ~ N(0,1)` | mean ≈ 1, std ≈ 2 (within tolerance) |
| F2 | `f(x) = 2x+1`, `x ~ U(-3,3)` | mean ≈ 1, std ≈ √12 (within tolerance) |
| F3 | `f(x) = x²`, `x ~ N(0,1)` (→ χ²(1)) | mean ≈ 1, var ≈ 2 (within tolerance) |
| F5 | Multi-input `f(x,y) = x+y`, `x,y ~ N(0,1)` iid | mean ≈ 0, std ≈ √2 |
| F6 | Undersampled oscillation (`sin(x)`, `N=5`) | documents a real miscalibration: RMSE > 0.2 while reported `std_hat` stays < 0.01 — undersampling here shows up as a biased/damped fit with a deceptively tight confidence band, not as legitimately wider ones (see test docstring for the two failed "sparse ⇒ wider" assumptions this replaced) |
| F7 | Mixed `f(x,y) = x+y`, `x ~ N(0,1)`, `y ~ U(-1,1)` | mean ≈ 0, std ≈ √(4/3) |

MOGA (Category G equivalent) is out of scope for this tier — see Deferred section below.

---

## Deferred — MOGA / Pareto (not this round)

MOGA is lower priority for now. The following already exist and keep passing, but are grouped
separately (bottom of their respective test files, under a `DEFERRED` banner) to signal they're not
part of the current tailoring effort. No new work is planned here until MOGA is picked back up.

**`is_dominated(point, other_points) -> bool`** (`funs_data_processing.py`)
- Dominated / not dominated / self-dominance (reflexive, since the implementation uses `>=`) / multi-objective cases

**`get_non_dominated_indices(data, optimized_vars, optimization_modes, sort_by_column) -> list[int]`**
- Known 2D Pareto front, maximization via sign-flip, all-points-on-front, single-point, mixed modes

**`create_moga_optimization_conffile(...)` / `add_moga_method(...)`** (`funs_create_dakota_conf.py`)
- Contains `moga` method with correct parameters and variable bounds

**Tier 2 — Pareto Dominance Invariants (P1, formerly)**
- Reflexivity, asymmetry, transitivity, non-dominated-set completeness/no-internal-domination, max-mode correctness

When MOGA work resumes, extend Tier 3 with a Category-G-equivalent (Pareto front correctness,
convergence, bound compliance) reusing `VERIFICATION_VALIDATION_PLAN.md`'s G1–G4.

---

## Implementation

- **Framework**: pytest (already configured in `pyproject.toml`)
- **Locations**:
  - Tier 1: `tests/test_unit_solver.py`
  - Tier 2: `tests/test_property_invariants.py`
  - Tier 3: `tests/test_metamodeling_analytical.py`
- **Fixtures**: use `tmp_path` for file I/O tests; no Dakota mocking needed anywhere (Tier 1/2 don't
  touch Dakota at all; Tier 3 deliberately runs the real solver)
- **Parametrize**: use `@pytest.mark.parametrize` for multi-case tests
- **Markers**: `@pytest.mark.unit` for Tier 1/2, `@pytest.mark.analytical` for Tier 3
- **Running**: `pytest -m "not analytical"` runs Tier 1/2 (fast, default); `pytest -m analytical`
  runs Tier 3 separately since each test spawns a real Dakota subprocess

## Test Count Estimate

| Category | Tests |
|----------|-------|
| LHS sampling | ~10 |
| Grid & sweep sampling | ~8 |
| Surrogate model building | ~6 |
| Cross-validation (config + parsing) | ~9 |
| Uncertainty quantification | ~9 |
| Data filtering / preprocessing | ~9 |
| Property invariants (P1–P5) | ~19 |
| Deferred — MOGA / Pareto | ~14 |
| **Tier 1+2 total** | **~84** |
| Tier 3 — analytical integration (Categories B–F) | ~26 |
