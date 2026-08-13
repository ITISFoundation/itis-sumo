# Verification & Validation Plan — itis-sumo

## Purpose

A comprehensive V&V document that demonstrates the correctness of the MMUX solver by testing the full computational pipeline (sampling → surrogate building → prediction → UQ propagation → MOGA optimization) against known analytical solutions. This serves as:

1. **Bug-finding tool**: results that don't match theory reveal implementation errors
2. **Capability reference**: documents what the techniques can and cannot do
3. **Limitation catalog**: characterizes where assumptions break down

---

## Test Functions (Escalating Complexity)

| ID | Function | Domain | Formula | Why |
|----|----------|--------|---------|-----|
| F1 | Constant | x ∈ [0, 1] | `f(x) = c` | Trivial baseline — GP must predict flat |
| F2 | Linear | x ∈ [0, 1] | `f(x) = a·x + b` | Exact interpolation; analytically tractable UQ |
| F3 | Quadratic | x ∈ [0, 1] | `f(x) = a·x² + b·x + c` | Curvature capture; error grows near boundaries |
| F4 | Sinusoidal | x ∈ [0, 2π] | `f(x) = A·sin(ωx + φ)` | Oscillatory — GP struggles without dense sampling |
| F5 | Logarithmic | x ∈ [0.01, 2] | `f(x) = a·ln(x) + b` | Monotonic; unbounded derivative near x=0 |
| F6 | Rosenbrock (2D) | x, y ∈ [-2, 2] | `f(x,y) = (a−x)² + b(y−x²)²` | Non-convex, narrow curved valley |
| F7 | Branin (2D) | x ∈ [−5, 10], y ∈ [0, 15] | Standard Branin function | Multiple global minima — MOGA test |
| F8 | Product of sines (ND) | xᵢ ∈ [0, 1] | `f(x) = Π sin(πxᵢ)` | High-dimensional behavior |

---

## Verification Categories

### Category A: Sampling Quality (LHS)

No Dakota involved. Verifies the LHS implementation itself.

| Test | What | Pass Criteria |
|------|------|---------------|
| A1 | Stratification: each column has exactly one sample per interval `[i/k, (i+1)/k]` | `np.histogram(col, bins=k).min() == 1` and `.max() == 1` |
| A2 | Value range: all values in `[0, 1]` | `0 ≤ lhs[i,j] ≤ 1` |
| A3 | Reproducibility: same seed → same output | `lhs(seed=42) == lhs(seed=42)` |
| A4 | Maximin: min pairwise distance of `_lhsmaximin` ≥ `_lhsclassic` | `dist_maximin >= dist_classic` |
| A5 | Correlation: max off-diagonal correlation of `_lhscorrelate` ≤ `_lhsclassic` | `corr_maximin <= corr_classic` |
| A6 | Marginal uniformity: empirical CDF ≈ uniform for large k | KS test p > 0.05 |

### Category B: Surrogate Model Accuracy

Build GP surrogates on known training data, evaluate on held-out test data.

| Test | Function | Training N | Pass Criteria |
|------|----------|-----------|---------------|
| B1 | Exact interpolation at training points | any | `y_hat ≈ y_train` to machine precision |
| B2 | Linear `f(x) = 2x + 1` | 20 | `RMSE < 1e-6`, `R² ≈ 1.0` |
| B3 | Quadratic `f(x) = x²` | 30 | `RMSE < 1e-4`, `R² > 0.99` |
| B4 | Sinusoidal `f(x) = sin(x)` | 50 | `RMSE < 0.05`, `R² > 0.95` |
| B5 | Logarithmic `f(x) = ln(x)` | 20 | `RMSE < 0.01` |
| B6 | Rosenbrock (2D) | 100 | Error larger than F1-F5; verify uncertainty in valley |
| B7 | Convergence study: `sin(x)` with N = 10, 20, 50, 100, 200 | — | RMSE monotonically decreasing |
| B8 | Variance at training points ≈ 0; variance grows away from training data | — | `std_hat[train] < 1e-6`; `std_hat[far] > std_hat[near]` |

### Category C: Axis Sweep

| Test | Function | What | Pass Criteria |
|------|----------|------|---------------|
| C1 | Linear `f(x,y) = 2x + 3y` | Sweep along x → slope 2; sweep along y → slope 3 | Slope within 5% of analytical |
| C2 | Sinusoidal | Sweep captures oscillation; uncertainty grows at edges | Visual + RMSE < 0.1 |
| C3 | Non-default cut values | Predictions at cut point match function value | `|y_hat(cut) − f(cut)| < 1e-4` |

### Category D: Grid Evaluation

| Test | Function | What | Pass Criteria |
|------|----------|------|---------------|
| D1 | Linear `f(x,y) = x + y` | 2D grid predictions | RMSE < 1e-4 |
| D2 | Quadratic `f(x,y) = x² + y²` | 2D grid predictions | RMSE < 1e-3 |
| D3 | Dimension consistency | `NSAMPLESPERVAR=10`, 2 grid vars → 10×10 output | Output shape = (10, 10) |
| D4 | Fixed non-grid variable | 3 inputs, 2 grid, 1 fixed | Fixed var = cut value in all evaluations |

### Category E: Cross-Validation

| Test | What | Pass Criteria |
|------|------|---------------|
| E1 | CV metrics for well-sampled linear function | RMSE ≈ 0, R² ≈ 1.0 |
| E2 | CV metrics improve with more data (N = 20, 50, 100) | RMSE monotonically decreasing |
| E3 | 95% prediction interval coverage | `~95%` of held-out points within `[y_hat ± 1.96·std_hat]` |
| E4 | Manual vs built-in CV comparison | RMSE within 20% of each other (documents built-in CV bug) |

### Category F: UQ Propagation

| Test | Function | Input Distribution | Analytical Output | Pass Criteria |
|------|----------|--------------------|--------------------|---------------|
| F1 | `f(x) = 2x+1` | `x ~ N(0,1)` | `y ~ N(1, 4)` | `|mean − 1| < 0.1`, `|std − 2| < 0.1` |
| F2 | `f(x) = 3x` | `x ~ U(0,1)` | `y ~ U(0,3)` | `|mean − 1.5| < 0.1`, range ≈ [0, 3] |
| F3 | `f(x) = x²` | `x ~ N(0,1)` | `y ~ χ²(1)` | `|mean − 1| < 0.1`, `|var − 2| < 0.2` |
| F4 | Convergence: `2x+1` | `x ~ N(0,1)` | — | Error decreases with n_samples = 100…10000 |
| F5 | Multi-input `f(x,y) = x+y` | `x ~ N(0,1)`, `y ~ N(0,1)` | `y ~ N(0, 2)` | `|mean| < 0.1`, `|std − √2| < 0.1` |
| F6 | Surrogate uncertainty effect | — | — | `std_propagated > std_analytical` when surrogate uncertain |
| F7 | Mixed distributions `f(x,y) = x+y` | `x ~ U(0,1)`, `y ~ N(1, 0.5)` | — | Reasonable mean ≈ 1.5, std ≈ 0.6–0.7 |

### Category G: MOGA Optimization

| Test | Function | What | Pass Criteria |
|------|----------|------|---------------|
| G1 | `f(x) = (x−0.5)²` | Single-objective min | At least one Pareto point with `x ≈ 0.5`, `f < 0.01` |
| G2 | `f1 = x²`, `f2 = (1−x)²` | Bi-objective | Front spans (0, 1) to (1, 0); all points non-dominated |
| G3 | Convergence: iter = 10, 50, 100 | Front improves with iterations | Front at 100 dominates front at 10 |
| G4 | Bound compliance | — | All Pareto inputs within [lower, upper] |

### Category H: Data Preprocessor

| Test | What | Pass Criteria |
|------|------|---------------|
| H1 | Z-score roundtrip | `inverse_transform(transform(x)) ≈ x` within `1e-10` |
| H2 | Min-max roundtrip | Same |
| H3 | Sign switch + normalization roundtrip | Same |
| H4 | Large dataset roundtrip (1000 rows, 20 vars) | Same |
| H5 | Normalization improves accuracy for `f(x) = 1000x+1` | RMSE_normalized < RMSE_unnormalized |

---

## Verification Document Structure

```
# Verification and Validation Report — itis-sumo

## 1. Introduction
   - Purpose, scope, computational pipeline overview

## 2. Test Functions
   - Table of analytical functions with formulas, domains, expected properties

## 3. Sampling Quality Verification (Category A)
   - LHS stratification, reproducibility, statistical properties
   - Results tables with pass/fail criteria

## 4. Surrogate Model Verification (Category B)
   - Accuracy metrics (RMSE, R²) for each test function
   - Training size convergence study
   - Interpolation vs prediction behavior
   - Variance estimation quality
   - Plots: predicted vs true, uncertainty bands

## 5. Evaluation Pathways Verification (Categories C, D, E)
   - Axis sweep accuracy
   - Grid evaluation accuracy and dimension consistency
   - Cross-validation metric quality

## 6. Uncertainty Quantification Verification (Category F)
   - Output distribution matching analytical solutions
   - Convergence with sample count
   - Multi-input propagation
   - Mixed distribution handling

## 7. Multi-Objective Optimization Verification (Category G)
   - Pareto front correctness
   - Convergence behavior
   - Bound compliance

## 8. Data Preprocessing Verification (Category H)
   - Roundtrip fidelity
   - Effect on surrogate accuracy

## 9. Known Limitations
   - Built-in CV parsing is broken (log_output hardcoded to "")
   - MOGA max_function_evaluations ignored (deprecated)
   - GP interpolation vs approximation tradeoffs
   - Grid reshaping complexity for >2 dimensions

## 10. Summary
   - Table of all test results (pass/fail per category)
   - Recommendations for usage
```

---

## Implementation Plan

| Step | What | Depends on |
|------|------|------------|
| 1 | Create test fixture functions (F1-F8) and analytical solutions | — |
| 2 | Write Category A tests (LHS sampling quality) | — |
| 3 | Write Category B tests (surrogate accuracy) — requires Dakota | Step 1 |
| 4 | Write Category C tests (axis sweep) — requires Dakota | Step 1 |
| 5 | Write Category D tests (grid evaluation) — requires Dakota | Step 1 |
| 6 | Write Category E tests (cross-validation) — requires Dakota | Step 1 |
| 7 | Write Category F tests (UQ propagation) — requires Dakota | Step 1 |
| 8 | Write Category G tests (MOGA) — requires Dakota | Step 1 |
| 9 | Write Category H tests (preprocessor) — pure Python | Step 1 |
| 10 | Run full test suite, collect results | Steps 2-9 |
| 11 | Write verification document with results and plots | Step 10 |

### Test Location

`tests/test_vv_analytical.py`

### Test Count Estimate

| Category | Tests |
|----------|-------|
| A: Sampling quality | 6 |
| B: Surrogate accuracy | 8 |
| C: Axis sweep | 3 |
| D: Grid evaluation | 4 |
| E: Cross-validation | 4 |
| F: UQ propagation | 7 |
| G: MOGA optimization | 4 |
| H: Data preprocessor | 5 |
| **Total** | **~41** |
