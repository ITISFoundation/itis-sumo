# Algorithms overview

itis-sumo wraps the [Dakota](https://dakota.sandia.gov/) 6.20 engine (via the
`itis-dakota==1.5.9` PyPI wheel) behind a Python API. Every "algorithm" here
is really a **NIDR config composer** (build the Dakota input block as a
string) paired with a **run + parse** step (execute the wheel, read back its
tabular/results output as a DataFrame). No Dakota binary is ever shelled
out to — everything runs in-process through `dakota.environment.study`.

## Pipeline shape

<div class="tip-workflow">
  <a class="tip-workflow__step twf-blue" href="sampling.md">Sampling</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-indigo" href="config.md">Config</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-orange" href="core.md">Core</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-teal" href="evaluate.md">Evaluate</a>
</div>

Two capabilities sit alongside that straight-line pipeline:

<div class="tip-workflow">
  <a class="tip-workflow__step twf-rose" href="sensitivity-uq.md">Sensitivity (Sobol) &amp; UQ</a>
  <span class="tip-workflow__arrow">·</span>
  <a class="tip-workflow__step twf-amber" href="preprocess.md">Preprocessing</a>
</div>

1. **[Sampling](sampling.md)** — draw design points (Latin Hypercube, grid,
   manual-UQ) or take user-supplied training data.
2. **[Config](config.md)** — compose the Dakota NIDR input string for the
   requested study type (surrogate fit, cross-validation, MOGA, UQ
   propagation).
3. **[Core](core.md)** — hand that string to `dakota.environment.study(...)`
   in a `ProcessPoolExecutor` worker, capture stdout/stderr, return the
   run directory.
4. **[Evaluate](evaluate.md)** — parse the run's `.dat`/tabular output back
   into a DataFrame, compute derived quantities (RMSE, R², prediction
   intervals, Pareto fronts, Sobol indices).

Two capabilities not part of that straight-line pipeline:

- **[Sensitivity (Sobol) & UQ propagation](sensitivity-uq.md)** — sits on
  top of `evaluate_sumo`: draws Saltelli/QMC sample sets or per-variable UQ
  distributions in pure Python, evaluates the already-built surrogate on
  them, and post-processes with `scipy.stats`.
- **[Data preprocessing](preprocess.md)** — an optional layer
  (`DataPreprocessor`) that normalizes/renames variables before training
  and inverse-transforms predictions after, independent of which study type
  ran.

!!! warning "Design invariants"
    These are enforced by the test suite (see [`SPEC.md`](../about/spec.md) §V for
    the full, authoritative list):

    - **No global RNG state** — every sampling function takes an explicit seed
      (`np.random.Generator`), never reaches into `numpy`'s or `scipy`'s global
      random state (`V3er`).
    - **No flask/oSPARC imports in core modules** — `core`, `config`, `data`,
      `sampling`, `evaluate` have zero web-framework dependencies; test-enforced
      (`V4ty`). Only `preprocess` carries the pydantic job models from the web layer.
    - **Run directories are explicit paths** — the only implicit `os.chdir` is
      scoped to the Dakota worker process via a context-manager guard, never
      left dangling on the caller's process (`V2qw`).
    - **Variable names preserve literal `-`** — `sanitize_varnames` treats
      hyphens as legal characters rather than substitution targets (`V6op`).
    - **Malformed rows are healed-or-dropped with a trace, never silently
      wiped** (`V7as`).
