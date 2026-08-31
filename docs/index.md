---
hide:
  - navigation
  - toc
---

# itis-sumo

## Headless surrogate-modeling core for design-space exploration

Running complex simulations across design spaces is slow and expensive.
itis-sumo lets you train a fast surrogate model that captures the
simulation's behavior — so you can sweep parameters, optimize designs, and
quantify sensitivity with honest uncertainty bars.

[Quick Start :material-rocket-launch:](tutorials/getting-started.md){ .md-button .md-button--primary }
[Workflows :material-sitemap:](how-to/cross-validate.md){ .md-button }

---

![A trained surrogate (blue) with 95% prediction-interval uncertainty bands
tracking the true response (black).](assets/examples/gp_fit_uncertainty.png)

**Scope:** itis-sumo is a headless computational core with no Flask, oSPARC,
or web-UI dependency. For the web UI, see the separate `mmux_documentation`
site.

---

## What you can do

<div class="grid cards" markdown>

- :material-flask-outline: __Train surrogates__
  Build GP / polynomial surrogates on simulation data — in-process via the
  Dakota wheel, no shell-outs.

- :material-chart-line: __Quantify sensitivity__
  Sobol' indices and UQ propagation with `scipy.stats`, on top of your
  already-built surrogate.

- :material-check-all: __Cross-validate__
  Rigorous surrogate evaluation with explicit-seed sampling and honest
  error metrics (RMSE, R²).

- :material-dna: __Multi-objective optimization__
  Find Pareto fronts with MOGA without leaving the Python API.

- :material-filter: __Preprocess__
  Normalize / rename training variables before fit and inverse-transform
  after, via `DataPreprocessor`.

- :material-book-open-page-variant: __Worked examples__
  Real Dakota fits validated against closed-form analytical solutions, with
  figures.

</div>

---

**Verification & Validation:** the full pipeline is verified against known
analytical solutions — available on request: [current report](verification-validation.md).
