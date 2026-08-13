# itis-sumo

Headless core of the SuMo (**Su**rrogate **Mo**del) meta-modeling tools: build,
evaluate, cross-validate, and interrogate surrogate models — UQ (Sobol'
indices, uncertainty propagation), sampling (LHS / grid / manual-UQ), and
MOGA optimization — as a standalone, importable Python package.

## Install & verify

```sh
uv sync                          # Python 3.11, resolves itis-dakota==1.5.9
uv run itis-sumo validate        # engine probe (expect: Version 1.5.9)
```

## Where to go

- **New here?** [Getting started](tutorials/getting-started.md) — build,
  cross-validate, and interrogate one surrogate end to end, real runnable
  code throughout.
- **Need to do something specific?** [How-to guides](how-to/cross-validate.md) —
  task recipes for cross-validation, sensitivity/UQ, MOGA optimization, and
  data preprocessing.
- **Want the why, not just the how?** [Explanation](explanation/surrogate-modeling.md) —
  why surrogate modeling, how Gaussian Processes work, worked examples.
- **Looking up an exact function signature?** [Reference](reference/index.md) —
  module-by-module API and the design invariants behind it.
- **Curious about the engine pin, fork provenance, or how this fits into
  MMUX?** [About](about/index.md).

Evidence the pipeline reproduces known analytical solutions, not just
"didn't crash": [Verification & Validation report](verification-validation.md).
