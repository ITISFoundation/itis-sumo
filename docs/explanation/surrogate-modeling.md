# Why surrogate modeling

This page is background theory, not itis-sumo usage docs — for how the pipeline
is implemented, see [Reference](../reference/index.md). For a from-scratch,
runnable walkthrough, see [Worked examples](../tutorials/examples.md).

\[
\newcommand{\R}{\mathbb{R}}
\]

## The problem: simulations are expensive, design spaces are large

A physics/engineering simulator (a Dakota "interface" in this codebase's
vocabulary — FEM solver, circuit model, bioelectronic field solve, whatever
produces one number per parameter set) computes

\[
y = f(\mathbf{x}), \qquad \mathbf{x} \in \R^d
\]

where \(f\) is a black box: no closed form, evaluating it costs real wall-clock
time (seconds to hours) and often real money (compute cluster time). Design,
optimization, and uncertainty analysis all require evaluating \(f\) at *many*
points — a systematic optimizer might need thousands of evaluations to converge;
a robustness study propagating input uncertainty through \(f\) needs enough
samples to characterize an output distribution, not just a single number.
Running the real simulator that many times is frequently not tractable.

**Surrogate modeling** (also called *metamodeling* or *response surface
modeling*) sidesteps this: run the expensive \(f\) a comparatively small number
of times (tens to low hundreds — this is the *training set*, produced via
[design-of-experiments sampling](../reference/sampling.md)), fit a cheap
approximation \(\hat{f}(\mathbf{x}) \approx f(\mathbf{x})\) to those observations,
then run everything downstream — optimization
([MOGA](../reference/moga.md)), sensitivity analysis
([Sobol'](../reference/sensitivity-uq.md)), uncertainty propagation — against
\(\hat{f}\), which costs microseconds per evaluation instead of minutes.

```
expensive f(x)              cheap  f̂(x)
  (n_train evals)    ──▶    (n_query evals, n_query ≫ n_train)
  "ground truth"             "surrogate model"
```

The entire value proposition rests on one assumption: **\(\hat{f}\) generalizes
well enough between training points that decisions made against it match
decisions that would have been made against \(f\) itself.** Everything in
itis-sumo's [verification & validation report](../verification-validation.md)
is ultimately testing that assumption on functions where the right answer is
known analytically.

## What makes a function a good (or bad) surrogate target

Surrogate accuracy is not free — it depends on how \(f\) behaves and how many
training points you can afford:

- **Smoothness.** Surrogates interpolate/extrapolate by assuming nearby inputs
  produce nearby outputs. A smooth, slowly-varying \(f\) (e.g. a linear or
  quadratic response) needs few training points to fit near-exactly. A highly
  oscillatory or discontinuous \(f\) needs many more points to avoid aliasing —
  itis-sumo's V&V suite demonstrates this directly: linear/quadratic test
  functions fit to R² > 0.99 with 20–30 points, while a sinusoid needs roughly
  double the training density for a comparable (looser) tolerance, and the
  harder 2D Rosenbrock function needs looser tolerances still even with more
  points (see [Category B](../verification-validation.md#category-b-surrogate-model-accuracy)).
- **Dimensionality.** The volume of the input space grows exponentially with
  \(d\), but a fixed training budget doesn't — this is the *curse of
  dimensionality*. The same 30 training points that densely cover a 1D interval
  cover almost nothing of a 10D hypercube. Surrogate modeling doesn't repeal
  this law; it just moves the sample-efficiency problem from "cost per
  simulator call" to "cost per training point," which is why *how* those
  training points are chosen ([Latin Hypercube sampling](../reference/sampling.md),
  not grid or plain-random) matters so much at fixed budget.
- **Noise.** If \(f\) itself is stochastic (Monte Carlo simulators, measurement
  noise), the surrogate is fitting a moving target — Gaussian Process
  regression (below) handles this gracefully via an explicit noise/nugget term,
  but the achievable accuracy is bounded by the noise floor no matter how many
  training points you add.

## Why not just fit a polynomial?

Classical response-surface methodology historically used low-order polynomial
regression (quadratic response surfaces). This still shows up (e.g. as one
term in polynomial-chaos UQ methods) but has a structural weakness for
engineering design work: a plain least-squares polynomial fit gives you a
point prediction and nothing else — no notion of *how much to trust* that
prediction at a given query point. A polynomial extrapolating wildly into a
region with zero nearby training data looks exactly as confident as one
interpolating between two nearby points. For design decisions and — especially
— uncertainty quantification, that missing confidence signal is the whole
problem. This is the direct motivation for the next page:
[Gaussian Processes](gaussian-processes.md), the surrogate family itis-sumo
uses, which produce a predictive *distribution* (mean **and** variance) at
every query point instead of a bare number.

## The recurring accuracy questions

Whatever surrogate family is used, three questions have to be answered before
trusting it for a real decision, and each maps directly onto part of the
itis-sumo pipeline:

1. **Does it reproduce the training data?** (sanity floor — see
   [Category B](../verification-validation.md#category-b-surrogate-model-accuracy))
2. **Does it generalize to unseen points?** — answered via
   [cross-validation](../reference/evaluate.md#cross-validation):
   hold out a fold of training data, fit on the rest, check prediction error on
   the held-out fold. itis-sumo runs both Dakota-native and a manual K-fold CV
   pathway (the latter is the one whose numbers are independently verified —
   see [Categories C–E](../verification-validation.md#evaluation-pathways-categories-ce)).
3. **Is its stated uncertainty calibrated?** — a surrogate that's *wrong* is
   recoverable if it *knows* it's wrong (wide predictive interval); a
   surrogate that's confidently wrong is dangerous. This is checked via
   prediction-interval coverage tests (does the true value fall inside the
   surrogate's claimed 95% interval roughly 95% of the time?) — see
   [Category F](../verification-validation.md#category-f-uq-propagation) — and
   is the reason GPs, which report variance natively, are structurally better
   suited to this whole workflow than point-prediction-only methods.

See [Worked examples](../tutorials/examples.md) for a from-scratch, runnable demonstration
of a surrogate fit and its uncertainty band on a known function.
