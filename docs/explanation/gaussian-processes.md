# How Gaussian Processes work (and why itis-sumo uses them)

Background theory — see [Reference → Config](../reference/config.md) for how
the `gaussian_process surfpack` block is actually composed and shipped to
Dakota, and [Worked examples](../tutorials/examples.md) for a runnable fit.

## What "Gaussian Process" means

A Gaussian Process (GP) is a way of putting a probability distribution
**directly over functions**, rather than over a fixed set of parameters
(the way "fit a line: find slope and intercept" does). Formally, a GP is
defined by a mean function \(m(\mathbf{x})\) (almost always taken as zero
after centering the data) and a covariance function — the **kernel**
\(k(\mathbf{x}, \mathbf{x}')\) — such that for *any* finite set of input points
\(\mathbf{x}_1, \ldots, \mathbf{x}_n\), the corresponding function values
\(f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)\) are jointly Gaussian:

\[
\begin{pmatrix} f(\mathbf{x}_1) \\ \vdots \\ f(\mathbf{x}_n) \end{pmatrix}
\sim \mathcal{N}(\mathbf{0}, K), \qquad K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)
\]

The kernel is doing all the work: it encodes the assumption "inputs that are
close together (in whatever metric the kernel uses) produce outputs that are
close together." The most common choice, and the one relevant here, is the
squared-exponential / RBF kernel:

\[
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\!\left(-\frac{1}{2}
\sum_{i=1}^d \frac{(x_i - x_i')^2}{\ell_i^2}\right)
\]

- \(\sigma_f^2\) — signal variance: how much the function is expected to vary
  overall.
- \(\ell_i\) — per-dimension **length scale**: how far you have to move along
  input dimension \(i\) before the function value is expected to change
  appreciably. A short length scale means a wiggly function needing dense
  sampling to pin down (matches the sinusoid-needs-more-points observation on
  the [surrogate modeling](surrogate-modeling.md) page); a long length scale
  means a smooth, slowly-varying function that a few points characterize well.

## Fitting: prior → posterior, conditioned on training data

Before seeing any data, the GP prior says "any function this smooth is
equally plausible." Conditioning on the training set
\(\{(\mathbf{x}_i, y_i)\}_{i=1}^n\) — using the joint-Gaussian conditioning
identity — collapses that prior down to a **posterior distribution over
functions that pass through (or near, if there's noise) the training points**.
At a new query point \(\mathbf{x}_*\), the posterior predictive distribution is
itself Gaussian, with closed-form mean and variance:

\[
\mu(\mathbf{x}_*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}
\]

\[
\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
\]

where \(\mathbf{k}_*\) is the vector of kernel values between \(\mathbf{x}_*\)
and every training point, \(K\) is the training-point kernel (Gram) matrix, and
\(\sigma_n^2\) is an observation-noise term (zero, or near-zero, for a
deterministic simulator; nonzero if the training data itself is noisy). This
is exactly the shape of every [worked example](../tutorials/examples.md) plot on this
site: a mean curve \(\mu(\mathbf{x})\) plus a shaded band from
\(\sigma(\mathbf{x})\) — narrow right at training points (where the kernel
"recognizes" the input as close to something already seen) and widening in
between and especially outside the training envelope.

**Hyperparameter fitting.** \(\sigma_f^2\), the length scales \(\ell_i\), and
\(\sigma_n^2\) are not chosen by hand — they're fit by maximizing the marginal
likelihood of the observed training data under the GP prior (equivalently,
minimizing its negative log):

\[
\log p(\mathbf{y} \mid X) = -\frac{1}{2}\mathbf{y}^\top (K+\sigma_n^2 I)^{-1}\mathbf{y}
- \frac{1}{2}\log|K+\sigma_n^2 I| - \frac{n}{2}\log 2\pi
\]

This objective self-balances fit quality (first term — how well does this
kernel explain the observed \(\mathbf{y}\)) against complexity (second term —
penalizes kernels that are needlessly wiggly/informative), which is why GPs
don't typically need a separate regularization hyperparameter tuned by hand
the way, say, ridge regression does. In itis-sumo, this whole fitting step
happens inside the Dakota/Surfpack `gaussian_process surfpack` engine — see
[Reference → Config](../reference/config.md) for the NIDR block that
requests it; itis-sumo itself never touches the kernel machinery directly, it
only supplies training data and reads back predictions.

## Why GPs specifically (vs. the alternatives)

| Alternative | What it gives you | What it's missing for this use case |
|---|---|---|
| Polynomial response surface | Point prediction, closed form, cheap to fit | No native uncertainty — see [surrogate modeling](surrogate-modeling.md#why-not-just-fit-a-polynomial); fixed global shape (degree) can't locally adapt to a function that's smooth in one region and steep in another |
| Radial basis function (RBF) network / plain kriging without a probabilistic layer | Flexible, local, interpolates well | Length-scale / kernel-width choice usually needs manual tuning or a separate cross-validation loop instead of falling out of one coherent maximum-likelihood fit; no principled variance unless a Gaussian-process interpretation is added back in anyway |
| Neural network surrogate | Very flexible, scales to large \(n\) | Needs orders of magnitude more training data than a GP to fit reliably (the whole point here is training points are *expensive*); predictive uncertainty requires extra machinery (ensembles, MC-dropout, deep kernel learning) that isn't native to the base model |
| **Gaussian Process** | Point prediction **and** calibrated variance, exact interpolation at training points (for \(\sigma_n^2 \to 0\)), competitive with a training budget of tens–low-hundreds of points, hyperparameters fit by one coherent likelihood-maximization step | Naive GP training cost is \(O(n^3)\) (Gram-matrix inversion) — irrelevant here since \(n\) is the *small* number (training set), not the *large* one (query set); \(n^3\) on 30–300 points is trivial |

The decisive reason for this codebase specifically: **every downstream
consumer of the surrogate needs the variance, not just the mean.**

- [Cross-validation](../reference/evaluate.md#cross-validation)
  metrics (RMSE, R²) only need the mean — but
- Prediction-interval coverage checks
  ([Category F](../verification-validation.md#category-f-uq-propagation))
  need \(\sigma(\mathbf{x})\) to construct the interval in the first place.
- The manual UQ-propagation pathway
  ([Reference → Evaluate](../reference/evaluate.md#uncertainty-propagation))
  explicitly injects each prediction's own \(\hat\sigma\) via an erfinv
  transform before propagating — this only works because the surrogate
  reports per-point variance natively (`V8df` in [SPEC.md](../about/spec.md) — the
  `{output}_std_hat` column comes straight from the GP posterior, not a
  side-channel estimate).
- MOGA optimization ([Reference → MOGA](../reference/moga.md)) over a
  surrogate is only trustworthy in regions the GP is actually confident about
  — a design that looks optimal purely because the surrogate is
  under-constrained there (wide \(\sigma\)) is a modeling artifact, not a real
  optimum, and \(\sigma(\mathbf{x})\) is exactly the signal that would catch
  that.

None of the alternatives in the table above give you that variance signal for
free — a GP does, which is why it's the one surrogate family used throughout
this pipeline (Dakota/Surfpack's `gaussian_process` model type — see
`V14nm` in [SPEC.md](../about/spec.md): itis-sumo deliberately avoids also vendoring
a standalone `surfpack` dependency, relying on the wheel's built-in GP path
instead).

## Where the theory shows up as a testable claim

Every abstract point above corresponds to something itis-sumo's test suite
actually checks against a real (unmocked) Dakota GP fit, not just asserts in
prose:

- *"Narrow uncertainty at training points, wide between them"* →
  `test_variance_near_zero_at_training_points_grows_away`
  ([Category B](../verification-validation.md#category-b-surrogate-model-accuracy)).
- *"More training points ⇒ better fit"* →
  `test_convergence_with_training_size` /
  `test_crossvalidation_rmse_improves_with_more_data`.
- *"Stated uncertainty is calibrated, not just present"* →
  the prediction-interval coverage tests under
  [Category F](../verification-validation.md#category-f-uq-propagation).
- *"Sparse sampling under-covers a genuinely wiggly function"* → the
  deliberately-adversarial `test_undersampled_oscillation_uncertainty_is_underestimated`
  — a case where the GP's own calibration breaks down because the training
  design didn't resolve the function's actual length scale, included precisely
  so this isn't overstated as "GPs are always well-calibrated."

See [Worked examples](../tutorials/examples.md) for the code and figures.
