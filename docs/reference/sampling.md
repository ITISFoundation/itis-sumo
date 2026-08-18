# Sampling

`itis_sumo.sampling.lhs` and parts of `itis_sumo.data.funs_data_processing`
generate the input points a surrogate is trained or evaluated on.

## Latin Hypercube Sampling (`sampling/lhs.py`)

`lhs(n, k, method=None, iter=None, seed=None)` draws `n` points in `k`
dimensions on the unit hypercube `[0, 1]^k`, stratified so every 1D
projection has exactly one sample per `[i/n, (i+1)/n)` interval. Ported
from a modified pyDOE implementation (originally published for Scilab by
Baudin, Christopoulou, Collette, Martinez — see file header for full
attribution), with an added seed argument for reproducibility.

Selectable variants (`method=`):

| Method | Strategy |
|---|---|
| `"center"` (`_lhscentered`) | each sample placed at the center of its stratum |
| `"maximin"` (`_lhsmaximin`) | of `iter` random LHS draws, keep the one maximizing the minimum pairwise distance |
| `"correlate"` (`_lhscorrelate`) | of `iter` random LHS draws, keep the one minimizing the max off-diagonal correlation |
| `"m"` / MU variant (`_lhsmu`) | alternative construction supporting a target correlation matrix |
| default (`_lhsclassic`) | one random point per stratum, no post-selection |

Every variant takes an explicit `randomstate` (`np.random.RandomState`/
`Generator`) rather than touching `numpy`'s global RNG (`V3er`) — draw the
same `seed` twice, get the same samples back.

**Verified properties** (see [Verification & Validation §
Category A](../verification-validation.md#category-a-sampling-quality-lhs)):
stratification, value range, seed-reproducibility, and that the
`maximin`/`correlate` variants actually improve on the classic baseline by
their respective criteria.

## Other sample generators (`data/funs_data_processing.py`)

- `create_manual_uq_samples` — draws per-variable UQ samples (normal /
  uniform / constant distributions) via a seeded `np.random.Generator`;
  the pure-Python counterpart to Dakota-native UQ propagation, and the one
  actually reachable from the manual-UQ evaluation pathway (see
  [Sensitivity & UQ](sensitivity-uq.md)).
- `create_samples_along_axes` — 1D sweeps holding all but one variable at
  a cut value, for axis-sweep evaluation.
- `create_grid_samples` — full factorial grid across two or more
  variables, for grid evaluation.
