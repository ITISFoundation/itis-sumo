"""End-to-end tests for the consumer API, against a real Dakota surrogate.

Nothing here is mocked: each test builds an actual surrogate and reads back what
Dakota produced. What is asserted is the promise the API makes -- results arrive
in the caller's own units under the caller's own column names -- rather than
particular numbers.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from itis_sumo.api import (
    DistributionSpec,
    DomainSpec,
    PreprocessingSpec,
    SumoInputError,
    VariableSpec,
    compute_correlations,
    cross_validate,
    evaluate_along_axes,
    evaluate_correlations,
    evaluate_cv_metrics,
    evaluate_grid,
    evaluate_sobol,
    evaluate_uncertainty,
    generate_grid_samples,
    generate_lhs_samples,
    optimize,
)

pytestmark = pytest.mark.integration

VARIABLES = ["width", "height"]
RESPONSE = "stress"

# Deliberately mismatched magnitudes: if anything leaked out in internal units,
# a height around 300 would come back looking nothing like a height.
WIDTH_RANGE = (1.0, 5.0)
HEIGHT_RANGE = (100.0, 500.0)


@pytest.fixture(scope="module")
def samples() -> pd.DataFrame:
    rng = np.random.default_rng(20260818)
    width = rng.uniform(*WIDTH_RANGE, 20)
    height = rng.uniform(*HEIGHT_RANGE, 20)
    return pd.DataFrame(
        {"width": width, "height": height, "stress": 3.0 * width + 0.01 * height}
    )


class TestCrossValidation:
    def test_predicts_every_sample_in_original_units(self, samples):
        result = cross_validate(samples, VARIABLES, RESPONSE)

        assert result.response == RESPONSE
        assert len(result.predicted) == len(samples)
        assert result.observed == pytest.approx(samples[RESPONSE].tolist())

        predicted = [value for value in result.predicted if not math.isnan(value)]
        assert predicted, "no fold produced a prediction"
        assert min(predicted) > 0.0
        assert max(predicted) < 10 * samples[RESPONSE].max()

    def test_echoes_the_seed_it_used(self, samples):
        assert cross_validate(samples, VARIABLES, RESPONSE).seed == 42
        assert cross_validate(samples, VARIABLES, RESPONSE, seed=7).seed == 7

    def test_is_reproducible_for_a_given_seed(self, samples):
        first = cross_validate(samples, VARIABLES, RESPONSE, seed=7)
        second = cross_validate(samples, VARIABLES, RESPONSE, seed=7)
        assert first.predicted == pytest.approx(second.predicted, nan_ok=True)

    def test_result_is_json_serializable(self, samples):
        result = cross_validate(samples, VARIABLES, RESPONSE)
        payload = dataclasses.asdict(result)
        # NaN is valid in Python's JSON dialect; the point is that nothing in the
        # structure is a type json cannot reach.
        assert json.loads(json.dumps(payload))["response"] == RESPONSE


class TestAlongAxes:
    def test_sweeps_each_variable_over_its_observed_range(self, samples):
        result = evaluate_along_axes(
            samples, VARIABLES, RESPONSE, points_per_variable=9
        )

        assert set(result.sweeps) == set(VARIABLES)
        for variable, sweep in result.sweeps.items():
            assert sweep.variable == variable
            assert len(sweep.x) == len(sweep.predicted)
            observed = samples[variable]
            assert min(sweep.x) >= observed.min() - abs(observed.min())
            assert max(sweep.x) <= observed.max() + abs(observed.max())

    def test_keeps_variables_in_their_own_units(self, samples):
        """A height must come back looking like a height, not like an x2."""
        result = evaluate_along_axes(
            samples, VARIABLES, RESPONSE, points_per_variable=9
        )
        heights = result.sweeps["height"].x
        assert min(heights) > WIDTH_RANGE[1]
        assert max(heights) <= HEIGHT_RANGE[1] * 1.01

    def test_honours_the_values_the_caller_holds_fixed(self, samples):
        low = evaluate_along_axes(
            samples, VARIABLES, RESPONSE, at={"height": 120.0}, points_per_variable=9
        )
        high = evaluate_along_axes(
            samples, VARIABLES, RESPONSE, at={"height": 480.0}, points_per_variable=9
        )
        # stress rises with height, so holding height higher must lift the width sweep
        assert sum(high.sweeps["width"].predicted) > sum(low.sweeps["width"].predicted)

    def test_result_is_json_serializable(self, samples):
        result = evaluate_along_axes(
            samples, VARIABLES, RESPONSE, points_per_variable=5
        )
        payload = json.loads(json.dumps(dataclasses.asdict(result)))
        assert set(payload["sweeps"]) == set(VARIABLES)


class TestWorkingFiles:
    def test_successful_run_leaves_nothing_behind(self, samples, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        evaluate_along_axes(samples, VARIABLES, RESPONSE, points_per_variable=5)
        assert not list(tmp_path.glob("itis-sumo-*"))

    def test_workspace_keeps_the_evidence(self, samples, tmp_path):
        evaluate_along_axes(
            samples,
            VARIABLES,
            RESPONSE,
            points_per_variable=5,
            workspace=tmp_path,
        )
        produced = list(tmp_path.rglob("processed_samples.dat"))
        assert produced, "workspace should retain the training file"


class TestGrid:
    def test_grid_preserves_original_names_and_units(self, samples):
        result = evaluate_grid(
            samples,
            VARIABLES,
            RESPONSE,
            grid_variables=["width", "height"],
            points_per_variable=5,
        )
        assert result.response == RESPONSE
        assert result.grid_variables == ("width", "height")
        assert "stress" in result.data
        assert min(cast("list[float]", result.data["width"])) >= WIDTH_RANGE[0] - 0.01
        assert max(cast("list[float]", result.data["height"])) <= HEIGHT_RANGE[1] + 0.01
        assert len(result.data["stress"]) == 5
        assert len(cast("dict[str, list[list[float]]]", result.data)["stress"][0]) == 5


class TestSobol:
    def test_returns_seeded_indices_for_explicit_distributions(self, samples):
        distributions = {
            "width": DistributionSpec("uniform", minimum=1.0, maximum=5.0),
            "height": DistributionSpec("uniform", minimum=100.0, maximum=500.0),
        }
        result = evaluate_sobol(
            samples, VARIABLES, RESPONSE, distributions=distributions, seed=7
        )
        assert result.response == RESPONSE
        assert result.seed == 7
        assert set(result.indices) == set(VARIABLES)
        assert set(result.second_order) <= set(VARIABLES)

    def test_requires_a_distribution_for_each_variable(self, samples):
        with pytest.raises(SumoInputError, match="cover variables exactly"):
            evaluate_sobol(
                samples,
                VARIABLES,
                RESPONSE,
                distributions={"width": DistributionSpec("constant", value=2.0)},
            )


class TestDiagnostics:
    def test_correlations_use_original_column_names(self, samples):
        result = compute_correlations(samples, VARIABLES, RESPONSE)
        assert result.response == RESPONSE
        assert set(result.coefficients) == set(VARIABLES)
        assert result.coefficients["width"]["pearson"] > 0.9

    def test_correlations_reject_missing_columns(self, samples):
        with pytest.raises(SumoInputError, match="do not contain"):
            compute_correlations(samples, ["depth"], RESPONSE)

    def test_cv_metrics_compose_cross_validation(self, samples):
        result = evaluate_cv_metrics(samples, VARIABLES, RESPONSE, seed=7)
        assert result.response == RESPONSE
        assert result.seed == 7
        assert result.root_mean_squared >= 0.0
        assert result.mean_abs >= 0.0


class TestUncertainty:
    def test_propagates_uncertainty_in_original_units(self, samples):
        distributions = {
            "width": DistributionSpec("uniform", minimum=1.0, maximum=5.0),
            "height": DistributionSpec("uniform", minimum=100.0, maximum=500.0),
        }
        result = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=distributions,
            num_samples=50,
            n_histograms=10,
            seed=7,
        )
        assert result.response == RESPONSE
        assert result.seed == 7
        assert result.bins_start < result.bins_end
        assert result.q1 <= result.median <= result.q3
        assert result.mean > 0.0
        assert len(result.bin_means) == len(result.bin_stds)

    def test_requires_a_distribution_for_each_variable(self, samples):
        with pytest.raises(SumoInputError, match="cover variables exactly"):
            evaluate_uncertainty(
                samples,
                VARIABLES,
                RESPONSE,
                distributions={"width": DistributionSpec("constant", value=2.0)},
            )


class TestOptimize:
    def test_finds_a_pareto_front_over_the_domain(self, samples):
        domains = {
            "width": DomainSpec(minimum=WIDTH_RANGE[0], maximum=WIDTH_RANGE[1]),
            "height": DomainSpec(minimum=HEIGHT_RANGE[0], maximum=HEIGHT_RANGE[1]),
        }
        result = optimize(
            samples,
            VARIABLES,
            {RESPONSE: "minimize"},
            domains=domains,
            max_evaluations=200,
        )
        assert result.objectives == {RESPONSE: "minimize"}
        assert set(result.data) >= {RESPONSE, "width", "height"}
        assert min(result.data["width"]) >= WIDTH_RANGE[0] - 0.5
        assert max(result.data["width"]) <= WIDTH_RANGE[1] + 0.5

    def test_requires_a_domain_for_each_variable(self, samples):
        with pytest.raises(SumoInputError, match="cover variables exactly"):
            optimize(
                samples,
                VARIABLES,
                {RESPONSE: "minimize"},
                domains={"width": DomainSpec(minimum=1.0, maximum=5.0)},
                max_evaluations=200,
            )

    def test_log_objective_front_returns_original_units(self, samples):
        domains = {
            "width": DomainSpec(minimum=WIDTH_RANGE[0], maximum=WIDTH_RANGE[1]),
            "height": DomainSpec(minimum=HEIGHT_RANGE[0], maximum=HEIGHT_RANGE[1]),
        }
        result = optimize(
            samples,
            VARIABLES,
            {RESPONSE: "minimize"},
            domains=domains,
            max_evaluations=200,
            preprocessing=_LOG_SCALE,
        )
        front = result.data[RESPONSE]
        assert front
        # A log-trained objective must come back exp-restored to original stress
        # units (O(10)), not the O(1..3) ln values.
        assert min(front) > 0.0
        assert max(front) < 10 * samples[RESPONSE].max()

    def test_log_variable_domain_rejects_non_positive_bounds(self, samples):
        domains = {
            "width": DomainSpec(minimum=0.0, maximum=5.0),  # log => must be > 0
            "height": DomainSpec(minimum=HEIGHT_RANGE[0], maximum=HEIGHT_RANGE[1]),
        }
        log_width = PreprocessingSpec(overrides={"width": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="domains must be strictly positive"):
            optimize(
                samples,
                VARIABLES,
                {RESPONSE: "minimize"},
                domains=domains,
                max_evaluations=100,
                preprocessing=log_width,
            )


_LOG_SCALE = PreprocessingSpec(overrides={RESPONSE: VariableSpec(scale="log")})


class TestLogScale:
    """The domain-level ``scale`` flag must reach the surrogate and come back
    in the caller's own units -- SPEC V21pf (scale in, no transform in the
    signature) and the port of the mmux_vite log-scale backend (T27fr)."""

    def test_log_scale_response_comes_back_in_original_units(self, samples):
        result = cross_validate(samples, VARIABLES, RESPONSE, preprocessing=_LOG_SCALE)

        assert result.effective_config[RESPONSE].scale == "log"
        predicted = [v for v in result.predicted if not math.isnan(v)]
        assert predicted, "no fold produced a prediction"
        # Log-space predictions must be exp-restored: same order of magnitude as
        # the raw stress (which is O(10)), not the O(1..3) log values.
        assert min(predicted) > 0.0
        assert max(predicted) < 10 * samples[RESPONSE].max()
        observed_mean = float(np.mean(samples[RESPONSE]))
        assert min(predicted) < 3 * observed_mean < 10 * max(predicted)

    def test_log_scale_survives_from_request_to_engine(self, samples):
        linear = cross_validate(samples, VARIABLES, RESPONSE)
        log = cross_validate(samples, VARIABLES, RESPONSE, preprocessing=_LOG_SCALE)
        # Both agree roughly with the observed stress -- the transform is internal.
        assert np.mean(log.predicted) == pytest.approx(
            np.mean(linear.predicted), rel=0.5, nan_ok=True
        )

    def test_along_axes_log_response_returns_original_units(self, samples):
        result = evaluate_along_axes(
            samples,
            VARIABLES,
            RESPONSE,
            preprocessing=_LOG_SCALE,
            points_per_variable=5,
        )
        assert result.effective_config[RESPONSE].scale == "log"
        for sweep in result.sweeps.values():
            assert len(sweep.x) == len(sweep.predicted)
            assert min(sweep.predicted) > 0.0
            assert max(sweep.predicted) < 10 * samples[RESPONSE].max()

    def test_grid_log_response_returns_original_units(self, samples):
        result = evaluate_grid(
            samples,
            VARIABLES,
            RESPONSE,
            grid_variables=["width", "height"],
            preprocessing=_LOG_SCALE,
            points_per_variable=5,
        )
        flat = [
            v
            for row in cast("dict[str, list[list[float]]]", result.data)["stress"]
            for v in row
        ]
        assert min(flat) > 0.0
        assert max(flat) < 10 * samples[RESPONSE].max()

    def test_non_positive_log_response_is_rejected_before_dakota(self):
        bad = pd.DataFrame(
            {
                "width": [1.0, 2.0, 3.0, 4.0, 5.0],
                "height": [100.0, 200.0, 300.0, 400.0, 500.0],
                "stress": [5.0, 8.0, 0.0, 12.0, 15.0],  # <= 0 under log
            }
        )
        with pytest.raises(SumoInputError, match="log-scale but hold values"):
            cross_validate(bad, VARIABLES, RESPONSE, preprocessing=_LOG_SCALE)

    def test_holding_a_log_variable_at_zero_is_rejected(self, samples):
        log_input = PreprocessingSpec(overrides={"height": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="log-scale .* fixed at a value"):
            evaluate_along_axes(
                samples,
                VARIABLES,
                RESPONSE,
                at={"height": 0.0},
                preprocessing=log_input,
                points_per_variable=5,
            )


_LOG_WIDTH = PreprocessingSpec(overrides={"width": VariableSpec(scale="log")})
_UQ_DISTS = {
    "width": DistributionSpec("uniform", minimum=1.0, maximum=5.0),
    "height": DistributionSpec("uniform", minimum=100.0, maximum=500.0),
}
# For a log-scale normal the μ/σ parameterize ln space (V46rn), so e^μ puts the
# raw-space centre where the uniform above has it -- scale flips stay near the
# training range instead of extrapolating.
_NORMAL_WIDTH = {
    "width": DistributionSpec("normal", mean=math.log(3.0), std=0.5),
    "height": _UQ_DISTS["height"],
}


class TestLogScaleUncertaintyAndMetrics:
    """Log must reach the UQ sampler and the CV metrics, not just the surrogate
    (SPEC T27fr -- 'log applied everywhere')."""

    def test_log_input_samples_log_uniform_skewing_response_low(self, samples):
        linear = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=400,
            n_histograms=5,
            seed=7,
        )
        logw = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            preprocessing=_LOG_WIDTH,
            num_samples=400,
            n_histograms=5,
            seed=7,
        )
        # width drawn log-uniform is skewed toward the low end, so the stress it
        # drives must sit meaningfully below the linear-uniform case.
        assert logw.mean > 0.0
        assert logw.mean < linear.mean - 0.5

    def test_log_input_with_a_normal_distribution_draws_lognormal(self, samples):
        # V46rn: log composes with normal -- μ/σ stay in ln space, raw draws are
        # lognormal (positive by construction), and the propagated statistics
        # move relative to the linear draw.
        linear = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_NORMAL_WIDTH,
            num_samples=400,
            n_histograms=5,
            seed=7,
        )
        logw = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_NORMAL_WIDTH,
            preprocessing=_LOG_WIDTH,
            num_samples=400,
            n_histograms=5,
            seed=7,
        )
        # E[lognormal] = exp(mu + sigma^2/2) > exp(mu) = E[normal] (Jensen), so
        # the lognormal drive sits above the linear one -- the flag reached the
        # sampler, it was not quietly treated as linear.
        assert logw.mean > 0.0
        assert logw.mean > linear.mean + 0.05

    def test_log_input_rejects_a_non_positive_lower_bound(self, samples):
        zero_min = {
            "width": DistributionSpec("uniform", minimum=0.0, maximum=5.0),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="not strictly positive"):
            evaluate_uncertainty(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=zero_min,
                preprocessing=_LOG_WIDTH,
                num_samples=50,
                n_histograms=3,
                seed=7,
            )

    def test_cv_metrics_honour_log_scale(self, samples):
        linear = evaluate_cv_metrics(samples, VARIABLES, RESPONSE, seed=7)
        log_response = evaluate_cv_metrics(
            samples, VARIABLES, RESPONSE, preprocessing=_LOG_SCALE, seed=7
        )
        assert linear.root_mean_squared >= 0.0
        assert log_response.root_mean_squared >= 0.0
        # A log-trained surrogate is a different fit, so the metrics must not be
        # byte-identical to the linear run (proving the flag reached the path).
        assert log_response.root_mean_squared != pytest.approx(
            linear.root_mean_squared, rel=1e-9
        )

    def test_cv_metrics_reject_non_positive_log_response(self, samples):
        bad = samples.copy()
        bad.loc[bad.index[0], RESPONSE] = -1.0
        with pytest.raises(SumoInputError, match="log-scale but hold values"):
            evaluate_cv_metrics(bad, VARIABLES, RESPONSE, preprocessing=_LOG_SCALE)


class TestLogScaleSobol:
    """Log must reach the Sobol sampler too (SPEC T27fr -- 'log everywhere'): a
    log-scale variable is drawn log-uniform, so the variance decomposition is
    taken over that distribution rather than a linear one."""

    def test_log_input_changes_the_variance_decomposition(self, samples):
        linear = evaluate_sobol(
            samples, VARIABLES, RESPONSE, distributions=_UQ_DISTS, seed=7
        )
        logw = evaluate_sobol(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            preprocessing=_LOG_WIDTH,
            seed=7,
        )
        assert set(logw.indices) == set(VARIABLES)
        # width drawn log-uniform is compressed toward its low end, so it explains
        # slightly less response variance than a linear width -- and height, taking
        # the residual share, rises. The direction proves the flag reached the
        # sampler, not just the surrogate fit.
        assert logw.indices["width"]["total"] < linear.indices["width"]["total"]
        assert logw.indices["height"]["total"] > linear.indices["height"]["total"]

    def test_log_input_with_a_normal_distribution_shifts_the_decomposition(
        self, samples
    ):
        # V46rn: a log-scale normal widens the dominant variable's raw-space
        # spread (lognormal tail), so its share of the response variance rises
        # and the residual variable's falls -- the decomposition is taken over
        # what the model sees, not a silently linear draw.
        linear = evaluate_sobol(
            samples, VARIABLES, RESPONSE, distributions=_NORMAL_WIDTH, seed=7
        )
        logw = evaluate_sobol(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_NORMAL_WIDTH,
            preprocessing=_LOG_WIDTH,
            seed=7,
        )
        assert set(logw.indices) == set(VARIABLES)
        assert logw.indices["width"]["total"] > linear.indices["width"]["total"]
        assert logw.indices["height"]["total"] < linear.indices["height"]["total"]

    def test_log_input_rejects_a_non_positive_lower_bound(self, samples):
        zero_min = {
            "width": DistributionSpec("uniform", minimum=0.0, maximum=5.0),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="not strictly positive"):
            evaluate_sobol(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=zero_min,
                preprocessing=_LOG_WIDTH,
                seed=7,
            )


_MOGA_DOMAINS = {
    "width": DomainSpec(minimum=WIDTH_RANGE[0], maximum=WIDTH_RANGE[1]),
    "height": DomainSpec(minimum=HEIGHT_RANGE[0], maximum=HEIGHT_RANGE[1]),
}
_SAMPLER_DOMAINS = {
    "width": DomainSpec(minimum=1.0, maximum=100.0),
    "height": DomainSpec(minimum=10.0, maximum=20.0),
}


class TestScaleGapCoverage:
    """The edges the first log-scale pass left untested (T46ls)."""

    def test_moga_log_objective_maximize_front_in_original_units(self, samples):
        result = optimize(
            samples,
            VARIABLES,
            {RESPONSE: "maximize"},
            domains=_MOGA_DOMAINS,
            max_evaluations=200,
            preprocessing=_LOG_SCALE,
        )
        front = result.data[RESPONSE]
        assert front
        # Maximizing a log objective inverts as -(ln y) -> ln y -> exp: the front
        # must come back as HIGH stresses in original units. A broken sign/log
        # order would return ln-space values (O(3)) or a minimized front.
        assert min(front) > 0.0
        assert max(front) < 10 * samples[RESPONSE].max()
        assert max(front) > float(samples[RESPONSE].mean())

    def test_log_variable_sweep_returns_geometric_original_units(self, samples):
        result = evaluate_along_axes(
            samples,
            VARIABLES,
            RESPONSE,
            preprocessing=_LOG_WIDTH,
            points_per_variable=7,
        )
        xs = np.asarray(result.sweeps["width"].x, dtype=float)
        assert xs.min() > 0.5
        assert xs.max() <= samples["width"].max() * 1.01
        # Swept in log space, restored to original units => ln-equispaced.
        diffs = np.diff(np.log(xs))
        assert np.allclose(diffs, diffs.mean(), rtol=1e-6)

    def test_uncertainty_log_response_spread_is_multiplicative(self, samples):
        result = evaluate_uncertainty(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            preprocessing=_LOG_SCALE,
            num_samples=300,
            n_histograms=5,
            seed=7,
        )
        assert result.q1 > 0.0
        assert result.q1 <= result.median <= result.q3
        # Noise is injected in log space then exp-restored -> bounded relative
        # (multiplicative) spread, not an additive tail around the mean.
        assert 0.5 < result.q1 / result.median
        assert result.q3 / result.median < 2.0

    def test_sobol_mixed_log_and_constant(self, samples):
        dists = {
            "width": DistributionSpec("uniform", minimum=1.0, maximum=5.0),
            "height": DistributionSpec("constant", value=300.0),
        }
        result = evaluate_sobol(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=dists,
            preprocessing=_LOG_WIDTH,
            seed=7,
        )
        assert set(result.indices) == set(VARIABLES)
        assert result.indices["width"]["total"] > 0.8
        assert result.indices["height"]["total"] < 0.3


class TestScaleAwareSamplers:
    """Standalone samplers and correlations honour scale like every other
    value producer (V45ls; defaults to linear, so old calls are unchanged)."""

    def test_lhs_honours_log_scale(self):
        linear = generate_lhs_samples(_SAMPLER_DOMAINS, 500, seed=11)
        logw = generate_lhs_samples(
            _SAMPLER_DOMAINS, 500, preprocessing=_LOG_WIDTH, seed=11
        )
        widths = np.asarray(logw["width"], dtype=float)
        assert widths.min() >= 1.0 and widths.max() <= 100.0
        # log-uniform in [1,100]: geometric mean at the log-midpoint.
        assert float(np.exp(np.log(widths).mean())) == pytest.approx(10.0, rel=0.15)
        assert float(widths.mean()) < 0.5 * float(np.mean(linear["width"]))

    def test_lhs_linear_default_unchanged(self):
        first = generate_lhs_samples(_SAMPLER_DOMAINS, 200, seed=5)
        second = generate_lhs_samples(_SAMPLER_DOMAINS, 200, seed=5)
        pd.testing.assert_frame_equal(first, second)
        widths = np.asarray(first["width"], dtype=float)
        assert widths.min() >= 1.0 and widths.max() <= 100.0

    def test_lhs_rejects_non_positive_log_domain(self):
        domains = {"width": DomainSpec(minimum=0.0, maximum=100.0)}
        log_width = PreprocessingSpec(overrides={"width": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="strictly positive"):
            generate_lhs_samples(domains, 50, preprocessing=log_width, seed=1)

    def test_grid_log_axis_is_geometric(self):
        from scipy.stats import loguniform

        domains = {"width": DomainSpec(minimum=1.0, maximum=100.0)}
        linear = generate_grid_samples(domains, {"width": 5})
        logw = generate_grid_samples(domains, {"width": 5}, preprocessing=_LOG_WIDTH)
        assert np.allclose(
            np.sort(np.unique(logw["width"])),
            loguniform(a=1.0, b=100.0).ppf(np.linspace(0.0, 1.0, 5)),
            rtol=1e-6,
        )
        assert np.allclose(
            np.sort(np.unique(linear["width"])),
            np.linspace(1.0, 100.0, 5),
            rtol=1e-6,
        )

    def test_grid_rejects_non_positive_log_domain(self):
        domains = {"width": DomainSpec(minimum=0.0, maximum=100.0)}
        log_width = PreprocessingSpec(overrides={"width": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="strictly positive"):
            generate_grid_samples(domains, {"width": 4}, preprocessing=log_width)

    def test_correlations_honour_log_scale(self, samples):
        linear = compute_correlations(samples, VARIABLES, RESPONSE)
        logw = compute_correlations(
            samples, VARIABLES, RESPONSE, preprocessing=_LOG_WIDTH
        )
        # Pearson on the log-scale column must move; the rank statistic is
        # monotone-invariant; the untouched variable is bit-identical.
        assert logw.coefficients["width"]["pearson"] != pytest.approx(
            linear.coefficients["width"]["pearson"], rel=1e-3
        )
        assert logw.coefficients["width"]["spearman"] == pytest.approx(
            linear.coefficients["width"]["spearman"], abs=1e-12
        )
        assert logw.coefficients["height"] == pytest.approx(
            linear.coefficients["height"]
        )


class TestEvaluateCorrelations:
    """`evaluate_correlations` owns the MC-through-surrogate correlation
    workflow (#470) so no consumer re-implements the sampling chain (V45ls)."""

    def test_recovers_dominant_variable_over_shared_sample_set(self, samples):
        result = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=300,
            seed=7,
        )
        assert result.response == RESPONSE
        assert result.seed == 7
        assert set(result.coefficients) == set(VARIABLES)
        # stress = 3*width + 0.01*height: width dominates on the shared set.
        assert abs(result.coefficients["width"]["pearson"]) > 0.9
        assert abs(result.coefficients["width"]["pearson"]) > abs(
            result.coefficients["height"]["pearson"]
        )

    def test_is_seed_reproducible(self, samples):
        first = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=150,
            seed=11,
        )
        second = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=150,
            seed=11,
        )
        assert first.coefficients == second.coefficients

    def test_log_scale_moves_the_coefficients(self, samples):
        # NOTE: unlike table-mode correlation, log mode ALSO changes the draw
        # (log-uniform vs uniform), so only "the numbers move" is guaranteed --
        # no rank-invariance claim here.
        linear = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=300,
            seed=7,
        )
        logw = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_UQ_DISTS,
            num_samples=300,
            preprocessing=_LOG_WIDTH,
            seed=7,
        )
        assert logw.coefficients["width"]["pearson"] != pytest.approx(
            linear.coefficients["width"]["pearson"], rel=1e-3
        )

    def test_log_normal_variable_moves_the_coefficients(self, samples):
        # V46rn: log composes with normal -- the shared MC set is drawn
        # lognormal, so Pearson over the surrogate predictions moves.
        linear = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_NORMAL_WIDTH,
            num_samples=300,
            seed=7,
        )
        logw = evaluate_correlations(
            samples,
            VARIABLES,
            RESPONSE,
            distributions=_NORMAL_WIDTH,
            num_samples=300,
            preprocessing=_LOG_WIDTH,
            seed=7,
        )
        assert logw.coefficients["width"]["pearson"] != pytest.approx(
            linear.coefficients["width"]["pearson"], rel=1e-3
        )

    def test_log_variable_rejects_constant_and_non_positive_uniform(self, samples):
        constant = {
            "width": DistributionSpec("constant", value=3.0),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="only a uniform or normal"):
            evaluate_correlations(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=constant,
                num_samples=50,
                preprocessing=_LOG_WIDTH,
                seed=7,
            )
        non_positive = {
            "width": DistributionSpec("uniform", minimum=0.0, maximum=5.0),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="strictly positive"):
            evaluate_correlations(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=non_positive,
                num_samples=50,
                preprocessing=_LOG_WIDTH,
                seed=7,
            )

    def test_distributions_must_cover_variables_exactly(self, samples):
        with pytest.raises(SumoInputError, match="cover variables exactly"):
            evaluate_correlations(
                samples,
                VARIABLES,
                RESPONSE,
                distributions={"width": _UQ_DISTS["width"]},
                num_samples=50,
                seed=7,
            )

    def test_producer_requires_scales(self):
        """V45ls structural tripwire on the new value-producing entry point."""
        from itis_sumo.evaluate.funs_evaluate import correlate_manual_uq_samples

        with pytest.raises(TypeError):
            correlate_manual_uq_samples(  # ty: ignore[missing-argument]
                Path("."), Path("."), ["x"], "y", {}, None, 10, seed=1
            )

    def test_table_entry_points_reject_unused_overrides(self, samples):
        """V47st: a misspelled log-scale column fails loud everywhere, not just
        in the session-backed workflows -- a silently ignored override would
        return plausible linear results."""
        misspelled = PreprocessingSpec(overrides={"wdith": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="not in play"):
            compute_correlations(samples, VARIABLES, RESPONSE, preprocessing=misspelled)
        with pytest.raises(SumoInputError, match="not in play"):
            generate_lhs_samples(_SAMPLER_DOMAINS, 20, preprocessing=misspelled, seed=7)
        with pytest.raises(SumoInputError, match="not in play"):
            generate_grid_samples(
                _SAMPLER_DOMAINS,
                {"width": 3, "height": 3},
                preprocessing=misspelled,
            )

    def test_log_uniform_requires_a_usable_upper_bound(self, samples):
        """V47st: the boundary checks both bounds -- a missing maximum must not
        reach the engine and resurface there as a SumoEngineError."""
        missing_max = {
            "width": DistributionSpec("uniform", minimum=1.0),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="upper bound"):
            evaluate_uncertainty(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=missing_max,
                preprocessing=_LOG_WIDTH,
                num_samples=50,
                n_histograms=3,
                seed=7,
            )


class TestScaleFlipMatrix:
    """V45ls behavioural enforcement in one place: EVERY public value-producing
    entry point's output must move when a column turns log. A silent scale-ignore
    in any path -- shipped or future -- fails here."""

    @staticmethod
    def _fp(values: object) -> list[float]:
        return [float(v) for v in np.asarray(values, dtype=float).ravel()]

    def test_every_entry_point_moves_under_log(self, samples):
        cases: list[tuple[str, Callable[[PreprocessingSpec | None], list[float]]]] = [
            (
                "cross_validate",
                lambda p: self._fp(
                    cross_validate(
                        samples, VARIABLES, RESPONSE, preprocessing=p, seed=7
                    ).predicted
                ),
            ),
            (
                "evaluate_cv_metrics",
                lambda p: self._fp(
                    evaluate_cv_metrics(
                        samples, VARIABLES, RESPONSE, preprocessing=p, seed=7
                    ).root_mean_squared
                ),
            ),
            (
                "evaluate_along_axes",
                lambda p: self._fp(
                    [
                        v
                        for sweep in evaluate_along_axes(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            preprocessing=p,
                            points_per_variable=5,
                        ).sweeps.values()
                        for v in sweep.predicted
                    ]
                ),
            ),
            (
                "evaluate_grid",
                lambda p: self._fp(
                    cast(
                        "dict[str, list[list[float]]]",
                        evaluate_grid(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            grid_variables=VARIABLES,
                            preprocessing=p,
                            points_per_variable=5,
                        ).data,
                    )[RESPONSE]
                ),
            ),
            (
                "evaluate_uncertainty",
                lambda p: self._fp(
                    evaluate_uncertainty(
                        samples,
                        VARIABLES,
                        RESPONSE,
                        distributions=_UQ_DISTS,
                        preprocessing=p,
                        num_samples=150,
                        n_histograms=4,
                        seed=7,
                    ).mean
                ),
            ),
            (
                "evaluate_sobol",
                lambda p: self._fp(
                    [
                        entry["total"]
                        for entry in evaluate_sobol(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            distributions=_UQ_DISTS,
                            preprocessing=p,
                            seed=7,
                        ).indices.values()
                    ]
                ),
            ),
            (
                "optimize",
                lambda p: sorted(
                    self._fp(
                        optimize(
                            samples,
                            VARIABLES,
                            {RESPONSE: "minimize"},
                            domains=_MOGA_DOMAINS,
                            max_evaluations=150,
                            preprocessing=p,
                        ).data[RESPONSE]
                    )
                ),
            ),
            (
                "evaluate_correlations",
                lambda p: self._fp(
                    [
                        entry["pearson"]
                        for entry in evaluate_correlations(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            distributions=_UQ_DISTS,
                            num_samples=150,
                            preprocessing=p,
                            seed=7,
                        ).coefficients.values()
                    ]
                ),
            ),
            (
                "compute_correlations",
                lambda p: self._fp(
                    [
                        entry["pearson"]
                        for entry in compute_correlations(
                            samples, VARIABLES, RESPONSE, preprocessing=p
                        ).coefficients.values()
                    ]
                ),
            ),
            (
                "generate_lhs_samples",
                lambda p: self._fp(
                    generate_lhs_samples(
                        _SAMPLER_DOMAINS, 200, preprocessing=p, seed=7
                    )["width"]
                ),
            ),
            (
                "generate_grid_samples",
                lambda p: self._fp(
                    generate_grid_samples(
                        _SAMPLER_DOMAINS,
                        {"width": 5, "height": 4},
                        preprocessing=p,
                    )["width"]
                ),
            ),
        ]
        assert len(cases) == 11
        for name, fingerprint in cases:
            linear = fingerprint(None)
            log = fingerprint(_LOG_WIDTH)
            assert log != pytest.approx(linear, rel=1e-9, nan_ok=True), (
                f"{name}: output is identical under log scale -- the scale "
                "override was silently ignored (V45ls)"
            )

    def test_every_distribution_entry_point_moves_for_a_log_normal(self, samples):
        """V46rn: the same flip guarantee for a NORMAL column turned log -- the
        entry points that take `distributions` must draw lognormal, not reject
        (the pre-B19ps behaviour) and not silently stay linear."""
        cases: list[tuple[str, Callable[[PreprocessingSpec | None], list[float]]]] = [
            (
                "evaluate_uncertainty",
                lambda p: self._fp(
                    evaluate_uncertainty(
                        samples,
                        VARIABLES,
                        RESPONSE,
                        distributions=_NORMAL_WIDTH,
                        preprocessing=p,
                        num_samples=150,
                        n_histograms=4,
                        seed=7,
                    ).mean
                ),
            ),
            (
                "evaluate_sobol",
                lambda p: self._fp(
                    [
                        entry["total"]
                        for entry in evaluate_sobol(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            distributions=_NORMAL_WIDTH,
                            preprocessing=p,
                            seed=7,
                        ).indices.values()
                    ]
                ),
            ),
            (
                "evaluate_correlations",
                lambda p: self._fp(
                    [
                        entry["pearson"]
                        for entry in evaluate_correlations(
                            samples,
                            VARIABLES,
                            RESPONSE,
                            distributions=_NORMAL_WIDTH,
                            num_samples=150,
                            preprocessing=p,
                            seed=7,
                        ).coefficients.values()
                    ]
                ),
            ),
        ]
        assert len(cases) == 3
        for name, fingerprint in cases:
            linear = fingerprint(None)
            log = fingerprint(_LOG_WIDTH)
            assert log != pytest.approx(linear, rel=1e-9, nan_ok=True), (
                f"{name}: output is identical under a log-scale normal -- the "
                "lognormal draw was bypassed (V46rn)"
            )
