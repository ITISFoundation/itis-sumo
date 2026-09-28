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
    evaluate_cv_metrics,
    evaluate_grid,
    evaluate_sobol,
    evaluate_uncertainty,
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
        flat = [v for row in result.data["stress"] for v in row]
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

    def test_log_input_rejects_a_normal_uncertainty_distribution(self, samples):
        normal_width = {
            "width": DistributionSpec("normal", mean=3.0, std=0.5),
            "height": _UQ_DISTS["height"],
        }
        with pytest.raises(SumoInputError, match="only a uniform supports log"):
            evaluate_uncertainty(
                samples,
                VARIABLES,
                RESPONSE,
                distributions=normal_width,
                preprocessing=_LOG_WIDTH,
                num_samples=50,
                n_histograms=3,
                seed=7,
            )

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
