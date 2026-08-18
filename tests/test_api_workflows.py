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

import numpy as np
import pandas as pd
import pytest

from itis_sumo.api import (
    DistributionSpec,
    DomainSpec,
    SumoInputError,
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
        assert min(result.data["width"]) >= WIDTH_RANGE[0] - 0.01
        assert max(result.data["height"]) <= HEIGHT_RANGE[1] + 0.01
        assert len(result.data["stress"]) == 5
        assert len(result.data["stress"][0]) == 5


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
