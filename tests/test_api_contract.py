"""Contract tests for the itis-sumo consumer API.

These assert the promises SPEC makes to whoever embeds itis-sumo, independently
of what Dakota happens to compute: the vocabulary of the public surface, what a
caller is allowed to pass, what comes back, which errors escape, and what happens
to working files. See SPEC V19cn, V20dm, V21pf, V22rs, V23er, V24af, V25sd.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

from itis_sumo import api
from itis_sumo.api import (
    AlongAxesResult,
    AxisSweep,
    CrossValidationResult,
    GridResult,
    PreprocessingSpec,
    SumoEngineError,
    SumoError,
    SumoInputError,
    VariableSpec,
    cross_validate,
    evaluate_along_axes,
    evaluate_grid,
)
from itis_sumo.api._session import SumoSession

pytestmark = pytest.mark.unit

VARIABLES = ["width", "height"]
RESPONSE = "stress"


def make_samples(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    width = rng.uniform(1.0, 5.0, n)
    height = rng.uniform(10.0, 50.0, n)
    return pd.DataFrame(
        {"width": width, "height": height, "stress": 3.0 * width + 0.1 * height}
    )


class TestPublicSurface:
    def test_exports_only_the_documented_names(self):
        assert set(api.__all__) == {
            "DEFAULT_SEED",
            "AlongAxesResult",
            "AxisSweep",
            "CrossValidationResult",
            "GridResult",
            "PreprocessingSpec",
            "Scale",
            "SumoEngineError",
            "SumoError",
            "SumoInputError",
            "SumoResultError",
            "VariableSpec",
            "cross_validate",
            "evaluate_along_axes",
            "evaluate_grid",
        }

    @pytest.mark.parametrize("workflow", [cross_validate, evaluate_along_axes])
    def test_speaks_the_spec_vocabulary(self, workflow):
        """SPEC V19cn: samples, variables, responses -- never jobs."""
        parameters = set(inspect.signature(workflow).parameters)
        assert {"samples", "variables", "response"} <= parameters
        assert not {"jobs", "function_jobs", "points", "records"} & parameters

    @pytest.mark.parametrize("workflow", [cross_validate, evaluate_along_axes])
    def test_hides_transform_vocabulary(self, workflow):
        """SPEC V21pf: overrides are expressed in domain terms, not transforms."""
        parameters = set(inspect.signature(workflow).parameters)
        assert (
            not {
                "normalization",
                "input_normalizations",
                "output_normalizations",
                "sign_switch",
                "input_sign_switches",
                "output_sign_switches",
                "mapped_name",
            }
            & parameters
        )

    @pytest.mark.parametrize("workflow", [cross_validate, evaluate_along_axes])
    def test_hides_dakota_file_plumbing(self, workflow):
        """SPEC §G: callers pass data and configuration, not file paths."""
        parameters = set(inspect.signature(workflow).parameters)
        assert (
            not {
                "run_dir",
                "training_file",
                "PROCESSED_TRAINING_FILE",
                "samples_file",
                "has_eval_id_column",
            }
            & parameters
        )

    def test_stochastic_workflow_seeds_itself(self):
        """SPEC V25sd: a fixed default, not a required argument."""
        seed = inspect.signature(cross_validate).parameters["seed"]
        assert seed.default == api.DEFAULT_SEED == 42

    def test_deterministic_workflow_takes_no_seed(self):
        assert "seed" not in inspect.signature(evaluate_along_axes).parameters

    def test_api_layer_is_free_of_web_and_platform_imports(self):
        """Sibling of SPEC V4ty, for the consumer-facing layer."""
        for module in list(sys.modules):
            if module.startswith("itis_sumo.api"):
                source = inspect.getsource(sys.modules[module])
                assert "import flask" not in source
                assert "osparc" not in source


class TestErrorTaxonomy:
    def test_every_public_error_descends_from_sumo_error(self):
        """SPEC V23er."""
        for name in api.__all__:
            member = getattr(api, name)
            if inspect.isclass(member) and issubclass(member, Exception):
                assert issubclass(member, SumoError)

    def test_engine_error_carries_its_evidence(self):
        error = SumoEngineError("boom", run_dir=None, stderr_tail="segfault")
        assert "segfault" in str(error)


class TestSampleValidation:
    """Judging whether a table can support a surrogate is itis-sumo's job."""

    def test_rejects_non_tabular_input(self):
        with pytest.raises(SumoInputError, match="DataFrame"):
            cross_validate([{"width": 1.0}], VARIABLES, RESPONSE)

    def test_rejects_unknown_columns(self):
        with pytest.raises(SumoInputError, match="not present"):
            cross_validate(make_samples(), ["width", "depth"], RESPONSE)

    def test_rejects_a_column_used_as_both_variable_and_response(self):
        with pytest.raises(SumoInputError, match="both"):
            cross_validate(make_samples(), VARIABLES, "width")

    def test_rejects_repeated_variables(self):
        with pytest.raises(SumoInputError, match="more than once"):
            cross_validate(make_samples(), ["width", "width"], RESPONSE)

    def test_rejects_non_numeric_columns(self):
        samples = make_samples()
        samples["width"] = "wide"
        with pytest.raises(SumoInputError, match="numeric"):
            cross_validate(samples, VARIABLES, RESPONSE)

    def test_rejects_missing_values(self):
        samples = make_samples()
        samples.loc[0, "stress"] = np.nan
        with pytest.raises(SumoInputError, match="missing or infinite"):
            cross_validate(samples, VARIABLES, RESPONSE)

    def test_rejects_too_few_samples(self):
        """Dakota aborts opaquely below max(5, n_variables + 1); we do not."""
        with pytest.raises(SumoInputError, match="samples are required"):
            cross_validate(make_samples(2), VARIABLES, RESPONSE)

    def test_rejects_overrides_for_columns_not_in_play(self):
        spec = PreprocessingSpec(overrides={"depth": VariableSpec()})
        with pytest.raises(SumoInputError, match="not in play"):
            cross_validate(make_samples(), VARIABLES, RESPONSE, preprocessing=spec)

    def test_reports_unsupported_scale_instead_of_ignoring_it(self):
        """An override that cannot be honoured must fail loudly (SPEC V21pf)."""
        spec = PreprocessingSpec(overrides={"width": VariableSpec(scale="log")})
        with pytest.raises(SumoInputError, match="Logarithmic"):
            cross_validate(make_samples(), VARIABLES, RESPONSE, preprocessing=spec)

    def test_rejects_a_single_fold(self):
        with pytest.raises(SumoInputError, match="at least 2 folds"):
            cross_validate(make_samples(), VARIABLES, RESPONSE, folds=1)

    def test_rejects_holding_a_variable_that_does_not_exist(self):
        with pytest.raises(SumoInputError, match="not variables"):
            evaluate_along_axes(make_samples(), VARIABLES, RESPONSE, at={"depth": 1.0})


class TestResultShape:
    """SPEC V22rs: original units, original names, JSON-serializable."""

    def test_cross_validation_result_survives_json(self):
        result = CrossValidationResult(
            response="stress",
            observed=[1.0],
            predicted=[1.1],
            predicted_std=[0.1],
            warnings=[],
            seed=42,
            effective_config={"width": VariableSpec()},
        )
        payload = json.loads(json.dumps(dataclasses.asdict(result)))
        assert payload["response"] == "stress"
        assert payload["effective_config"]["width"] == {"scale": "linear"}

    def test_along_axes_result_survives_json(self):
        result = AlongAxesResult(
            response="stress",
            sweeps={"width": AxisSweep(variable="width", x=[1.0], predicted=[2.0])},
            effective_config={"width": VariableSpec()},
        )
        payload = json.loads(json.dumps(dataclasses.asdict(result)))
        assert payload["sweeps"]["width"]["variable"] == "width"

    @pytest.mark.parametrize("result_type", [CrossValidationResult, AlongAxesResult])
    def test_results_carry_no_transform_suffixes(self, result_type):
        fields = {field.name for field in dataclasses.fields(result_type)}
        assert not any(name.endswith(("_hat", "_std_hat")) for name in fields)

    def test_results_are_immutable(self):
        sweep = AxisSweep(variable="width", x=[1.0], predicted=[2.0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            sweep.variable = "height"


class TestWorkingFileLifetime:
    """SPEC V24af: discarded on success, preserved on failure."""

    def test_run_directory_is_discarded_when_nothing_goes_wrong(self):
        with SumoSession(make_samples(), VARIABLES, RESPONSE) as session:
            run_dir = session._run_dir
            assert run_dir.is_dir()
        assert not run_dir.exists()

    def test_run_directory_survives_a_failure(self):
        try:
            with SumoSession(make_samples(), VARIABLES, RESPONSE) as session:
                run_dir = session._run_dir
                raise RuntimeError("engine exploded")
        except RuntimeError:
            pass
        assert run_dir.is_dir()

    def test_workspace_keeps_its_files(self, tmp_path):
        with SumoSession(
            make_samples(), VARIABLES, RESPONSE, workspace=tmp_path
        ) as session:
            run_dir = session._run_dir
        assert run_dir.is_dir()
        assert tmp_path in run_dir.parents

    def test_engine_failures_are_translated_and_carry_the_run_directory(
        self, monkeypatch
    ):
        import itis_sumo.api._session as session_module

        def explode(*args, **kwargs):
            raise IndexError("map::at")

        monkeypatch.setattr(session_module, "evaluate_sumo_along_axes", explode)

        with pytest.raises(SumoEngineError) as caught:
            evaluate_along_axes(make_samples(), VARIABLES, RESPONSE)

        assert caught.value.run_dir is not None
        assert caught.value.run_dir.is_dir()
        assert "map::at" in str(caught.value)


class TestEffectiveConfiguration:
    """SPEC V21pf: readable, but not settable in transform terms."""

    def test_reports_a_specification_for_every_column_in_play(self):
        session = SumoSession(make_samples(), VARIABLES, RESPONSE)
        assert set(session.effective_config) == {"width", "height", "stress"}
        assert all(
            isinstance(spec, VariableSpec) for spec in session.effective_config.values()
        )

    def test_defaults_are_reported_even_when_nothing_was_supplied(self):
        session = SumoSession(make_samples(), VARIABLES, RESPONSE)
        assert session.effective_config["width"].scale == "linear"


class TestGridContract:
    def test_grid_rejects_unknown_grid_variables(self):
        with pytest.raises(SumoInputError, match="not variables"):
            evaluate_grid(make_samples(), VARIABLES, RESPONSE, grid_variables=["depth"])

    def test_grid_rejects_too_few_points(self):
        with pytest.raises(SumoInputError, match="at least 2"):
            evaluate_grid(
                make_samples(),
                VARIABLES,
                RESPONSE,
                grid_variables=["width"],
                points_per_variable=1,
            )

    def test_grid_result_is_typed_and_serializable(self):
        result = GridResult(
            response=RESPONSE,
            grid_variables=("width", "height"),
            data={"width": [1.0], "stress": [[2.0]]},
            effective_config={"width": VariableSpec()},
        )
        payload = json.loads(json.dumps(dataclasses.asdict(result)))
        assert payload["grid_variables"] == ["width", "height"]
