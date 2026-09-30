"""Internal fit-then-query engine behind the public one-shot workflows.

The public functions in :mod:`itis_sumo.api.workflows` are deliberately thin
wrappers over a session that is fitted once and then queried many times. Keeping
that split in place from the outset is what makes the eventual public
fitted-model handle a re-export rather than a second port (SPEC V27fq).

Nothing in this module is public API.
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
import pandas as pd

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from itis_sumo.api.errors import (
    SumoEngineError,
    SumoError,
    SumoInputError,
    SumoResultError,
)
from itis_sumo.api.types import (
    AlongAxesResult,
    AxisSweep,
    CorrelationResult,
    CrossValidationResult,
    Direction,
    DistributionSpec,
    DomainSpec,
    GridResult,
    OrderMasses,
    ParetoFrontResult,
    PreprocessingSpec,
    SobolResult,
    UncertaintyResult,
    VariableSpec,
)
from itis_sumo.evaluate.funs_evaluate import (
    correlate_manual_uq_samples,
    evaluate_sobol_indices,
    evaluate_sumo_along_axes,
    evaluate_sumo_manual_crossvalidation,
    evaluate_sumo_on_grid,
    perform_moga_optimization,
    propagate_manual_uq_with_uncertainty,
    summarize_uncertainty_samples,
)
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor
from itis_sumo.utils.helpers import create_run_dir

_logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 40


def _minimum_samples(variable_count: int, floor: int = 5) -> int:
    """Return the minimum tabular sample count Dakota can fit safely."""
    return max(floor, variable_count + 1)


def _stderr_tail(run_dir: Path | None) -> str:
    if run_dir is None:
        return ""
    logs = sorted(
        run_dir.rglob("dakota_stderr.txt"), key=lambda path: path.stat().st_mtime
    )
    if not logs:
        return ""
    lines = logs[-1].read_text(errors="replace").splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def column_scale(spec: PreprocessingSpec | None, column: str) -> str:
    """The scale asked for on ``column``; ``None`` spec ≡ all-linear (V21pf).

    One home for the lookup, shared by the session, the optimizer and the
    standalone samplers (V45ls: every value producer reads scale through here).
    """
    if spec is None:
        return "linear"
    return spec.overrides.get(column, VariableSpec()).scale


def _validate_samples(
    samples: pd.DataFrame,
    variables: Sequence[str],
    responses: Sequence[str],
    spec: PreprocessingSpec,
) -> pd.DataFrame:
    """Reduce the caller's table to the columns in play, or explain why we can't.

    Deciding that a table cannot support a surrogate is itis-sumo's job; deciding
    which rows deserve to be in the table in the first place is the caller's
    (SPEC §C). Everything raised here is a :class:`SumoInputError`, because
    everything raised here is fixable by sending different data.
    """
    if not isinstance(samples, pd.DataFrame):
        raise SumoInputError(
            f"samples must be a pandas DataFrame, got {type(samples).__name__}"
        )
    if not variables:
        raise SumoInputError("At least one variable is required")
    if not responses:
        raise SumoInputError("At least one response is required")

    duplicates = sorted({v for v in variables if list(variables).count(v) > 1})
    if duplicates:
        raise SumoInputError(f"Variables listed more than once: {duplicates}")
    response_duplicates = sorted({r for r in responses if list(responses).count(r) > 1})
    if response_duplicates:
        raise SumoInputError(f"Responses listed more than once: {response_duplicates}")
    overlap = sorted(set(variables) & set(responses))
    if overlap:
        raise SumoInputError(f"{overlap} listed as both a variable and a response")

    columns = [*variables, *responses]
    missing = [column for column in columns if column not in samples.columns]
    if missing:
        raise SumoInputError(
            f"Columns {missing} are not present in the samples. "
            f"Available columns: {sorted(map(str, samples.columns))}"
        )

    unknown_overrides = sorted(set(spec.overrides) - set(columns))
    if unknown_overrides:
        raise SumoInputError(
            f"Preprocessing overrides given for columns that are not in play: "
            f"{unknown_overrides}"
        )

    selected = samples.loc[:, columns].copy()
    try:
        selected = selected.astype(float)
    except (TypeError, ValueError) as exc:
        raise SumoInputError(
            f"Every variable and response must be numeric: {exc}"
        ) from exc

    unusable = [
        column
        for column in columns
        if not np.isfinite(selected[column].to_numpy()).all()
    ]
    if unusable:
        raise SumoInputError(
            f"Columns {unusable} contain missing or infinite values. "
            "Incomplete samples must be filtered out before they are passed in"
        )

    # A log-scale column is trained as log(value); log is undefined at or below
    # zero, so rejecting a non-positive sample here (as a SumoInputError the
    # consumer can surface) preempts the preprocessor's raw ValueError deeper in.
    non_positive = sorted(
        name
        for name, override in spec.overrides.items()
        if override.scale == "log" and (selected[name].to_numpy() <= 0).any()
    )
    if non_positive:
        raise SumoInputError(
            f"Columns {non_positive} are marked log-scale but hold values <= 0, "
            "for which the logarithm is undefined"
        )

    minimum = _minimum_samples(len(variables))
    if len(selected) < minimum:
        raise SumoInputError(
            f"{minimum} samples are required to build a surrogate over "
            f"{len(variables)} variables, but only {len(selected)} were supplied"
        )

    return selected.reset_index(drop=True)


class SumoSession:
    """A surrogate fitted once from a table of samples, then queried.

    Used as a context manager so that the run directory it needs has a defined
    lifetime: discarded when everything worked, kept when something did not
    (SPEC V24af).
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        variables: Sequence[str],
        response: str,
        *,
        preprocessing: PreprocessingSpec | None = None,
        workspace: Path | None = None,
    ) -> None:
        self._variables = tuple(variables)
        self._response = response
        self._spec = preprocessing or PreprocessingSpec()
        self._samples = _validate_samples(
            samples, self._variables, [self._response], self._spec
        )
        self._workspace = workspace
        self._run_dir: Path | None = None
        self._preprocessor: DataPreprocessor | None = None
        self._training_file: Path | None = None

    # ---------------------------------------------------------------- lifetime

    def __enter__(self) -> Self:
        if self._workspace is None:
            self._run_dir = Path(tempfile.mkdtemp(prefix="itis-sumo-"))
        else:
            self._run_dir = create_run_dir(Path(self._workspace), "sumo")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._run_dir is None:
            return False
        if exc_type is None:
            if self._workspace is None:
                shutil.rmtree(self._run_dir, ignore_errors=True)
        else:
            _logger.warning(
                "itis-sumo run failed; run directory preserved at %s", self._run_dir
            )
        return False

    # ----------------------------------------------------------------- fitting

    def fit(self) -> Self:
        """Fit the preprocessing and write the training file Dakota will read."""
        preprocessor = DataPreprocessor()
        preprocessor.setup_variables(
            input_vars=list(self._variables), output_vars=[self._response]
        )
        preprocessor.setup_log_transform(
            input_log_vars=[
                variable
                for variable in self._variables
                if self._scale_of(variable) == "log"
            ],
            output_log_vars=[
                response
                for response in [self._response]
                if self._scale_of(response) == "log"
            ],
        )
        transformed = preprocessor.fit_transform(self._samples)

        assert self._run_dir is not None
        training_file = self._run_dir / "processed_samples.dat"
        transformed.to_csv(training_file, sep=" ", index=False)

        self._preprocessor = preprocessor
        self._training_file = training_file
        return self

    # ----------------------------------------------------------------- queries

    def cross_validate(self, *, folds: int, seed: int) -> CrossValidationResult:
        response = self._mapped_response
        results = self._run_engine(
            "cross-validating the surrogate",
            evaluate_sumo_manual_crossvalidation,
            self._run_dir,
            self._training_file,
            self._mapped_variables,
            response,
            N_CROSS_VALIDATION=folds,
            seed=seed,
            has_eval_id_column=False,
        )

        missing = [
            key
            for key in (response, f"{response}_hat", f"{response}_std_hat")
            if key not in results
        ]
        if missing:
            raise SumoResultError(
                f"Cross-validation produced no {missing} values for "
                f"'{self._response}'. The surrogate may not have been trained "
                "with uncertainty estimates"
            )

        predicted_original = self._to_original_units(
            response, results[f"{response}_hat"]
        )
        return CrossValidationResult(
            response=self._response,
            observed=self._to_original_units(response, results[response]),
            predicted=predicted_original,
            predicted_std=self._to_original_std(
                response,
                results[f"{response}_std_hat"],
                {self._response: predicted_original},
            ),
            warnings=list(results.get("warnings", [])),
            seed=seed,
            effective_config=self.effective_config,
        )

    def along_axes(
        self,
        *,
        at: Mapping[str, float] | None,
        points_per_variable: int,
    ) -> AlongAxesResult:
        response = self._mapped_response
        results = self._run_engine(
            "evaluating the surrogate along its axes",
            evaluate_sumo_along_axes,
            self._run_dir,
            self._training_file,
            self._mapped_variables,
            response,
            cut_values=self._map_held_values(at),
            NSAMPLESPERVAR=points_per_variable,
        )
        if not results:
            raise SumoResultError(
                f"No axis sweeps were produced for response '{self._response}'"
            )

        assert self._preprocessor is not None
        original_names = self._preprocessor.get_inverse_mapping()
        sweeps: dict[str, AxisSweep] = {}
        for mapped_variable, axis in results.items():
            variable = original_names.get(mapped_variable, mapped_variable)
            predicted = self._to_original_units(response, axis["y_hat"])
            # A standard deviation is a width, not a position: it goes through the
            # delta-method std inverse (a no-op remap unless this response is log
            # scale), not the plain point inverse applied to ``predicted``.
            predicted_std = (
                self._to_original_std(
                    response, axis["std_hat"], {self._response: predicted}
                )
                if "std_hat" in axis
                else None
            )
            sweeps[variable] = AxisSweep(
                variable=variable,
                x=self._to_original_units(mapped_variable, axis["x"]),
                predicted=predicted,
                predicted_std=predicted_std,
            )

        return AlongAxesResult(
            response=self._response,
            sweeps=sweeps,
            effective_config=self.effective_config,
        )

    def grid(
        self,
        *,
        grid_variables: Sequence[str],
        at: Mapping[str, float] | None,
        points_per_variable: int,
    ) -> GridResult:
        """Evaluate a one-, two-, or higher-dimensional grid."""
        grid_variables = tuple(grid_variables)
        unknown = sorted(set(grid_variables) - set(self._variables))
        if not grid_variables:
            raise SumoInputError("At least one grid variable is required")
        if unknown:
            raise SumoInputError(
                f"Grid variables {unknown} are not variables of this model"
            )
        if points_per_variable < 2:
            raise SumoInputError("A grid needs at least 2 points per variable")

        results = self._run_engine(
            "evaluating the surrogate on a grid",
            evaluate_sumo_on_grid,
            self._run_dir,
            self._training_file,
            [self._mapped_name(variable) for variable in grid_variables],
            self._mapped_variables,
            self._mapped_response,
            cut_values=self._map_held_values(at),
            NSAMPLESPERVAR=points_per_variable,
        )
        if self._mapped_response not in results:
            raise SumoResultError(
                f"No grid predictions were produced for '{self._response}'"
            )

        assert self._preprocessor is not None
        original_names = self._preprocessor.get_inverse_mapping()
        converted: dict[str, list[float] | list[list[float]]] = {}
        for mapped_name, values in results.items():
            original_name = original_names.get(mapped_name, mapped_name)
            if mapped_name == self._mapped_response:
                converted[original_name] = self._inverse_nested_values(
                    mapped_name, values
                )
            else:
                converted[original_name] = self._inverse_nested_values(
                    mapped_name, values
                )
        return GridResult(
            response=self._response,
            grid_variables=grid_variables,
            data=converted,
            effective_config=self.effective_config,
        )

    def sobol(
        self,
        *,
        domains: Mapping[str, DomainSpec] | None = None,
        fixed: Mapping[str, float] | None = None,
        seed: int,
    ) -> SobolResult:
        """Compute sensitivity indices over the exploration DOMAIN (V26dd).

        The Saltelli sampling box comes from the domain, never from modeller
        distributions: ``domains`` boxes are optional (unknown names are
        rejected), anything not given is auto-inferred from the observed sample
        bounds, and a column constant in the samples stays fixed. Explicitly
        ``fixed`` values pin a factor at a caller-stated value (the
        domain-vocabulary way to freeze a factor; ⊥ a distribution's
        ``constant`` parameter). Draws are uniform across the box -- log-uniform
        under a log-scale override (V44ls), which also requires a strictly
        positive box or pin.
        """
        given = domains or {}
        given_fixed = fixed or {}
        unknown = sorted(set(given) - set(self._variables))
        if unknown:
            raise SumoInputError(
                f"Domains given for variables that are not in play: {unknown}"
            )
        unknown_fixed = sorted(set(given_fixed) - set(self._variables))
        if unknown_fixed:
            raise SumoInputError(
                f"Fixed values given for variables that are not in play: "
                f"{unknown_fixed}"
            )
        overlap = sorted(set(given_fixed) & set(given))
        if overlap:
            raise SumoInputError(f"Variables cannot be both boxed and fixed: {overlap}")
        sampling: dict[str, dict[str, float | bool]] = {}
        effective_boxes: dict[str, DomainSpec] = {}
        fixed_map: dict[str, float] = {}
        for variable in self._variables:
            log_scale = self._scale_of(variable) == "log"
            if variable in given_fixed:
                value = float(given_fixed[variable])
                if not math.isfinite(value):
                    raise SumoInputError(
                        f"Fixed value for '{variable}' must be finite, got {value}"
                    )
                if log_scale and value <= 0:
                    raise SumoInputError(
                        f"Log-scale '{variable}' fixed values must be strictly positive"
                    )
                fixed_map[variable] = value
                sampling[variable] = {"value": value}
                continue
            dom = given.get(variable)
            if dom is None:
                column = self._samples[variable]
                lo, hi = float(column.min()), float(column.max())
                if lo == hi:
                    # Constant in the samples -> fixed factor; a DomainSpec
                    # cannot express it (V47st requires minimum < maximum), and
                    # inventing a box around it would fabricate sensitivity.
                    fixed_map[variable] = lo
                    sampling[variable] = {"value": lo}
                    continue
                dom = DomainSpec(minimum=lo, maximum=hi)
            if log_scale and (dom.minimum <= 0 or dom.maximum <= 0):
                raise SumoInputError(
                    f"Log-scale '{variable}' domains must be strictly positive"
                )
            entry: dict[str, float | bool] = {
                "minimum": dom.minimum,
                "maximum": dom.maximum,
            }
            if log_scale:
                entry["log_scale"] = True
            sampling[variable] = entry
            effective_boxes[variable] = dom
        results = self._run_engine(
            "computing Sobol indices",
            evaluate_sobol_indices,
            self._run_dir,
            self._training_file,
            self._variables,
            self._mapped_response,
            sampling,
            self._preprocessor,
            seed=seed,
        )
        if "sobol" not in results:
            raise SumoResultError(
                f"No Sobol indices were produced for '{self._response}'"
            )
        masses = results["sobolOrderContributions"]
        return SobolResult(
            response=self._response,
            indices=results["sobol"],
            second_order=results["sobolSecondOrder"],
            order_contributions=OrderMasses(**masses) if masses is not None else None,
            seed=seed,
            domains=effective_boxes,
            fixed=fixed_map,
            effective_config=self.effective_config,
        )

    def uncertainty(
        self,
        *,
        distributions: Mapping[str, DistributionSpec],
        num_samples: int,
        n_histograms: int,
        seed: int,
    ) -> UncertaintyResult:
        """Propagate explicit uncertainty through the surrogate's own predictive
        uncertainty and summarise the result as a histogram + boxplot."""
        missing = sorted(set(self._variables) - set(distributions))
        unknown = sorted(set(distributions) - set(self._variables))
        if missing or unknown:
            raise SumoInputError(
                f"Distributions must cover variables exactly; missing={missing}, "
                f"unknown={unknown}"
            )
        engine_distributions = self._uq_engine_distributions(distributions)
        samples = self._run_engine(
            "propagating uncertainty",
            propagate_manual_uq_with_uncertainty,
            self._run_dir,
            self._training_file,
            self._variables,
            self._response,
            engine_distributions,
            self._preprocessor,
            num_samples,
            n_histograms=n_histograms,
            seed=seed,
        )
        summary = summarize_uncertainty_samples(samples)
        return UncertaintyResult(
            response=self._response,
            distributions=dict(distributions),
            seed=seed,
            bins_start=summary["bins_start"],
            bins_end=summary["bins_end"],
            bin_means=summary["bin_means"],
            bin_stds=summary["bin_stds"],
            q1=summary["q1"],
            median=summary["median"],
            q3=summary["q3"],
            whisker_min=summary["whisker_min"],
            whisker_max=summary["whisker_max"],
            outliers=summary["outliers"],
            mean=summary["mean"],
            std=summary["std"],
            minimum=summary["min"],
            maximum=summary["max"],
        )

    def correlations(
        self,
        *,
        distributions: Mapping[str, DistributionSpec],
        num_samples: int,
        seed: int,
    ) -> CorrelationResult:
        """Correlate each variable with the surrogate-predicted response over a
        shared Monte Carlo sample set drawn from ``distributions``.

        Scale comes from the session's own spec (V45ls): each column's samples
        and the prediction are correlated on their declared scale.
        """
        missing = sorted(set(self._variables) - set(distributions))
        unknown = sorted(set(distributions) - set(self._variables))
        if missing or unknown:
            raise SumoInputError(
                f"Distributions must cover variables exactly; missing={missing}, "
                f"unknown={unknown}"
            )
        engine_distributions = self._uq_engine_distributions(distributions)
        coefficients = self._run_engine(
            "correlating through the surrogate",
            correlate_manual_uq_samples,
            self._run_dir,
            self._training_file,
            self._variables,
            self._response,
            engine_distributions,
            self._preprocessor,
            num_samples,
            input_scales={
                variable: self._scale_of(variable) for variable in self._variables
            },
            output_scale=self._scale_of(self._response),
            seed=seed,
        )
        return CorrelationResult(
            response=self._response, seed=seed, coefficients=coefficients
        )

    def _uq_engine_distributions(
        self, distributions: Mapping[str, DistributionSpec]
    ) -> dict[str, dict[str, float | str]]:
        """Translate distributions for the sampler, flagging log-scale variables.

        A log-scale variable is sampled in the space the surrogate trains on: a
        uniform is drawn log-uniform and a normal keeps its μ/σ in ln space, so
        the raw draws are lognormal (V46rn); the surrogate preprocessor
        re-applies the log either way. A log-scale uniform needs a strictly-
        positive lower bound; a log-scale normal is positive by construction and
        needs no bounds. Anything else (e.g. ``constant``) is rejected here -- at
        the API boundary -- rather than surfacing as the sampler's raw
        ``ValueError``.
        """
        engine: dict[str, dict[str, float | str]] = {}
        for variable, spec in distributions.items():
            entry = spec.as_engine_dict()
            if self._scale_of(variable) == "log":
                if spec.distribution == "uniform":
                    if spec.minimum is None or spec.minimum <= 0:
                        raise SumoInputError(
                            f"'{variable}' is log-scale but its distribution lower "
                            "bound is not strictly positive"
                        )
                    if spec.maximum is None or spec.maximum <= spec.minimum:
                        raise SumoInputError(
                            f"'{variable}' is log-scale but its distribution upper "
                            f"bound is missing or not above its lower bound "
                            f"({spec.maximum!r} <= {spec.minimum!r})"
                        )
                elif spec.distribution != "normal":
                    raise SumoInputError(
                        f"'{variable}' is log-scale but its distribution is a "
                        f"'{spec.distribution}'; only a uniform or normal "
                        "supports log sampling"
                    )
                entry["log_scale"] = True
            engine[variable] = entry
        return engine

    def _scale_of(self, column: str) -> str:
        """The scale the caller asked for on ``column`` (default: linear)."""
        return column_scale(self._spec, column)

    def _to_original_std(
        self,
        mapped_name: str,
        std_values: Sequence[float],
        point_estimates_original: Mapping[str, Sequence[float]],
    ) -> list[float]:
        """Restore a predicted *standard deviation* to original units.

        A std is a width, not a position, so it cannot go through the ordinary
        point inverse-transform. A log-scale response in particular needs the
        multiplicative delta-method rule, which is why the point estimates (already
        back in original units) are threaded in alongside. For every other column
        this is a plain name remap, matching the pre-log behaviour exactly.
        """
        assert self._preprocessor is not None
        original = self._preprocessor.get_inverse_mapping().get(
            mapped_name, mapped_name
        )
        points = {
            name: [float(value) for value in values]
            for name, values in point_estimates_original.items()
        }
        restored = self._preprocessor.inverse_transform_output_std(
            {mapped_name: [float(value) for value in std_values]},
            point_estimates_original=points,
        )
        return [float(value) for value in restored.get(original, list(std_values))]

    def _mapped_name(self, variable: str) -> str:
        assert self._preprocessor is not None
        return self._preprocessor.input_variables[variable].mapped_name

    def _inverse_nested_values(
        self, mapped_name: str, values: object
    ) -> list[float] | list[list[float]]:
        if (
            not isinstance(values, list)
            or not values
            or not isinstance(values[0], list)
        ):
            return self._to_original_units(mapped_name, cast("Sequence[float]", values))
        return [
            self._to_original_units(mapped_name, cast("Sequence[float]", row))
            for row in values
        ]

    # -------------------------------------------------------------- internals

    @property
    def effective_config(self) -> dict[str, VariableSpec]:
        """What was actually used -- readable, but not settable in transform terms."""
        return {
            column: self._spec.overrides.get(column, VariableSpec())
            for column in (*self._variables, self._response)
        }

    @property
    def _mapped_variables(self) -> list[str]:
        assert self._preprocessor is not None
        return [
            self._preprocessor.input_variables[variable].mapped_name
            for variable in self._variables
        ]

    @property
    def _mapped_response(self) -> str:
        assert self._preprocessor is not None
        return self._preprocessor.output_variables[self._response].mapped_name

    def _map_held_values(
        self, at: Mapping[str, float] | None
    ) -> dict[str, float] | None:
        """Complete and translate the values the caller wants held fixed.

        A caller may pin only the variables they care about. Every remaining
        variable has to be given a value anyway, and it gets the same one it
        would have got had the caller said nothing at all: its mean across the
        samples.
        """
        if not at:
            return None
        unknown = sorted(set(at) - set(self._variables))
        if unknown:
            raise SumoInputError(
                f"Cannot hold {unknown} fixed: they are not variables of this model"
            )
        bad = sorted(
            name
            for name, value in at.items()
            if self._scale_of(name) == "log" and float(value) <= 0
        )
        if bad:
            raise SumoInputError(f"Cannot hold log-scale {bad} fixed at a value <= 0")
        assert self._preprocessor is not None
        held_row = {
            **self._samples.mean().to_dict(),
            **{name: float(value) for name, value in at.items()},
        }
        held = self._preprocessor.transform(pd.DataFrame([held_row]))
        mapped_variables = set(self._mapped_variables)
        return {
            name: float(value)
            for name, value in held.iloc[0].to_dict().items()
            if name in mapped_variables
        }

    def _to_original_units(
        self, mapped_name: str, values: Sequence[float]
    ) -> list[float]:
        assert self._preprocessor is not None
        original = self._preprocessor.get_inverse_mapping().get(
            mapped_name, mapped_name
        )
        restored = self._preprocessor.inverse_transform({mapped_name: list(values)})
        return [float(value) for value in restored.get(original, list(values))]

    def _run_engine(self, description: str, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except SumoError:
            raise
        except Exception as exc:
            raise SumoEngineError(
                f"Dakota failed while {description}: {exc}",
                run_dir=self._run_dir,
                stderr_tail=self._stderr_tail(),
            ) from exc

    def _stderr_tail(self) -> str:
        return _stderr_tail(self._run_dir)


def optimize_pareto_front(
    samples: pd.DataFrame,
    variables: Sequence[str],
    objectives: Mapping[str, Direction],
    *,
    domains: Mapping[str, DomainSpec],
    max_evaluations: int,
    workspace: Path | None,
    preprocessing: PreprocessingSpec | None = None,
) -> ParetoFrontResult:
    """Fit a surrogate per objective and find its Pareto-optimal trade-off front.

    A ``scale="log"`` override applies the same way everywhere else: a log-scale
    *variable* trains and is explored in log space (so its search domain is mapped
    into log space and must be strictly positive), and a log-scale *objective* is
    fitted on ``ln(y)`` with the front restored to original units on the way out.
    """
    variables = tuple(variables)
    spec = preprocessing or PreprocessingSpec()
    missing_domains = sorted(set(variables) - set(domains))
    unknown_domains = sorted(set(domains) - set(variables))
    if missing_domains or unknown_domains:
        raise SumoInputError(
            f"Domains must cover variables exactly; missing={missing_domains}, "
            f"unknown={unknown_domains}"
        )

    log_inputs = [
        variable for variable in variables if column_scale(spec, variable) == "log"
    ]
    log_objectives = [
        objective for objective in objectives if column_scale(spec, objective) == "log"
    ]

    non_positive_domain = sorted(
        name
        for name, dom in domains.items()
        if name in log_inputs and (dom.minimum <= 0 or dom.maximum <= 0)
    )
    if non_positive_domain:
        raise SumoInputError(
            f"Log-scale {non_positive_domain} domains must be strictly positive"
        )

    # The training-data positivity guard covers any log-scale objective (log is
    # undefined for <= 0 outputs) and reuses the shared validation path.
    validated = _validate_samples(samples, variables, list(objectives), spec)

    run_dir = (
        create_run_dir(Path(workspace), "sumo")
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="itis-sumo-"))
    )
    try:
        preprocessor = DataPreprocessor()
        preprocessor.setup_variables(
            input_vars=list(variables), output_vars=list(objectives)
        )
        maximize = [
            response
            for response, direction in objectives.items()
            if direction == "maximize"
        ]
        if maximize:
            preprocessor.setup_sign_switching(output_sign_switches=maximize)
        if log_inputs or log_objectives:
            preprocessor.setup_log_transform(
                input_log_vars=log_inputs, output_log_vars=log_objectives
            )
        transformed = preprocessor.fit_transform(validated)
        training_file = run_dir / "processed_samples.dat"
        transformed.to_csv(training_file, sep=" ", index=False)

        mapped_variables = [
            preprocessor.input_variables[variable].mapped_name for variable in variables
        ]
        mapped_objectives = [
            preprocessor.output_variables[response].mapped_name
            for response in objectives
        ]
        mapped_domains: dict[str, dict[str, float | str]] = {}
        for name, dom in domains.items():
            if name in log_inputs:
                dom = DomainSpec(
                    minimum=float(np.log(dom.minimum)),
                    maximum=float(np.log(dom.maximum)),
                )
            mapped_domains[preprocessor.input_variables[name].mapped_name] = (
                dom.as_engine_dict()
            )

        try:
            results = perform_moga_optimization(
                run_dir,
                training_file,
                mapped_variables,
                mapped_domains,
                mapped_objectives,
                moga_kwargs={"max_function_evaluations": max_evaluations},
            )
        except Exception as exc:
            raise SumoEngineError(
                f"Dakota failed while optimizing the Pareto front: {exc}",
                run_dir=run_dir,
                stderr_tail=_stderr_tail(run_dir),
            ) from exc

        if not results:
            raise SumoResultError("No Pareto front points were produced")

        original = preprocessor.inverse_transform(results)
        original_names = preprocessor.get_inverse_mapping()
        data = {
            original_names.get(mapped_name, mapped_name): [float(v) for v in values]
            for mapped_name, values in original.items()
        }
        if workspace is None:
            shutil.rmtree(run_dir, ignore_errors=True)
        return ParetoFrontResult(
            objectives=dict(objectives), variables=variables, data=data
        )
    except SumoError:
        if workspace is None:
            _logger.warning(
                "itis-sumo run failed; run directory preserved at %s", run_dir
            )
        raise
