"""Internal fit-then-query engine behind the public one-shot workflows.

The public functions in :mod:`itis_sumo` are deliberately thin
wrappers over a session that is fitted once and then queried many times. Keeping
that split in place from the outset is what makes the eventual public
fitted-model handle a re-export rather than a second port (SPEC V27fq).

Nothing in this module is public API.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import numpy as np
import pandas as pd

from itis_sumo.engine.evaluation import (
    _UQ_METRIC_KEYS,
    _UQ_METRICS,
    _convergence_subset_sizes,
    _cv_mean,
    _cv_spread,
    evaluate_sobol_indices,
    evaluate_sumo,
    evaluate_sumo_along_axes,
    evaluate_sumo_manual_crossvalidation,
    evaluate_sumo_on_grid,
    export_sumo_model,
    import_sumo_model,
    perform_moga_optimization,
    sample_and_propagate_uq,
    summarize_uncertainty_samples,
)
from itis_sumo.engine.helpers import create_run_dir
from itis_sumo.engine.sumo_model_store import (
    load_model_metadata,
    stage_model_for_import,
)
from itis_sumo.errors import (
    SumoEngineError,
    SumoError,
    SumoInputError,
    SumoResultError,
)
from itis_sumo.tabular.preprocessor import DataPreprocessor
from itis_sumo.tabular.processing import load_data
from itis_sumo.types import (
    DEFAULT_SEED,
    AlongAxesResult,
    AxisSweep,
    CrossValidationResult,
    Direction,
    DistributionSpec,
    DomainSpec,
    GridResult,
    OrderMasses,
    ParetoFrontResult,
    PredictionsResult,
    PreprocessingSpec,
    SobolResult,
    UncertaintyDraws,
    VariableSpec,
)

_logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 40


def _minimum_samples(variable_count: int, floor: int = 5) -> int:
    """Return the minimum tabular sample count Dakota can fit safely."""
    return max(floor, variable_count + 1)


def _require_exact_coverage(
    provided: Mapping[str, object], variables: Sequence[str], *, what: str
) -> None:
    """Guard the 'spec mapping must cover the variables exactly' contract shared
    by the UQ, correlation and MOGA entry points (distributions in the first two,
    search domains in the third)."""
    missing = sorted(set(variables) - set(provided))
    unknown = sorted(set(provided) - set(variables))
    if missing or unknown:
        raise SumoInputError(
            f"{what} must cover variables exactly; missing={missing}, unknown={unknown}"
        )


def _uq_draw_point(
    base: int, n_total: int, size: int, draw: int
) -> tuple[list[int], int]:
    """One draw's (training rows, propagation seed), deterministic per (size, draw).

    The rows are a DISTINCT without-replacement subset (no training row duplicated
    inside a subset — the same semantics `compute_cv_convergence` gives its
    bootstrap draws), and both numbers come from a SeedSequence over
    [base, size, draw] (NEP 19) INSTEAD OF one shared Generator consumed in sweep
    order — so neither a point's subset nor its randomness depends on how many
    sizes or draws came before it: raising `n_bootstrap` or adding a size to the
    sweep leaves the other points' numbers alone instead of quietly renumbering
    them (V25sd).
    """
    rng = np.random.default_rng([abs(base), size, draw])
    rows = rng.choice(n_total, size=size, replace=False).tolist()
    return rows, int(rng.integers(0, 2**31 - 1))


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
        self._loaded_model_id: str | None = None
        self._failed = False

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

    # ------------------------------------------------------------- persistence

    def save(self, *, export_format: str = "text_archive") -> str:
        """Persist the fitted SuMo to the model store; return its `sumo_model_id`.

        Runs Dakota's `export_model` in this handle's run dir and persists the
        fitted preprocessing config alongside the archive (T26eq), so `load()`
        restores the inverse transform and keeps results in original units
        (V22rs). Callers never choose the on-disk key (V12).
        """
        assert self._run_dir is not None and self._preprocessor is not None
        export_samples = self._run_dir / "save_samples.dat"
        self._preprocessor.transform(self._samples)[self._mapped_variables].to_csv(
            export_samples, sep=" ", index=False
        )
        _, sumo_model_id = self._run_engine(
            "exporting the surrogate",
            export_sumo_model,
            self._run_dir,
            self._training_file,
            export_samples,
            self._mapped_variables,
            self._mapped_response,
            export_format,
            preprocessor_config=self._preprocessor.to_config(),
        )
        return sumo_model_id

    def predict(self, samples: pd.DataFrame) -> PredictionsResult:
        """Predict the response for fresh samples -- original units in and out.

        On a fitted handle this builds the surrogate and evaluates it; on a
        loaded handle (T57rc) the surrogate is re-imported from the store
        instead (E1, R9: the archive, not the staged points, defines the model).

        Args:
            samples: One row per sample, one column per variable (original names).

        Raises:
            SumoInputError: The samples cannot be translated for this model.
            SumoResultError: The run finished without producing predictions.
            SumoEngineError: Dakota failed; the error carries the surviving run
                directory and the tail of Dakota's stderr.
        """
        assert self._run_dir is not None and self._preprocessor is not None
        evaluation_file = self._run_dir / "predict_samples.dat"
        self._preprocessor.transform(samples)[self._mapped_variables].to_csv(
            evaluation_file, sep=" ", index=False
        )
        if self._loaded_model_id is None:
            results = self._run_engine(
                "predicting with the surrogate",
                evaluate_sumo,
                self._run_dir,
                self._training_file,
                evaluation_file,
                self._mapped_variables,
                self._mapped_response,
            )
        else:
            results = self._run_engine(
                "predicting with the loaded surrogate",
                import_sumo_model,
                self._run_dir,
                self._loaded_model_id,
                evaluation_file,
                self._mapped_variables,
                self._mapped_response,
            )
        mapped = self._mapped_response
        if f"{mapped}_hat" not in results:
            raise SumoResultError(
                f"Prediction produced no {mapped}_hat values for '{self._response}'. "
                "The surrogate may not have been trained."
            )
        values = self._to_original_units(mapped, results[f"{mapped}_hat"])
        std: list[float] | None = None
        if f"{mapped}_std_hat" in results:
            std = self._to_original_std(
                mapped, results[f"{mapped}_std_hat"], {self._response: values}
            )
        return PredictionsResult(
            response=self._response,
            values=values,
            std=std,
            sumo_model_id=self._loaded_model_id,
            effective_config=self.effective_config,
        )

    def close(self) -> None:
        """End the handle's lifetime: the run dir is discarded on success and
        preserved on failure (V24af). Safe to call more than once; `__exit__`
        and garbage collection do the same."""
        if self._run_dir is None:
            return
        if self._failed:
            _logger.warning(
                "itis-sumo run failed; run directory preserved at %s", self._run_dir
            )
        elif self._workspace is None:
            shutil.rmtree(self._run_dir, ignore_errors=True)
        self._run_dir = None

    def __del__(
        self,
    ) -> None:  # best effort: an unclosed fitted handle leaks no temp dir
        run_dir = getattr(self, "_run_dir", None)
        if (
            run_dir is not None
            and not getattr(self, "_failed", True)
            and getattr(self, "_workspace", "unset") is None
        ):
            shutil.rmtree(run_dir, ignore_errors=True)

    # ----------------------------------------------------------------- queries

    def cross_validate(self, *, folds: int, seed: int) -> CrossValidationResult:
        """Held-out predictions for every sample via manual K-fold CV.

        Each fold is predicted by a surrogate that never saw it, so the result
        is an honest read of generalization, not memorization. ``observed``,
        ``predicted`` and ``predicted_std`` stay aligned to the caller's rows;
        a fold Dakota abandons leaves NaNs there with the reason in
        ``warnings`` instead of failing the run.
        """
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
        """One sweep per variable: response along each axis with every other
        variable held at its centre (or at your ``at`` values).

        The cheap shape of the response surface — one line per variable where
        ``grid`` would be a hypercube.
        """
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
            if mapped_name == self._mapped_response:
                converted[self._response] = self._inverse_nested_values(
                    mapped_name, values
                )
            elif mapped_name.startswith(f"{self._mapped_response}_"):
                # Companion series (the predicted-std column) hang off the
                # MAPPED response name (y1_std); V21pf requires the ORIGINAL
                # name with the suffix kept -- a mapped-name leak gets
                # camelCased by consumers and the frontend cannot look it up.
                suffix = mapped_name[len(self._mapped_response) :]
                converted[f"{self._response}{suffix}"] = self._inverse_nested_values(
                    mapped_name, values
                )
            else:
                original_name = original_names.get(mapped_name, mapped_name)
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

    def sample_uncertainty(
        self,
        *,
        distributions: Mapping[str, DistributionSpec],
        num_samples: int,
        n_histograms: int,
        seed: int,
    ) -> UncertaintyDraws:
        """Propagate explicit uncertainty through the surrogate's own predictive
        uncertainty, returning the raw material (V53md).

        Draws are the per-variable Monte-Carlo inputs, predictions and spreads
        travel per draw, and `propagated` is the realisation grid that
        `summarize_uncertainty` re-summarises — the caller may correlate it,
        plot it, or summarise it themselves (V52rc).
        """
        _require_exact_coverage(distributions, self._variables, what="Distributions")
        engine_distributions = self._uq_engine_distributions(distributions)
        pieces = self._run_engine(
            "propagating uncertainty",
            sample_and_propagate_uq,
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
        return UncertaintyDraws(
            response=self._response,
            distributions=dict(distributions),
            seed=seed,
            num_samples=num_samples,
            n_histograms=n_histograms,
            draws=pieces["draws"],
            predictions=pieces["predictions"],
            prediction_stds=pieces["prediction_stds"],
            propagated=[list(row) for row in pieces["propagated"]],
        )

    def uq_convergence_curve(
        self,
        *,
        distributions: Mapping[str, DistributionSpec],
        min_samples: int = 5,
        max_points: int = 5,
        n_bootstrap: int = 1,
        num_samples: int = 1000,
        n_histograms: int = 100,
        seed: int = DEFAULT_SEED,
    ) -> list[dict[str, Any]]:
        """How the PROPAGATED output distribution moves as the training set grows.

        Refits the surrogate on each subset size and propagates the SAME stated
        ``distributions`` through each refit, so the only thing changing along the
        curve is how much data the surrogate saw. This is the sibling of the CV
        convergence series, not the same question: that one asks whether the
        surrogate's own predictions and spread are trustworthy at N, this one asks
        what the input uncertainty you declared does to the response at N. Each
        size therefore carries the two-sided tail (`p1`/``p5``/``median``/``p95``/
        ``p99``) alongside ``mean``/``std``/``skew``/``kurt`` — two-sided because a
        lower-tail failure (a QoI going negative, say) is invisible to a curve that
        only plots the upper tail.

        Composition, not a second implementation: each subset is a fresh session
        carrying this handle's own ``preprocessing`` (V21pf keeps the column scales
        there, so a QoI declared log is refit as log at every size), the
        propagation is this handle's own :meth:`sample_uncertainty` (V53md), and
        every statistic is read off the one summary point the codebase has. No
        number here is re-derived, and the series is returned in
        ``compute_cv_convergence``'s point shape so ``render_convergence`` consumes
        it with ``metric="uq_p99"`` and no new rendering (V54hz).

        Args:
            distributions: Per-variable Monte-Carlo input distributions, held
                FIXED along the curve and covering the variables exactly. They are
                a specification, not something inferred from the subset in play —
                the subset changes, the claim about the world does not.
            min_samples: Smallest training subset; may not be below the
                ``max(5, n_variables + 1)`` Dakota floor.
            max_points: Cap on the number of subset sizes, evenly spaced from
                ``min_samples`` to the full sample count — the same rule, and the
                same parameter names, as `compute_cv_convergence`, so the two
                curves share an x-axis. Bounds the number of engine runs.
            n_bootstrap: Distinct random subsets per size (never resampling with
                replacement, so no training row is duplicated inside a subset),
                matching `compute_cv_convergence`'s semantics: ``metrics_std`` then
                answers the same question on both curves. ``1`` means one
                deterministic first-``n``-rows subset per size. The full-sample
                point always runs once — there is only one subset of that size.
            num_samples: Monte-Carlo draws per propagation.
            n_histograms: Predictive-spread injections per draw, kept at
                :meth:`sample_uncertainty`'s default so the curve's LAST point is
                exactly the number ``summarize_uncertainty`` reports for the whole
                training set today — the point of a convergence curve inside an app
                is that it ends at the number the app shows. The consequence is
                that the pool is input uncertainty PLUS the surrogate's own spread
                at that N; ``n_histograms=1`` gives pure input propagation.
            seed: Base seed. Per-(size, draw) seeds are derived from it (V25sd), so
                the whole sweep is reproducible from the one number you see in the
                result.

        Returns:
            One entry per subset size, in increasing ``n_samples``:

                {
                    "n_samples": int,          # training rows this point refit on
                    "metrics": {token: float},  # all nine, aggregated over draws
                    "metrics_std": {token: float},  # spread ACROSS draws (0.0 for
                                               # a single deterministic subset)
                    "draws_by_metric": {token: [float, ...]},  # raw per-draw value
                    "n_bootstrap": int,        # draws that PRODUCED this point
                    "num_samples": int,        # MC draws behind each of them
                    "n_histograms": int,
                    "distributions": dict,     # the spec held fixed along the curve
                    "seed": int,               # base seed; draws derive from it
                    "warnings": [str, ...],    # per-(size, draw) refit aborts
                }

            Tokens are ``uq_mean``, ``uq_std``, ``uq_p1``, ``uq_p5``,
            ``uq_median``, ``uq_p95``, ``uq_p99``, ``uq_skew``, ``uq_kurt``. All
            carry the response's ORIGINAL units (V22rs) except ``uq_skew``/
            ``uq_kurt``, which are dimensionless — say so on the axis with
            ``yaxis_title`` when plotting those two.

            None semantics: a subset whose surrogate Dakota cannot train is
            skipped, its reason appended to that point's ``warnings``, and the
            point is aggregated over the draws that did land — the same warn-and-
            continue `compute_cv_diagnostics` gives an abandoned fold (B22wk,
            B23rt), because small subsets are exactly where a near-degenerate fit
            is expected ⊥ exceptional. A size whose every draw failed keeps its
            place on the axis with ``None`` metrics rather than vanishing: a gap
            you can see beats a curve that quietly got shorter. ``n_bootstrap``
            then reads as the number of draws that survived, which is also why the
            raw per-draw values are returned — ``uq_skew``/``uq_kurt`` sampling
            error is a function of that count (SE ≈ √(24/M)).

            A settled tail is not automatically a measured one: at small N the
            declared distributions can reach further than the subset ever covered,
            and the surrogate's answer out there is extrapolation. That is the
            expected shape of this curve — the instrument exists to show the tails
            wild at low N ∧ settling as data grows — so nothing here clips,
            censors, or caveats them away.

        Raises:
            SumoInputError: Distributions do not cover the variables exactly, a
                log-scale column's distribution cannot be sampled in the fit space,
                ``min_samples`` is below the Dakota floor, the handle is a loaded
                model (this retrains, T57rc scope), or there is no room for a
                single subset size.
            SumoEngineError: Only for a failure outside the sweep's per-draw
                handling; a refit that aborts is reported in ``warnings``.
        """
        if self._loaded_model_id is not None:
            raise SumoInputError(
                "uq_convergence_curve retrains at every subset size and is not "
                "supported on a loaded model (T57rc scope): fit it again."
            )
        _require_exact_coverage(distributions, self._variables, what="Distributions")
        # Validate the distributions against the FIT SPACE before the first engine
        # call: inside the sweep a raise is read as a refit abort, and a caller
        # error would otherwise come back as a curve of None instead of a message.
        self._uq_engine_distributions(distributions)

        n_total = len(self._samples)
        floor = _minimum_samples(len(self._variables))
        if min_samples < floor:
            raise SumoInputError(
                f"min_samples={min_samples} is below the {floor} samples needed to "
                f"fit a surrogate over {len(self._variables)} variables"
            )
        sizes = _convergence_subset_sizes(n_total, min_samples, max_points)
        if not sizes:
            raise SumoInputError(
                f"{n_total} training samples leave no room for a size between "
                f"min_samples={min_samples} and the full count; fit on more samples "
                "or lower min_samples"
            )

        series: list[dict[str, Any]] = []
        for size in sizes:
            if size >= n_total or n_bootstrap <= 1:
                # the single-draw point shape: first `size` rows on the base seed
                draws_spec = [(list(range(size)), seed)]
            else:
                draws_spec = [
                    _uq_draw_point(seed, n_total, size, draw)
                    for draw in range(n_bootstrap)
                ]

            draw_metrics: list[dict[str, float]] = []
            warnings: list[str] = []
            for draw, (rows, draw_seed) in enumerate(draws_spec):
                try:
                    draw_metrics.append(
                        self._uq_propagation_metrics(
                            rows,
                            distributions,
                            num_samples=num_samples,
                            n_histograms=n_histograms,
                            seed=draw_seed,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - draw-scoped warn & continue
                    warnings.append(
                        f"n_samples={size} draw {draw} was skipped: refitting or "
                        f"propagating on that subset failed ({exc})"
                    )
                    _logger.warning(
                        "uq_convergence_curve: size %d draw %d skipped: %s",
                        size,
                        draw,
                        exc,
                    )

            if draw_metrics:
                draws_by_metric = {
                    token: [metrics[token] for metrics in draw_metrics]
                    for token in _UQ_METRICS
                }
                # Same aggregation the CV series uses, so `metrics_std` means one
                # thing on both curves: mean over the draws, spread ACROSS them.
                metrics = {
                    token: _cv_mean(values) for token, values in draws_by_metric.items()
                }
                metrics_std = {
                    token: _cv_spread(values)
                    for token, values in draws_by_metric.items()
                }
            else:
                # a comprehension, not `dict.fromkeys(..., [])`: the latter would
                # hand every token the SAME list object
                draws_by_metric = {token: [] for token in _UQ_METRICS}
                metrics = {token: None for token in _UQ_METRICS}
                metrics_std = {token: None for token in _UQ_METRICS}

            series.append(
                {
                    "n_samples": size,
                    "metrics": metrics,
                    "metrics_std": metrics_std,
                    "draws_by_metric": draws_by_metric,
                    "n_bootstrap": len(draw_metrics),
                    "num_samples": num_samples,
                    "n_histograms": n_histograms,
                    "distributions": dict(distributions),
                    "seed": seed,
                    "warnings": warnings,
                }
            )
        return series

    def _uq_propagation_metrics(
        self,
        rows: Sequence[int],
        distributions: Mapping[str, DistributionSpec],
        *,
        num_samples: int,
        n_histograms: int,
        seed: int,
    ) -> dict[str, float]:
        """Refit on ONE subset and read the nine statistics off that propagation.

        The subset session gets this handle's own ``preprocessing``, so the scale
        each column is fit in is the declared one at every size (V21pf) ∧ the
        preprocessing itself is re-fit on the subset, exactly as a consumer
        refitting by hand would. Its run dir lands under this handle's run dir, so
        the artifacts follow the handle's lifetime (V24af) ∧ a preserved subset
        run dir is the evidence of the abort named in that point's ``warnings``.
        """
        assert self._run_dir is not None
        subset = self._samples.iloc[list(rows)]
        with SumoSession(
            subset,
            self._variables,
            self._response,
            preprocessing=self._spec,
            workspace=self._run_dir,
        ) as session:
            draws = session.fit().sample_uncertainty(
                distributions=distributions,
                num_samples=num_samples,
                n_histograms=n_histograms,
                seed=seed,
            )
        summary = summarize_uncertainty_samples(
            np.asarray(draws.propagated, dtype=float)
        )
        return {token: float(summary[key]) for token, key in _UQ_METRIC_KEYS.items()}

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
        if self._loaded_model_id is not None and self._preprocessor is not None:
            inputs = self._preprocessor.input_variables
            outputs = self._preprocessor.output_variables
            scale_of = {**inputs, **outputs}
            return {
                column: VariableSpec(
                    scale="log" if scale_of[column].log_transform else "linear"
                )
                for column in (*self._variables, self._response)
                if column in scale_of
            }
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
        if self._loaded_model_id is not None and function is not import_sumo_model:
            raise SumoInputError(
                "This query retrains and is not supported on a loaded model "
                f"(T57rc scope): '{getattr(function, '__name__', function)}' "
                "needs a freshly fitted SuMo handle."
            )
        try:
            return function(*args, **kwargs)
        except SumoError:
            raise
        except Exception as exc:
            self._failed = True
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
    _require_exact_coverage(domains, variables, what="Domains")

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


# --------------------------------------------------------------------------- handle
# The handle-primary surface (§C architecture, V27fq, V52rc). One-shot workflow
# functions are thin fit-then-query delegations over exactly this machinery, so
# the handle is a re-export, not a second port.

SuMo = SumoSession


def fit(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
    *,
    preprocessing: PreprocessingSpec | None = None,
    workspace: Path | None = None,
) -> SumoSession:
    """Fit a SuMo and hand back the live handle (the catalog is its methods).

    The run directory lives as long as the handle (V24af): discard it with
    `close()` (or a `with` block) on success; a failed run keeps it and the
    raised `SumoEngineError` carries its path and a stderr tail.

    Args:
        samples: One row per sample, one column per variable or response.
        variables: Columns to treat as inputs.
        response: Column to predict.
        preprocessing: Optional per-column overrides (domain vocabulary only).
        workspace: If given, working files are written here and kept.

    Raises:
        SumoInputError: The samples or configuration cannot support a fit.
        SumoEngineError: Dakota failed while fitting; the error carries the
            surviving run directory and the tail of Dakota's stderr.
    """
    handle = SumoSession(
        samples,
        variables,
        response,
        preprocessing=preprocessing,
        workspace=workspace,
    )
    handle.__enter__()
    handle.fit()
    return handle


def load(sumo_model_id: str, *, workspace: Path | None = None) -> SumoSession:
    """Reopen a stored SuMo by id (E1 restart escape hatch, V12/V13).

    `predict` runs against the re-imported surrogate; queries that retrain
    (cross-validate, Sobol, MOGA, ...) raise `SumoInputError` at T57rc scope.
    Models stored before preprocessing persistence (T26eq) are rejected with a
    re-export instruction rather than returning wrong-unit numbers.
    """
    metadata = load_model_metadata(sumo_model_id)
    if metadata.preprocessor_config is None:
        raise SumoInputError(
            f"SuMo model '{sumo_model_id}' predates preprocessing persistence "
            "(T26eq); re-export it from a fitted handle to use load()."
        )
    preprocessor = DataPreprocessor.from_config(metadata.preprocessor_config)
    inverse = preprocessor.get_inverse_mapping()

    handle = SumoSession.__new__(SumoSession)
    handle._spec = PreprocessingSpec()
    handle._workspace = workspace
    handle._loaded_model_id = sumo_model_id
    handle._failed = False
    handle._preprocessor = preprocessor
    handle._variables = tuple(inverse.get(v, v) for v in metadata.input_descriptors)
    handle._response = inverse.get(
        metadata.output_descriptor, metadata.output_descriptor
    )
    if workspace is None:
        handle._run_dir = Path(tempfile.mkdtemp(prefix="itis-sumo-"))
    else:
        handle._run_dir = create_run_dir(Path(workspace), "sumo")
    _, staged_training_file = stage_model_for_import(sumo_model_id, handle._run_dir)
    handle._training_file = staged_training_file
    handle._samples = load_data(staged_training_file)
    return handle
