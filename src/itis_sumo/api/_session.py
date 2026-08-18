"""Internal fit-then-query engine behind the public one-shot workflows.

The public functions in :mod:`itis_sumo.api.workflows` are deliberately thin
wrappers over a session that is fitted once and then queried many times. Keeping
that split in place from the outset is what makes the eventual public
fitted-model handle a re-export rather than a second port (SPEC V27fq).

Nothing in this module is public API.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pandas as pd

from itis_sumo.api.errors import (
    SumoEngineError,
    SumoError,
    SumoInputError,
    SumoResultError,
)
from itis_sumo.api.types import (
    AlongAxesResult,
    AxisSweep,
    CrossValidationResult,
    PreprocessingSpec,
    VariableSpec,
)
from itis_sumo.evaluate.funs_evaluate import (
    evaluate_sumo_along_axes,
    evaluate_sumo_manual_crossvalidation,
)
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor

# The Dakota sufficiency rule. Named in job vocabulary for historical reasons;
# aliased here so that vocabulary does not leak into the API layer (SPEC V19cn).
# T24cm moves the rule itself and only this import line changes.
from itis_sumo.preprocess.models import required_completed_jobs as _minimum_samples
from itis_sumo.utils.helpers import create_run_dir

_logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 40


def _validate_samples(
    samples: pd.DataFrame,
    variables: Sequence[str],
    response: str,
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
    if not response:
        raise SumoInputError("A response is required")

    duplicates = sorted({v for v in variables if list(variables).count(v) > 1})
    if duplicates:
        raise SumoInputError(f"Variables listed more than once: {duplicates}")
    if response in variables:
        raise SumoInputError(
            f"'{response}' is listed as both a variable and the response"
        )

    columns = [*variables, response]
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
    logarithmic = sorted(
        name for name, override in spec.overrides.items() if override.scale == "log"
    )
    if logarithmic:
        raise SumoInputError(
            f"Logarithmic scale is not supported yet (requested for {logarithmic}); "
            "it arrives together with the domain/distribution split"
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

    minimum = _minimum_samples(list(variables))
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
            samples, self._variables, self._response, self._spec
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

        return CrossValidationResult(
            response=self._response,
            observed=self._to_original_units(response, results[response]),
            predicted=self._to_original_units(response, results[f"{response}_hat"]),
            predicted_std=self._to_original_units(
                response, results[f"{response}_std_hat"]
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
            has_eval_id_column=False,
        )
        if not results:
            raise SumoResultError(
                f"No axis sweeps were produced for response '{self._response}'"
            )

        original_names = self._preprocessor.get_inverse_mapping()
        sweeps: dict[str, AxisSweep] = {}
        for mapped_variable, axis in results.items():
            variable = original_names.get(mapped_variable, mapped_variable)
            sweeps[variable] = AxisSweep(
                variable=variable,
                x=self._to_original_units(mapped_variable, axis["x"]),
                predicted=self._to_original_units(response, axis["y_hat"]),
                # A standard deviation is a width, not a position: it is reported
                # as produced rather than shifted back through the transform.
                predicted_std=(
                    [float(value) for value in axis["std_hat"]]
                    if "std_hat" in axis
                    else None
                ),
            )

        return AlongAxesResult(
            response=self._response,
            sweeps=sweeps,
            effective_config=self.effective_config,
        )

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
        if self._run_dir is None:
            return ""
        logs = sorted(
            self._run_dir.rglob("dakota_stderr.txt"),
            key=lambda path: path.stat().st_mtime,
        )
        if not logs:
            return ""
        lines = logs[-1].read_text(errors="replace").splitlines()
        return "\n".join(lines[-_STDERR_TAIL_LINES:])
