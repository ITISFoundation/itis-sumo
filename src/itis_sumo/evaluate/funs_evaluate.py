import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold

from itis_sumo.core.dakota_object import DakotaObject
from itis_sumo.config.funs_create_dakota_conf import (
    create_moga_optimization_conffile,
    create_sumo_crossvalidation_conffile,
    create_sumo_evaluation_conffile,
    create_sumo_manual_crossvalidation_conffile,
    create_uq_propagation_conffile,
)
from itis_sumo.data.funs_data_processing import (
    create_grid_samples,
    create_samples_along_axes,
    extract_predictions_along_axes,
    extract_predictions_gridpoints,
    get_bounds_uniform_distributions,
    get_results,
    load_data,
    process_input_file,
    sanitize_varname,
    sanitize_varnames,
    sanitize_varnames_dict,
)

_logger = logging.getLogger(__name__)


def retrieve_csv_result(
    csv_file_path: str, inputs: dict[str, float], outputs: list[str] | None = None
) -> dict[str, float]:
    """
    Retrieve the result from a csv file.
    """

    df = pd.read_csv(csv_file_path)

    for col in inputs:
        if col not in df.columns:
            raise ValueError(f"Input {col} not in the csv file. Columns are: {df.columns.values}")

    if outputs is not None:
        for col in outputs:
            if col not in df.columns:
                raise ValueError(
                    f"Output {col} not in the csv file. Columns are: {df.columns.values}"
                )
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1), outputs]
    else:
        result = df.loc[np.all(df[inputs.keys()] == inputs.values(), axis=1)]
    # Check if the result is empty or has multiple rows
    if len(result) == 0:
        raise ValueError(f"No result found for inputs {inputs}.")
    if len(result) > 1:
        raise ValueError(f"Multiple results found for inputs {inputs}.")

    return result.iloc[0].to_dict()


def evaluate_sumo_along_axes(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    response_var: str,
    cut_values: dict[str, float] | None = None,
    sumo_import_name: str | None = None,
    sumo_export_name: str | None = None,
    NSAMPLESPERVAR: int = 21,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    label_converter: Callable | None = None,
    MAKEPLOT: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Given a training data to create a SuMo, generate it, and plot the profile along the central axes
    (e.g. all variables but the sweeped one will be set to its central value).
    No callback is necessary (everything internal to Dakota).

    Log / Linear scale of the variable is inferred its name; mean value is taken in the corresponding scale.
    Plots scales (after SuMo creation and sampling) can be either linear or logarithmic.
    """
    # sanitize variable names
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    cut_values = sanitize_varnames(cut_values) if cut_values else None

    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_SWEEP_INPUT_FILE = create_samples_along_axes(
        run_dir, data, input_vars, NSAMPLESPERVAR, cut_values=cut_values
    )

    if sumo_import_name:
        models_dir = run_dir.parent / "models"
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Models dir {models_dir} does not exist, but SuMo import is trying to copy files there"
            )
        for file in models_dir.glob(f"{sumo_import_name}*"):
            shutil.copy(file, run_dir)

    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        sumo_import_name=sumo_import_name,
        sumo_export_name=sumo_export_name,
        samples_file=PROCESSED_SWEEP_INPUT_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    results = extract_predictions_along_axes(run_dir, response_var, input_vars, NSAMPLESPERVAR)
    return results


### TODO refactor in new MMUX-compatible version (like above)
def propagate_uq(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    means: dict[str, float],
    stds: dict[str, float],
    n_samples: int = 1000,
    xscale: Literal["linear", "log"] = "linear",
    label_converter: Callable | None = None,
) -> list[float]:
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)
    means = {sanitize_varnames(k): v for k, v in means.items()}
    stds = {sanitize_varnames(k): v for k, v in stds.items()}

    # create dakota file
    dakota_conf = create_uq_propagation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        input_variables=input_vars,
        input_means=means,
        input_stds=stds,
        output_responses=[output_response],
        n_samples=n_samples,
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    x = get_results(run_dir / "predictions.dat", output_response)
    return x.tolist()


def _parse_crossvalidation_outputlogs(log_output: str, N_CROSS_VALIDATION: int):
    variable_name_pattern = (
        rf"Surrogate quality metrics \({N_CROSS_VALIDATION}-fold CV\) for (\w+):"
    )
    metrics_pattern = r"\s+(root_mean_squared|sum_abs|mean_abs|max_abs)\s+([\d.e+-]+|nan)"

    # Find all occurrences of variable names in the log
    variables = re.findall(variable_name_pattern, log_output)

    # Split the log output by the variable name to handle each output separately
    log_parts = re.split(variable_name_pattern, log_output)
    log_parts = log_parts[1:]  # Skip the first part (before the first variable name)

    # Dictionary to hold the parsed results for each output variable
    parsed_error_metrics = {}

    # Loop through the log parts, and extract metrics for each output variable
    for i, variable in enumerate(variables):
        # The log part after each variable name contains the metrics section for that variable
        metrics_section = log_parts[2 * i + 1]  # The log part immediately after the variable name

        ## remove the training error of the next variable
        metrics_section = metrics_section.split("build (training) points")[0]

        # Find all the surrogate quality metrics for this particular output variable
        metrics_matches = re.findall(metrics_pattern, metrics_section)

        if metrics_matches:
            metrics = {metric: value for metric, value in metrics_matches}
            parsed_error_metrics[variable] = metrics
        else:
            parsed_error_metrics[variable] = "No surrogate quality metrics found."

    _logger.debug("Parsed cross-validation metrics: %s", parsed_error_metrics)
    return parsed_error_metrics


def evaluate_sumo_crossvalidation(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
):
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)

    dakota_conf = create_sumo_crossvalidation_conffile(
        PROCESSED_TRAINING_FILE,
        input_vars,
        [output_response],
        N_CROSS_VALIDATION=N_CROSS_VALIDATION,
    )
    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    # `dakobj.run` writes captured stdout to "dakota_stdout.txt" in run_dir (see DakotaObject.run)
    stdout_file = run_dir / "dakota_stdout.txt"
    log_output = stdout_file.read_text() if stdout_file.is_file() else ""
    parsed_error_metrics = _parse_crossvalidation_outputlogs(log_output, N_CROSS_VALIDATION)

    return parsed_error_metrics


def evaluate_sumo_manual_crossvalidation(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
):
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)

    all_observations = load_data(PROCESSED_TRAINING_FILE)[output_response].astype(float)
    n_samples = len(all_observations)
    indices = np.arange(n_samples)
    all_predictions = np.full(n_samples, np.nan)
    all_stds = np.full(n_samples, np.nan)
    kf = KFold(n_splits=N_CROSS_VALIDATION, shuffle=True, random_state=42)
    parse_warnings: list[str] = []

    for fold, (_, val_idx) in enumerate(kf.split(indices)):
        fold_run_dir = run_dir / f"fold_{fold}"
        os.makedirs(fold_run_dir, exist_ok=True)

        # Create Dakota config for this fold
        dakota_conf = create_sumo_manual_crossvalidation_conffile(
            fold_run_dir,
            PROCESSED_TRAINING_FILE,
            input_vars,
            output_response,
            validation_indices=val_idx.tolist(),
            dakota_conf_file=fold_run_dir / "dakota_config.in",
        )
        # V43 (root SPEC §T33 / B23): a Dakota fold run is non-deterministic in practice
        # (near-degenerate surrogate training can make Dakota abort a fold and never write
        # `predictions.dat`). A bare `load_data` on the missing file raised an opaque 500.
        # Treat a Dakota abort like B22/B23's "warn & continue": skip the fold (its
        # validation points stay NaN), surface a fold-scoped warning, and keep going so the
        # endpoint returns instead of 500ing the whole cross-validation.
        try:
            dakobj = DakotaObject()
            dakobj.run(dakota_conf, fold_run_dir)
        except Exception as exc:
            parse_warnings.append(
                f"Fold {fold} of {N_CROSS_VALIDATION}: Dakota run raised "
                f"({exc}); skipping this fold"
            )
            continue

        predictions_path = fold_run_dir / "predictions.dat"
        if not predictions_path.is_file():
            parse_warnings.append(
                f"Fold {fold} of {N_CROSS_VALIDATION}: Dakota did not produce "
                f"{predictions_path} (surrogate training likely aborted); skipping this fold"
            )
            continue

        # Extract predictions for this fold and store in the correct positions.
        # `on_malformed_row="heal_or_drop"` recovers/skips rows corrupted by Dakota's
        # own tabular-writer defect (B22) instead of failing the whole fold; rows are
        # matched back to `val_idx` via their own `_eval_id` (their sequential position
        # among this fold's validation points), NOT by array position, so a dropped row
        # simply leaves that sample's prediction as NaN rather than misaligning the rest.
        try:
            predictions_df = load_data(
                predictions_path,
                on_malformed_row="heal_or_drop",
                warnings=parse_warnings,
            )
        except Exception as exc:
            parse_warnings.append(
                f"Fold {fold} of {N_CROSS_VALIDATION}: failed to parse Dakota's "
                f"predictions.dat ({exc}); skipping this fold"
            )
            continue
        # V41: Dakota can fail to produce the requested-output surrogate for a fold
        # and emit a predictions.dat whose header OMITS `output_response` (B23) — a
        # bare `predictions_df[output_response]` would raise an opaque KeyError('y1')
        # 500. Skip the fold (its validation points stay NaN), surface a fold+file
        # scoped warning, and continue per B22's "warn & continue" resolution.
        if output_response not in predictions_df.columns:
            parse_warnings.append(
                f"Fold {fold} of {N_CROSS_VALIDATION}: predictions.dat for "
                f"{fold_run_dir / 'predictions.dat'} has columns "
                f"{list(predictions_df.columns)}, missing '{output_response}' — "
                f"output surrogate not produced for this fold; skipping it"
            )
            continue
        fold_eval_ids = predictions_df["_eval_id"].astype(int).values
        fold_predictions = predictions_df[output_response].astype(float).values
        for local_eval_id, prediction in zip(fold_eval_ids, fold_predictions):
            if not (1 <= local_eval_id <= len(val_idx)):
                parse_warnings.append(
                    f"Fold {fold}: predictions.dat row had _eval_id={local_eval_id}, "
                    f"outside this fold's {len(val_idx)} validation points; dropped"
                )
                continue
            all_predictions[val_idx[local_eval_id - 1]] = prediction
        print(f"Fold {fold} predictions: {fold_predictions}")
        print(f"Validation indices: {val_idx}")

        if (fold_run_dir / "variances.dat").is_file():
            variances_df = load_data(
                fold_run_dir / "variances.dat",
                on_malformed_row="heal_or_drop",
                warnings=parse_warnings,
            )
            # V41: same missing-column guard for the variance output (B23).
            if output_response + "_variance" not in variances_df.columns:
                parse_warnings.append(
                    f"Fold {fold} of {N_CROSS_VALIDATION}: variances.dat for "
                    f"{fold_run_dir / 'variances.dat'} has columns "
                    f"{list(variances_df.columns)}, missing '{output_response}_variance' — "
                    f"output variance not produced for this fold; skipping it"
                )
            else:
                var_eval_ids = variances_df["_eval_id"].astype(int).values
                fold_var = variances_df[output_response + "_variance"].astype(float).values
                for local_eval_id, var in zip(var_eval_ids, fold_var):
                    if not (1 <= local_eval_id <= len(val_idx)):
                        parse_warnings.append(
                            f"Fold {fold}: variances.dat row had _eval_id={local_eval_id}, "
                            f"outside this fold's {len(val_idx)} validation points; dropped"
                        )
                        continue
                    all_stds[val_idx[local_eval_id - 1]] = np.sqrt(var)

    result = {
        output_response: all_observations.tolist(),
        output_response + "_hat": all_predictions.tolist(),
        output_response + "_std_hat": all_stds.tolist(),
    }
    if parse_warnings:
        result["warnings"] = parse_warnings
    return result


def compute_cv_accuracy_metrics(
    actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray
) -> dict[str, float]:
    """Compute RMSE/MAE/sum-abs/max-abs directly from paired CV actual/predicted values.

    Unlike `_parse_crossvalidation_outputlogs`, this does not depend on parsing Dakota's
    stdout (which `evaluate_sumo_crossvalidation` no longer captures) - it derives the
    same metrics straight from the actual/predicted arrays already produced by
    `evaluate_sumo_manual_crossvalidation`.
    """
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual (shape {actual_arr.shape}) and predicted (shape {predicted_arr.shape}) "
            "must have the same shape"
        )
    # NaN entries come from CV points evaluate_sumo_manual_crossvalidation couldn't fill
    # (a fold row dropped per B22) - exclude them rather than let a single dropped point
    # turn every metric into NaN.
    valid = ~np.isnan(actual_arr) & ~np.isnan(predicted_arr)
    actual_arr = actual_arr[valid]
    predicted_arr = predicted_arr[valid]
    # If every CV fold failed (B23), no valid pairs remain: `np.max` has no identity for
    # an empty array and would raise, turning a degraded-but-recoverable CV run into an
    # opaque 500. Report NaN metrics instead, matching the NaN/None-tolerant
    # `CVAccuracyMetrics` response model.
    if actual_arr.size == 0:
        return {
            "root_mean_squared": float("nan"),
            "sum_abs": float("nan"),
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
        }
    residuals = actual_arr - predicted_arr
    abs_residuals = np.abs(residuals)
    return {
        "root_mean_squared": float(np.sqrt(np.mean(residuals**2))),
        "sum_abs": float(np.sum(abs_residuals)),
        "mean_abs": float(np.mean(abs_residuals)),
        "max_abs": float(np.max(abs_residuals)),
    }


def compute_paired_ttest(
    actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray
) -> dict[str, float]:
    """Paired t-test (`scipy.stats.ttest_rel`) on CV actual-vs-predicted residuals.

    Tests H0: mean(actual - predicted) == 0, i.e. no systematic surrogate bias.
    A low p-value (e.g. < 0.05) indicates the surrogate is systematically biased
    beyond what scalar MAE/RMSE reveal.
    """
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"actual (shape {actual_arr.shape}) and predicted (shape {predicted_arr.shape}) "
            "must have the same shape"
        )
    # See compute_cv_accuracy_metrics: NaN entries are dropped CV rows (B22), not real data.
    valid = ~np.isnan(actual_arr) & ~np.isnan(predicted_arr)
    actual_arr = actual_arr[valid]
    predicted_arr = predicted_arr[valid]
    if actual_arr.size < 2:
        raise ValueError("Paired t-test requires at least 2 CV samples")
    result = ttest_rel(actual_arr, predicted_arr)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def _convergence_subset_sizes(n_total: int, min_samples: int, max_points: int) -> list[int]:
    """Evenly-spaced, deduplicated subset sizes from `min_samples` up to `n_total`."""
    if n_total < min_samples:
        return []
    n_points = min(max_points, n_total - min_samples + 1)
    sizes = np.linspace(min_samples, n_total, num=n_points, dtype=int).tolist()
    seen: set[int] = set()
    unique_sizes = []
    for size in sizes:
        if size not in seen:
            seen.add(size)
            unique_sizes.append(size)
    return unique_sizes


def compute_cv_convergence(
    run_dir: Path,
    training_file: Path,
    input_vars: list[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
    min_samples: int = 5,
    max_points: int = 5,
) -> list[dict[str, float]]:
    """Rerun manual K-fold CV at increasing training-sample-count subsets.

    Reuses `evaluate_sumo_manual_crossvalidation` (the same compute path
    `/sumo_cross_validation` already runs) on the first `n` rows of `training_file` for
    each subset size, deriving RMSE via `compute_cv_accuracy_metrics` at each step.
    Subset sizes are evenly spaced between `min_samples` and the full sample count,
    capped at `max_points` to bound the number of extra Dakota reruns (⊥ single-N
    snapshot only). Returns a `{n_samples, metric}` series for accuracy-vs-N plotting.
    """
    n_total = len(load_data(training_file))
    subset_sizes = _convergence_subset_sizes(n_total, min_samples, max_points)

    series = []
    for n in subset_sizes:
        subset_file = process_input_file(
            training_file,
            columns_to_keep=input_vars + [output_response],
            filter_N_samples=n,
            suffix=f"convergence_{n}",
        )
        subset_run_dir = run_dir / f"convergence_{n}"
        os.makedirs(subset_run_dir, exist_ok=True)
        n_folds = min(N_CROSS_VALIDATION, n)
        result = evaluate_sumo_manual_crossvalidation(
            subset_run_dir,
            subset_file,
            input_vars,
            output_response,
            N_CROSS_VALIDATION=n_folds,
        )
        metrics = compute_cv_accuracy_metrics(
            result[output_response], result[output_response + "_hat"]
        )
        series.append({"n_samples": n, "metric": metrics["root_mean_squared"]})

    return series


def evaluate_sumo(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    PROCESSED_EVALUATION_SAMPLES_FILE: Path,
    input_vars: list[str],
    response_var: str,
) -> dict[str, list[float]]:
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)

    """Given a training data to create a SuMo, generate it, and evaluate on the training data.
    No callback is necessary (everything internal to Dakota).
    """
    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        samples_file=PROCESSED_EVALUATION_SAMPLES_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {
        response_var + "_hat": get_results(run_dir / "predictions.dat", response_var).tolist()
    }
    if (run_dir / "variances.dat").is_file():
        variances = get_results(run_dir / "variances.dat", response_var + "_variance")
        results[response_var + "_std_hat"] = np.sqrt(variances).tolist()

    return results


def evaluate_sumo_on_grid(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    grid_vars: list[str],
    input_vars: list[str],
    response_var: str,
    cut_values: dict[str, float] | None = None,
    # sumo_import_name: Optional[str] = None,
    # sumo_export_name: Optional[str] = None,
    NSAMPLESPERVAR: int = 21,
    # xscale: Literal["linear", "log"] = "linear",
    # yscale: Literal["linear", "log"] = "linear",
    # label_converter: Optional[Callable] = None,
    # MAKEPLOT: bool = False,
) -> dict[str, list[float]]:
    """Given a training data to create a SuMo, generate it, and evaluate on a grid of points.
    The grid is created by sweeping the variables in `grid_vars` over their min and max values,
    while the other variables in `input_vars` are set to their central values.
    The grid is created by sampling `NSAMPLESPERVAR` points per variable.
    The results are returned as a dictionary, where the keys are the variable names and the values are lists of values (inputs / predictions).
    No callback is necessary (everything internal to Dakota).

    Log / Linear scale of the variable is inferred its name; mean value is taken in the corresponding scale.
    Plots scales (after SuMo creation and sampling) can be either linear or logarithmic.
    """
    NPOINTSPERDIMENSION = [NSAMPLESPERVAR] * len(
        input_vars
    )  # default number of points per dimension
    grid_vars = sanitize_varnames(grid_vars)
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    cut_values = sanitize_varnames(cut_values) if cut_values else None

    # create sweeps data
    data = pd.read_csv(PROCESSED_TRAINING_FILE, sep=" ")
    PROCESSED_GRIDPOINTS_INPUT_FILE = create_grid_samples(
        run_dir=run_dir,
        grid_vars=grid_vars,
        input_vars=input_vars,
        mins=[
            data[var].min() for var in input_vars
        ],  ## TODO it is here that we should use the distribution values (passed directly from the frontend)
        cut_values=(
            [cut_values[var] for var in input_vars]
            if cut_values
            else [data[var].mean() for var in input_vars]
        ),
        maxs=[
            data[var].max() for var in input_vars
        ],  # TODO it is here that we should use the distribution values (passed directly from the frontend)
        n_points_per_dimension=NPOINTSPERDIMENSION,
    )

    # create dakota file
    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        # sumo_import_name=sumo_import_name,
        # sumo_export_name=sumo_export_name,
        ### TODO once this works, try to get it to work wo evaluation (or just one sample, if not possible?)
        samples_file=PROCESSED_GRIDPOINTS_INPUT_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
    )

    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = extract_predictions_gridpoints(run_dir, response_var, input_vars, NSAMPLESPERVAR)

    if len(grid_vars) == 2:  ## this is not necessary for 3D
        output = np.array(results[response_var])
        reshape_indices = [
            NPOINTSPERDIMENSION[i] for i in range(len(input_vars)) if input_vars[i] in grid_vars
        ]
        if grid_vars[0] in input_vars[:2] and grid_vars[1] in input_vars[:2]:
            ## reshape fills in row order. For some reason, this needs to be done reversed in XY / YX cases
            ## but NOT for any other input combination...
            output = output.reshape(reshape_indices[::-1]).T
        else:
            output = output.reshape(reshape_indices)
        input_vars_in_grid_vars = [var for var in input_vars if var in grid_vars]
        transpose_indices = [
            input_vars_in_grid_vars.index(grid_vars[i]) for i in range(len(grid_vars))
        ]
        final_output = output.transpose(
            transpose_indices[::-1]
        )  # ZX, XZ, YZ, ZY work; but not YX, XY. Why???
        results[response_var] = final_output.tolist()

    return results


def perform_moga_optimization(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    distributions: dict[str, dict[str, float]],
    output_responses: list[str],
    moga_kwargs: dict,
) -> dict[str, list[float | int]]:
    _logger.debug("Minimizing responses: %s", ", ".join(output_responses))

    input_vars = sanitize_varnames(input_vars)
    output_responses = [sanitize_varnames(resp) for resp in output_responses]
    distributions = sanitize_varnames(distributions)

    # assumes uniform distribution for MOGA - raises Error otherwise
    lower_bounds, upper_bounds = get_bounds_uniform_distributions(input_vars, distributions)

    # create dakota file
    dakota_conf = create_moga_optimization_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        input_variables=input_vars,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        output_responses=output_responses,
        moga_kwargs=moga_kwargs,
        dakota_conf_file=run_dir / "dakota_config.in",
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {}
    for res in output_responses:
        x = get_results(run_dir / "predictions.dat", res)
        results[res] = x.tolist()
    for inv in input_vars:
        x = get_results(run_dir / "predictions.dat", inv)
        results[inv] = x.tolist()

    return results


SOBOL_BASE_SAMPLES = 1024
"""Fixed base sample count N for Sobol' Saltelli sampling (V36).

A widely-used practical default (SALib/scipy tutorials, Saltelli et al.
"Global Sensitivity Analysis: The Primer") that gives reliable index estimates
for typical dimensionalities. Deliberately decoupled from the frontend's
shared UQ ``numSamples`` (used by Histogram/Correlation) since Sobol' cost is
``SOBOL_BASE_SAMPLES * (d_varying + 2)`` -- reusing the UQ default of 10,000
rounds to 16,384 and multiplies out to 5-10x more surrogate evaluations than
necessary for reliable rankings.
"""

SOBOL_BOOTSTRAP_RESAMPLES = 1000
"""Bootstrap resamples for first/total-order confidence intervals (V37).

Resampling reuses the already-computed f_A/f_B/f_AB evaluations (row indices
resampled with replacement) -- no extra ``evaluate_sumo()`` calls, so this is
effectively free relative to the surrogate evaluation cost.
"""

SOBOL_BOOTSTRAP_CONFIDENCE = 0.95


def evaluate_sobol_indices(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    response_var: str,
    distributions: dict[str, dict],
    preprocessor,
    seed: int | None = None,
) -> dict[str, dict]:
    """Compute Sobol' first-order, total-order, and second-order sensitivity indices.

    Generates Saltelli A/B/AB sample matrices locally (honouring per-input
    distributions via ``scipy.stats.rv_continuous.ppf``), evaluates all samples
    in ONE batch through ``evaluate_sumo()`` (surrogate-only, Dakota does not run
    ``variance_based_decomp`` itself), then applies ``scipy.stats.sobol_indices``
    for first-order + total-order indices plus a closed-form second-order
    (pairwise interaction) estimator.

    Second-order estimator: the Jansen/Saltelli 2010 algebraic identity for
    pairwise interaction variances, V_ij = ((V_Ti - V_i) + (V_Tj - V_j) -
    Σ_{k≠i,j} (V_Tk - V_k)) / 2, which is exact for d=3 with no third-order
    interactions (validated against Ishigami analytical reference, §R1) and a
    standard approximation for d > 3.  See: Saltelli, A. (2010). "Variance
    based sensitivity analysis of model output. Design and estimator for the
    total sensitivity index." Computer Physics Communications, 181(2), 259-270.

    Base sample count is the fixed ``SOBOL_BASE_SAMPLES`` constant (V36), NOT
    the frontend's shared UQ ``numSamples`` -- Sobol' has fundamentally
    different sample-cost scaling (multiplicative in ``d_varying``) than the
    other UQ views, so it uses its own well-established practical default.
    First/total-order indices also come with bootstrap confidence intervals
    (V37), computed by resampling the existing evaluations (no extra cost).

    Args:
        run_dir: Dakota run directory for intermediate files.
        PROCESSED_TRAINING_FILE: Path to the preprocessed training data file.
        input_vars: Original (unmapped) input variable names.
        response_var: Mapped response variable name (as known to Dakota).
        distributions: Dict mapping original var names to distribution params
            (``{"distribution": "normal", "mean":, "std":}`` /
            ``{"distribution": "uniform", "min":, "max":}`` /
            ``{"distribution": "constant", "value":}``).
        preprocessor: Fitted ``DataPreprocessor`` for transforming samples.
        seed: Random seed for reproducibility (numpy/scipy RNGs accept 0).

    Returns:
        Dict with keys ``"sobol"`` (``{var: {"main": float, "total": float,
        "main_ci_low": float, "main_ci_high": float, "total_ci_low": float,
        "total_ci_high": float}}``) and ``"sobolSecondOrder"``
        (``{varA: {varB: float}}`` symmetric over unordered pairs, no self-pair).
    """
    import math

    import pandas as pd
    from scipy.stats import norm, sobol_indices, uniform
    from scipy.stats.qmc import Sobol

    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    distributions = {
        sanitize_varname(k): sanitize_varnames_dict(v) for k, v in distributions.items()
    }

    # --- 1. Separate constant vs. varying input variables ---
    constant_vars: dict[str, float] = {}
    varying_vars: list[str] = []
    for var in input_vars:
        dist_info = distributions[var]
        if dist_info["distribution"] == "constant":
            constant_vars[var] = float(dist_info["value"])
        else:
            varying_vars.append(var)

    d_varying = len(varying_vars)

    # Build frozen scipy distributions with .ppf for each varying variable
    ppfs = {}
    for var in varying_vars:
        dist_info = distributions[var]
        dist_type = dist_info["distribution"]
        if dist_type == "normal":
            ppfs[var] = norm(loc=dist_info["mean"], scale=dist_info["std"])
        elif dist_type == "uniform":
            ppfs[var] = uniform(loc=dist_info["min"], scale=dist_info["max"] - dist_info["min"])
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    # --- 2. Fixed base sample count, rounded up to next power of 2 (V36) ---
    if d_varying == 0:
        # All variables are constant — indices are trivially zero
        sobol = {
            var: {
                "main": 0.0,
                "total": 0.0,
                "main_ci_low": 0.0,
                "main_ci_high": 0.0,
                "total_ci_low": 0.0,
                "total_ci_high": 0.0,
            }
            for var in input_vars
        }
        return {"sobol": sobol, "sobolSecondOrder": {}}

    n = 2 ** math.ceil(math.log2(max(SOBOL_BASE_SAMPLES, 2)))

    # --- 3. Generate Saltelli A/B sample matrices via Sobol' QMC ---
    sampler = Sobol(d=2 * d_varying, seed=seed, scramble=True)
    U = sampler.random(n)  # shape (n, 2*d_varying)
    U_A = U[:, :d_varying]
    U_B = U[:, d_varying:]

    # Map through ppf to get real-space A and B
    A = np.column_stack([ppfs[var].ppf(U_A[:, i]) for i, var in enumerate(varying_vars)])
    B = np.column_stack([ppfs[var].ppf(U_B[:, i]) for i, var in enumerate(varying_vars)])

    # --- 4. Build AB_i matrices: A with column i replaced by B's column i ---
    # (Saltelli 2010 convention: AB_i uses B's values for variable i, A's for the rest)
    AB = np.empty((d_varying, n, d_varying))
    for i in range(d_varying):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        AB[i] = AB_i

    # --- 5. Concatenate into one big sample matrix, restore constant columns ---
    # Layout: A (n) + B (n) + AB_0 (n) + AB_1 (n) + ... + AB_{d-1} (n)
    all_samples_varying = np.vstack([A, B] + [AB[i] for i in range(d_varying)])

    # Build DataFrame with varying variables only
    df_varying = pd.DataFrame(all_samples_varying, columns=pd.Index(varying_vars))

    # Add constant columns (fixed values for all rows)
    for var, val in constant_vars.items():
        df_varying[var] = val

    # Reorder columns to match original input_vars order
    df_samples = df_varying[input_vars]

    # --- 6. Transform and write processed samples, call evaluate_sumo ONCE ---
    SAMPLES_FILE = run_dir / "sobol_samples.csv"
    df_samples.to_csv(SAMPLES_FILE, index=False)

    df_samples_transformed = preprocessor.transform(df_samples)
    PROCESSED_SAMPLES_FILE = run_dir / "sobol_samples_processed.csv"
    df_samples_transformed.to_csv(PROCESSED_SAMPLES_FILE, sep=" ", index=False)

    mapped_input_vars = [preprocessor.input_variables[var].mapped_name for var in input_vars]
    results = evaluate_sumo(
        run_dir,
        PROCESSED_TRAINING_FILE,
        PROCESSED_SAMPLES_FILE,
        mapped_input_vars,
        response_var,
    )

    prediction_key = response_var + "_hat"
    if prediction_key not in results:
        raise ValueError(
            f"Surrogate evaluation did not produce '{prediction_key}'. "
            f"Available keys: {list(results.keys())}."
        )

    # --- 7. Split the single batch of predictions back into f_A, f_B, f_AB_i ---
    all_preds = np.asarray(results[prediction_key])
    total_rows = n * (d_varying + 2)
    if len(all_preds) != total_rows:
        raise ValueError(
            f"Expected {total_rows} predictions (n={n}, d_varying={d_varying}), "
            f"got {len(all_preds)}."
        )

    idx = 0
    f_A = all_preds[idx : idx + n].reshape(1, n)  # shape (1, n)
    idx += n
    f_B = all_preds[idx : idx + n].reshape(1, n)  # shape (1, n)
    idx += n
    f_AB = np.empty((d_varying, 1, n))
    for i in range(d_varying):
        f_AB[i] = all_preds[idx : idx + n].reshape(1, 1, n)
        idx += n

    # --- 8. Compute first-order/total-order + bootstrap CIs (V37) ---
    # Bootstrap resamples the already-computed f_A/f_B/f_AB evaluations (row
    # indices, with replacement) -- no extra evaluate_sumo() calls, effectively free.
    rng = np.random.default_rng(seed)
    if d_varying == 1:
        # scipy.stats.sobol_indices squeezes to scalar when d=1 and s=1,
        # causing an internal "item assignment" error.  For a single variable
        # the Saltelli 2010 estimators reduce to simple formulas:
        #   S_1  = Cov(f_A, f_AB_0) / Var(f_A)
        #   ST_1 = 0.5 * Var(f_A - f_AB_0) / Var(f_A)
        fA_flat = f_A.ravel()
        fAB_flat = f_AB[0].ravel()
        var_f = float(np.var(fA_flat))
        if var_f == 0:
            first_order = np.array([0.0])
            total_order = np.array([0.0])
            first_order_ci = np.array([[0.0, 0.0]])
            total_order_ci = np.array([[0.0, 0.0]])
        else:
            cov_val = float(np.mean((fA_flat - fA_flat.mean()) * (fAB_flat - fAB_flat.mean())))
            first_order = np.array([cov_val / var_f])
            total_order = np.array([0.5 * np.mean((fA_flat - fAB_flat) ** 2) / var_f])

            boot_s1 = np.empty(SOBOL_BOOTSTRAP_RESAMPLES)
            boot_st = np.empty(SOBOL_BOOTSTRAP_RESAMPLES)
            for b in range(SOBOL_BOOTSTRAP_RESAMPLES):
                idx_resample = rng.integers(0, n, size=n)
                fa_b = fA_flat[idx_resample]
                fab_b = fAB_flat[idx_resample]
                var_b = np.var(fa_b)
                if var_b == 0:
                    boot_s1[b] = 0.0
                    boot_st[b] = 0.0
                    continue
                cov_b = np.mean((fa_b - fa_b.mean()) * (fab_b - fab_b.mean()))
                boot_s1[b] = cov_b / var_b
                boot_st[b] = 0.5 * np.mean((fa_b - fab_b) ** 2) / var_b
            alpha = (1 - SOBOL_BOOTSTRAP_CONFIDENCE) / 2
            first_order_ci = np.array(
                [[np.percentile(boot_s1, 100 * alpha), np.percentile(boot_s1, 100 * (1 - alpha))]]
            )
            total_order_ci = np.array(
                [[np.percentile(boot_st, 100 * alpha), np.percentile(boot_st, 100 * (1 - alpha))]]
            )
    else:
        si = sobol_indices(func={"f_A": f_A, "f_B": f_B, "f_AB": f_AB}, n=n)
        # np.squeeze in scipy can collapse to scalar when d_varying=1; ensure 1-d
        first_order = np.atleast_1d(si.first_order)  # shape (d_varying,)
        total_order = np.atleast_1d(si.total_order)  # shape (d_varying,)
        boot = si.bootstrap(
            confidence_level=SOBOL_BOOTSTRAP_CONFIDENCE, n_resamples=SOBOL_BOOTSTRAP_RESAMPLES
        )
        first_order_ci = np.column_stack(
            [
                np.atleast_1d(boot.first_order.confidence_interval.low),
                np.atleast_1d(boot.first_order.confidence_interval.high),
            ]
        )
        total_order_ci = np.column_stack(
            [
                np.atleast_1d(boot.total_order.confidence_interval.low),
                np.atleast_1d(boot.total_order.confidence_interval.high),
            ]
        )

    # --- 9. Compute second-order indices S_ij for every unordered pair ---
    # Using the Jansen/Saltelli 2010 closed-form estimator:
    #   S_ij = ((S_Ti - S_i) + (S_Tj - S_j) - Σ_{k≠i,j} (S_Tk - S_k)) / 2
    # This is exact for d=3 with no third-order interactions (validated vs Ishigami §R1).
    sobol_second_order: dict[str, dict[str, float]] = {}
    if d_varying >= 2:
        higher_order = total_order - first_order  # U_k = S_Tk - S_k
        for ii in range(d_varying):
            for jj in range(ii + 1, d_varying):
                other_sum = float(np.sum(higher_order) - higher_order[ii] - higher_order[jj])
                s_ij = (float(higher_order[ii] + higher_order[jj]) - other_sum) / 2.0
                var_a = varying_vars[ii]
                var_b = varying_vars[jj]
                if var_a not in sobol_second_order:
                    sobol_second_order[var_a] = {}
                sobol_second_order[var_a][var_b] = float(s_ij)
                # Symmetric entry
                if var_b not in sobol_second_order:
                    sobol_second_order[var_b] = {}
                sobol_second_order[var_b][var_a] = float(s_ij)

    # --- 10. Assemble final response (all requested input_vars, constants as zeros) ---
    sobol: dict[str, dict[str, float]] = {}
    for i, var in enumerate(input_vars):
        if var in constant_vars:
            # Constant variable: zero variance, Sobol' index is undefined/zero.
            # A constant input contributes no variance to the output, so its
            # first-order and total-order indices are both zero by definition.
            sobol[var] = {
                "main": 0.0,
                "total": 0.0,
                "main_ci_low": 0.0,
                "main_ci_high": 0.0,
                "total_ci_low": 0.0,
                "total_ci_high": 0.0,
            }
        else:
            idx_varying = varying_vars.index(var)
            sobol[var] = {
                "main": float(first_order[idx_varying]),
                "total": float(total_order[idx_varying]),
                "main_ci_low": float(first_order_ci[idx_varying][0]),
                "main_ci_high": float(first_order_ci[idx_varying][1]),
                "total_ci_low": float(total_order_ci[idx_varying][0]),
                "total_ci_high": float(total_order_ci[idx_varying][1]),
            }

    # Only np.isfinite validated — small-N Monte Carlo noise can yield small
    # negative estimates; do NOT clip or reject negative values (§V32).
    for var in sobol:
        for key in (
            "main",
            "total",
            "main_ci_low",
            "main_ci_high",
            "total_ci_low",
            "total_ci_high",
        ):
            val = sobol[var][key]
            if not np.isfinite(val):
                raise ValueError(f"Sobol' index for {var}.{key} is not finite: {val}")
    for var_a in sobol_second_order:
        for var_b in sobol_second_order[var_a]:
            val = sobol_second_order[var_a][var_b]
            if not np.isfinite(val):
                raise ValueError(f"Second-order Sobol' index {var_a}:{var_b} is not finite: {val}")

    return {"sobol": sobol, "sobolSecondOrder": sobol_second_order}


if __name__ == "__main__":
    _logger.info("Dakota evaluation module executed")
