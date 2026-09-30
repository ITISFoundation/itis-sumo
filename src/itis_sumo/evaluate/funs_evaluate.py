import logging
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold

from itis_sumo.config.funs_create_dakota_conf import (
    add_surrogate_model,
    create_moga_optimization_conffile,
    create_sumo_crossvalidation_conffile,
    create_sumo_evaluation_conffile,
    create_sumo_manual_crossvalidation_conffile,
    create_uq_propagation_conffile,
    infer_has_eval_id_column_from_filename,
)
from itis_sumo.core.dakota_object import DakotaObject
from itis_sumo.core.sumo_model_store import stage_model_for_import, store_exported_model
from itis_sumo.data.funs_data_processing import (
    compute_correlation_indices,
    create_grid_samples,
    create_manual_uq_samples,
    create_samples_along_axes,
    extract_predictions_along_axes,
    extract_predictions_gridpoints,
    get_bounds_uniform_distributions,
    get_results,
    load_data,
    process_input_file,
    resolve_log_scale,
    sanitize_varnames,
    scale_distribution,
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
            raise ValueError(
                f"Input {col} not in the csv file. Columns are: {df.columns.values}"
            )

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
    has_eval_id_column: bool | None = None,
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
        has_eval_id_column=has_eval_id_column,
    )

    # run dakota
    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)
    results = extract_predictions_along_axes(
        run_dir, response_var, input_vars, NSAMPLESPERVAR
    )
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


def propagate_manual_uq_with_uncertainty(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    distributions: dict[str, dict[str, float | str]],
    preprocessor,
    num_samples: int,
    n_histograms: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Propagate explicit per-variable uncertainty through a surrogate's own
    predictive uncertainty.

    Draws ``num_samples`` per-variable samples from ``distributions`` (uniform /
    normal / constant, in the caller's original units), evaluates the surrogate
    once, then injects the surrogate's predictive std via the erfinv trick
    (``sqrt(2) * erfinv(U) ~ N(0, 1)`` for ``U ~ Uniform(-1, 1)``), repeated
    ``n_histograms`` times to characterise realisation-to-realisation spread.

    This mirrors the historical ``/manual_uq_propagation_with_uncertainty``
    Flask route rather than ``propagate_uq`` (Dakota-native, normal-only, no
    predictive-uncertainty injection) -- see test_metamodeling_analytical.py.

    Returns:
        A ``(n_histograms, num_samples)`` array of propagated output samples,
        already inverse-transformed to the caller's original units.
    """
    from scipy.special import erfinv

    # NOTE: input_vars/output_response/distributions must stay in the caller's
    # original (unsanitized) form here -- preprocessor.input_variables and
    # preprocessor.output_variables are keyed by original names (the
    # preprocessor is fit before any sanitization happens), and
    # preprocessor.transform() looks samples up by those same original column
    # names. Sanitizing eagerly breaks both lookups for any var name containing
    # characters sanitize_varnames rewrites. Dakota-safe names are obtained
    # correctly below via preprocessor.input_variables[var].mapped_name.
    samples = create_manual_uq_samples(input_vars, distributions, num_samples, seed)
    df_samples = pd.DataFrame(samples)
    SAMPLES_FILE = run_dir / "manual_uq_samples.csv"
    df_samples.to_csv(SAMPLES_FILE, index=False)

    df_samples_transformed = preprocessor.transform(df_samples)
    PROCESSED_SAMPLES_FILE = run_dir / "manual_uq_samples_processed.csv"
    df_samples_transformed.to_csv(PROCESSED_SAMPLES_FILE, sep=" ", index=False)

    mapped_input_vars = [
        preprocessor.input_variables[var].mapped_name for var in input_vars
    ]
    mapped_response = preprocessor.output_variables[output_response].mapped_name
    results = evaluate_sumo(
        run_dir,
        PROCESSED_TRAINING_FILE,
        PROCESSED_SAMPLES_FILE,
        mapped_input_vars,
        mapped_response,
    )

    prediction_key = mapped_response + "_hat"
    uncertainty_key = mapped_response + "_std_hat"
    if prediction_key not in results or uncertainty_key not in results:
        raise ValueError(
            f"Cannot propagate uncertainty without '{prediction_key}' and "
            f"'{uncertainty_key}' predictions. Available keys: {list(results.keys())}."
        )

    prediction = np.asarray(results[prediction_key])
    uncertainty = np.asarray(results[uncertainty_key])

    rng = np.random.default_rng(seed)
    all_results_transformed = np.empty((n_histograms, num_samples), dtype=float)
    for i in range(n_histograms):
        r = np.sqrt(2) * erfinv(rng.uniform(-1 + 1e-10, 1 - 1e-10, size=num_samples))
        all_results_transformed[i, :] = prediction + r * uncertainty

    all_samples_dict = {mapped_response: all_results_transformed.flatten().tolist()}
    all_samples_original = preprocessor.inverse_transform(all_samples_dict)
    return np.asarray(all_samples_original[output_response]).reshape(
        n_histograms, num_samples
    )


def correlate_manual_uq_samples(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    distributions: dict[str, dict[str, float | str]],
    preprocessor,
    num_samples: int,
    *,
    input_scales: Mapping[str, str],
    output_scale: str,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Correlate each input variable with the surrogate-predicted response over
    a shared Monte Carlo sample set.

    Draws ``num_samples`` per-variable samples from ``distributions`` (in the
    caller's original units), evaluates the surrogate once, then correlates
    every input's samples against the inverse-transformed prediction -- the
    workflow behind the historical ``/flask/dakota/compute_correlation_indices``
    route (#470), owned here so no consumer re-implements the sampling/
    surrogate/correlation chain.

    Correlations run on values mapped onto each column's own scale (V45ls):
    ``input_scales``/``output_scale`` are REQUIRED keyword arguments -- a
    producer of values cannot forget to decide a scale (``TypeError``, never a
    silent linear default). Pearson moves under a log reparametrization;
    Spearman is monotone-invariant by construction.
    """
    samples = create_manual_uq_samples(input_vars, distributions, num_samples, seed)
    df_samples = pd.DataFrame(samples)
    SAMPLES_FILE = run_dir / "correlation_samples.csv"
    df_samples.to_csv(SAMPLES_FILE, index=False)

    df_samples_transformed = preprocessor.transform(df_samples)
    PROCESSED_SAMPLES_FILE = run_dir / "correlation_samples_processed.csv"
    df_samples_transformed.to_csv(PROCESSED_SAMPLES_FILE, sep=" ", index=False)

    mapped_input_vars = [
        preprocessor.input_variables[var].mapped_name for var in input_vars
    ]
    mapped_response = preprocessor.output_variables[output_response].mapped_name
    results = evaluate_sumo(
        run_dir,
        PROCESSED_TRAINING_FILE,
        PROCESSED_SAMPLES_FILE,
        mapped_input_vars,
        mapped_response,
    )

    prediction_key = mapped_response + "_hat"
    if prediction_key not in results:
        raise ValueError(
            f"Cannot compute correlation indices without '{prediction_key}' "
            f"predictions. Available keys: {list(results.keys())}."
        )

    predictions_original = preprocessor.inverse_transform(
        {mapped_response: results[prediction_key]}
    )
    return compute_correlation_indices(
        df_samples,
        predictions_original[output_response],
        input_vars,
        input_scales=input_scales,
        output_scale=output_scale,
    )


def summarize_uncertainty_samples(
    values: np.ndarray, num_bins: int | None = None
) -> dict:
    """Compute histogram + boxplot summary statistics from propagated UQ samples.

    ``values`` is ``(n_histograms, num_samples)``: histogram bin heights are
    averaged (with their std) across histogram realisations, while boxplot and
    summary statistics are computed on the flattened pool of all samples.
    """
    all_values_flat = values.flatten()
    if num_bins is None:
        num_bins = min(50, max(10, values.shape[1] // 10))

    hist_min = float(np.percentile(all_values_flat, 1))
    hist_max = float(np.percentile(all_values_flat, 99))
    if hist_min == hist_max:
        hist_range = max(1e-10, abs(hist_min) * 1e-6)
        hist_min -= hist_range
        hist_max += hist_range

    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)
    histograms = np.array(
        [
            np.histogram(values[i, :], bins=bin_edges, density=True)[0]
            for i in range(values.shape[0])
        ]
    )
    bin_means = np.mean(histograms, axis=0)
    bin_stds = np.std(histograms, axis=0)

    q1 = float(np.percentile(all_values_flat, 25))
    median = float(np.percentile(all_values_flat, 50))
    q3 = float(np.percentile(all_values_flat, 75))
    iqr = q3 - q1
    whisker_min = max(hist_min, q1 - 1.5 * iqr)
    whisker_max = min(hist_max, q3 + 1.5 * iqr)
    outliers = all_values_flat[
        (all_values_flat < whisker_min) | (all_values_flat > whisker_max)
    ]

    return {
        "bins_start": hist_min,
        "bins_end": hist_max,
        "bin_means": bin_means.tolist(),
        "bin_stds": bin_stds.tolist(),
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_min": whisker_min,
        "whisker_max": whisker_max,
        "outliers": outliers.tolist(),
        "mean": float(np.mean(all_values_flat)),
        "std": float(np.std(all_values_flat)),
        "min": float(np.min(all_values_flat)),
        "max": float(np.max(all_values_flat)),
    }


def _parse_crossvalidation_outputlogs(log_output: str, N_CROSS_VALIDATION: int):
    variable_name_pattern = (
        rf"Surrogate quality metrics \({N_CROSS_VALIDATION}-fold CV\) for (\w+):"
    )
    metrics_pattern = (
        r"\s+(root_mean_squared|sum_abs|mean_abs|max_abs)\s+([\d.e+-]+|nan)"
    )

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
        metrics_section = log_parts[
            2 * i + 1
        ]  # The log part immediately after the variable name

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
    parsed_error_metrics = _parse_crossvalidation_outputlogs(
        log_output, N_CROSS_VALIDATION
    )

    return parsed_error_metrics


def evaluate_sumo_manual_crossvalidation(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    output_response: str,
    N_CROSS_VALIDATION: int = 5,
    seed: int = 42,
    has_eval_id_column: bool | None = None,
):
    input_vars = sanitize_varnames(input_vars)
    output_response = sanitize_varnames(output_response)

    all_observations = load_data(PROCESSED_TRAINING_FILE)[output_response].astype(float)
    n_samples = len(all_observations)
    indices = np.arange(n_samples)
    all_predictions = np.full(n_samples, np.nan)
    all_stds = np.full(n_samples, np.nan)
    kf = KFold(n_splits=N_CROSS_VALIDATION, shuffle=True, random_state=seed)
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
            has_eval_id_column=has_eval_id_column,
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
        except Exception as exc:  # noqa: BLE001 - fold-scoped warn & continue (B23)
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
        except Exception as exc:  # noqa: BLE001 - fold-scoped warn & continue (B22)
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
                fold_var = (
                    variances_df[output_response + "_variance"].astype(float).values
                )
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


def _convergence_subset_sizes(
    n_total: int, min_samples: int, max_points: int
) -> list[int]:
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
        response_var + "_hat": get_results(
            run_dir / "predictions.dat", response_var
        ).tolist()
    }
    if (run_dir / "variances.dat").is_file():
        variances = get_results(run_dir / "variances.dat", response_var + "_variance")
        results[response_var + "_std_hat"] = np.sqrt(variances).tolist()

    return results


def export_sumo_model(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    PROCESSED_EVALUATION_SAMPLES_FILE: Path,
    input_vars: list[str],
    response_var: str,
    export_format: str = "text_archive",
) -> tuple[dict[str, list[float]], str]:
    """Build+evaluate a SuMo surrogate and persist it to the model store (E1, T12).

    Behaves like `evaluate_sumo` but also has Dakota export the trained
    surrogate (`export_model`), then hands the resulting archive files off to
    `sumo_model_store` under a freshly minted `sumo_model_id` (V12) -- callers
    never choose the on-disk key.

    Returns the usual `evaluate_sumo` predictions dict plus that `sumo_model_id`.
    """
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)
    export_prefix = (
        "export"  # internal Dakota-side prefix; the real key is sumo_model_id
    )

    dakota_conf = create_sumo_evaluation_conffile(
        build_file=PROCESSED_TRAINING_FILE,
        samples_file=PROCESSED_EVALUATION_SAMPLES_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
        sumo_export_name=export_prefix,
        export_import_format=export_format,
    )

    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {
        response_var + "_hat": get_results(
            run_dir / "predictions.dat", response_var
        ).tolist()
    }
    if (run_dir / "variances.dat").is_file():
        variances = get_results(run_dir / "variances.dat", response_var + "_variance")
        results[response_var + "_std_hat"] = np.sqrt(variances).tolist()

    training_samples_file = str(PROCESSED_TRAINING_FILE.resolve())
    surrogate_conf_block = add_surrogate_model(
        sumo_export_name=export_prefix,
        export_import_format=export_format,
        training_samples_file=training_samples_file,
        has_eval_id_column=infer_has_eval_id_column_from_filename(
            training_samples_file
        ),
    )
    sumo_model_id = store_exported_model(
        run_dir=run_dir,
        training_file=PROCESSED_TRAINING_FILE,
        surrogate_conf_block=surrogate_conf_block,
        input_descriptors=input_vars,
        output_descriptor=response_var,
        export_prefix=export_prefix,
        export_format=export_format,
    )

    return results, sumo_model_id


def import_sumo_model(
    run_dir: Path,
    sumo_model_id: str,
    PROCESSED_EVALUATION_SAMPLES_FILE: Path,
    input_vars: list[str],
    response_var: str,
) -> dict[str, list[float]]:
    """Evaluate samples through a previously exported SuMo model (E1, T12).

    No re-training / no training file from the caller: stages the stored
    archive + the stored copy of the original training-data file into
    `run_dir` (falling back to a synthesized header-only placeholder with a
    loud warning if that copy is missing -- safe because R9 established the
    surrogate is fully reconstructed from the archive; the points file's
    values are never read back, only its descriptors matter), then runs
    Dakota's `import_model` (V11: same surrogate-model conf block as at
    export time, bar the export_model -> import_model swap -- the staged
    file supplies the `import_build_points_file` keyword the block still
    needs, R2).

    Validates that `input_vars`/`response_var` (order-sensitive) match the
    model's stored descriptors (V10) before evaluating, since Dakota's
    archive formats are not proven to self-describe variable names/order (R8).
    """
    input_vars = sanitize_varnames(input_vars)
    response_var = sanitize_varnames(response_var)

    metadata, staged_training_file = stage_model_for_import(sumo_model_id, run_dir)
    if (
        input_vars != metadata.input_descriptors
        or response_var != metadata.output_descriptor
    ):
        raise ValueError(
            f"SuMo model '{sumo_model_id}' was exported with inputs "
            f"{metadata.input_descriptors} / output '{metadata.output_descriptor}', "
            f"but import was requested with inputs {input_vars} / output '{response_var}'"
        )

    dakota_conf = create_sumo_evaluation_conffile(
        build_file=staged_training_file,
        samples_file=PROCESSED_EVALUATION_SAMPLES_FILE,
        input_variables=input_vars,
        output_responses=[response_var],
        sumo_import_name=sumo_model_id,
        export_import_format=metadata.export_format,
    )

    dakobj = DakotaObject()
    dakobj.run(dakota_conf, run_dir)

    results = {
        response_var + "_hat": get_results(
            run_dir / "predictions.dat", response_var
        ).tolist()
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

    results = extract_predictions_gridpoints(
        run_dir, response_var, input_vars, NSAMPLESPERVAR
    )

    if len(grid_vars) == 2:  ## this is not necessary for 3D
        output = np.array(results[response_var])
        reshape_indices = [
            NPOINTSPERDIMENSION[i]
            for i in range(len(input_vars))
            if input_vars[i] in grid_vars
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
    distributions: dict[str, dict[str, float | str]],
    output_responses: list[str],
    moga_kwargs: dict,
) -> dict[str, list[float | int]]:
    _logger.debug("Minimizing responses: %s", ", ".join(output_responses))

    input_vars = sanitize_varnames(input_vars)
    output_responses = [sanitize_varnames(resp) for resp in output_responses]
    distributions = sanitize_varnames(distributions)

    # assumes uniform distribution for MOGA - raises Error otherwise
    lower_bounds, upper_bounds = get_bounds_uniform_distributions(
        input_vars, distributions
    )

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


def _saltelli_abc(
    ppfs_list: list[Any], n: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saltelli A/B/C sample blocks (n, d) from per-variable frozen scipy
    distributions, drawn from one scrambled Sobol' QMC stream of dim 3*d.
    C is an independent third stream feeding the exact pair designs."""
    from scipy.stats.qmc import Sobol

    d = len(ppfs_list)
    U = Sobol(d=3 * d, seed=seed, scramble=True).random(n)  # (n, 3*d)
    A = np.column_stack([p.ppf(U[:, i]) for i, p in enumerate(ppfs_list)])
    B = np.column_stack([p.ppf(U[:, d + i]) for i, p in enumerate(ppfs_list)])
    C = np.column_stack([p.ppf(U[:, 2 * d + i]) for i, p in enumerate(ppfs_list)])
    return A, B, C


def _saltelli_pair_designs(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AB_i and exact pair U/V design matrices for sample blocks (n, d).

    Returns (AB, uv, pairs): AB (d, n, d), AB[i] = A with column i swapped to
    B's (Saltelli 2010 convention); uv (2K, n, d) with row k = U^ij (B with
    columns [i, j] from A) and row K+k = V^ij (C with columns [i, j] from A);
    pairs (K, 2) in np.triu_indices order, aligned with uv."""
    n, d = A.shape
    AB = np.empty((d, n, d))
    for i in range(d):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        AB[i] = AB_i
    pair_ii, pair_jj = np.triu_indices(d, k=1)
    pairs = np.column_stack([pair_ii, pair_jj])
    K = pairs.shape[0]
    uv = np.empty((2 * K, n, d))
    for k in range(K):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        rest_b = B.copy()
        rest_b[:, [i, j]] = A[:, [i, j]]
        rest_c = C.copy()
        rest_c[:, [i, j]] = A[:, [i, j]]
        uv[k] = rest_b  # U^ij
        uv[K + k] = rest_c  # V^ij
    return AB, uv, pairs


def _sobol_algebra(
    f_A: np.ndarray,
    f_B: np.ndarray,
    f_AB: np.ndarray,
    f_U: np.ndarray,
    f_V: np.ndarray,
    pairs: np.ndarray,
) -> dict:
    """Full Saltelli index algebra on aligned sample vectors.

    Single source for BOTH point estimates and bootstrap replicates (the CIs
    need the identical estimator per resample). ``f_A``/``f_B`` are (n,) baseline
    and alternative evaluations; ``f_AB`` is (d, n), row i = Saltelli AB_i design
    (f(A) with column i swapped to B's). ``f_U``/``f_V`` are (K, n) pair designs
    (K = C(d,2), row k for ``pairs[k] = (i, j)``):
    U^ij_n = f(A_i, A_j, B_rest) and V^ij_n = f(A_i, A_j, C_rest) with ``C`` an
    independent third sample stream.

    Estimators (Saltelli et al., Comput.Phys.Commun. 181 (2010) 259-270, in the
    EXACT form scipy.stats.sobol_indices implements it - _sensitivity_analysis.py
    ``saltelli_2010`` - including its Sobol' & Levitan (1999) pooled mean removal
    and pooled A/B variance. Centering only through ``mu_hat``/``var_hat`` keeps
    point estimates, pair subtraction and bootstrap replicates on ONE estimator
    and makes every index translation-invariant under ``f -> f + c``: the raw
    products below are mean-subtracted before averaging, so physically-offset
    outputs (e.g. stress in Pa, mean >> std) cannot corrupt the ratios.
      first_i  = E[(f_B - mu) . (f_AB_i - f_A)] / Var([f_A, f_B])  Table 2(b)
      total_i  = E[(f_A - f_AB_i)^2] / (2 Var([f_A, f_B]))          Table 2(f)
      S_(i,j)  = E[(f_U^ij - mu) . (f_V^ij - mu)] / Var([f_A, f_B]) joint pair
        EXACT for arbitrary d: conditional on the shared pair values (A_i, A_j),
        U and V differ only in fully independent rest columns, so
        E[f_U . f_V | pair] = g(pair)^2 with g = E[Y | X_i, X_j]; averaging rows
        estimates E[g^2], and E[g] = E[Y] makes E[g^2] - E^2 = Var(g) exactly.
        The ANOVA terms contained in {i, j} are only {i}, {j}, {i, j}, hence
        Var(g)/V = S_i + S_j + S_ij with NO truncation assumption at all.
      S_ij     = S_(i,j) - S_i - S_j                             EXACT any d.
        (Contrast the retired B26nc identity (mmux_vite SPEC §B), which inferred
        O(d^2) pair values from d first/total gaps - underdetermined for d>=4,
        with pair sums provably collapsing to 0 at d=4 and negative beyond.)

    Order masses: M1 = sum_i S_i; M2 = sum_{i<j} S_ij (each unordered pair
    once); R = 1 - M1 - M2 closure residual. Raw finite-sample estimates are
    never clamped.

    Cost note: the exact pair estimator needs 2x C(d,2) mixed designs, so total
    surrogate cost is n*(2 + d + d(d-1)) ~ n*d^2 - the O(d^2) pair-specific
    mixed evaluation cost that is unavoidable for exact arbitrary-d pairs
    (verified against the analytic additive d=8, pair-interaction d=5, Ishigami
    and pair-quadratic d=10 benchmarks in tests/test_sobol_indices.py within
    bootstrap-CI-scaled tolerances).
    """
    n = f_A.shape[0]
    d = f_AB.shape[0]
    # Pooled mean/variance of A and B - exactly what scipy.stats.sobol_indices
    # uses (Sobol' & Levitan 1999 mean removal; var of the independent A/B pool).
    # All products are mean-subtracted before averaging, so every index is
    # invariant under f -> f + c (a raw E[f.f'] - E[f_A]^2 form drifts with the
    # output offset whenever mean >> std, e.g. physical units in Pa).
    pooled = np.concatenate((f_A, f_B))
    mu_hat = float(np.mean(pooled))
    var_hat = float(np.var(pooled))
    if var_hat == 0.0:
        return {
            "first": np.zeros(d),
            "total": np.zeros(d),
            "second": np.zeros((d, d)),
            "m1": 0.0,
            "m2": 0.0,
            "r": 0.0,
            "var_zero": True,
        }
    # scipy saltelli_2010 Table 2(b): mean((f_B - mu) * (f_AB_i - f_A)) / var
    # (the difference f_AB_i - f_A is centering-free).
    first = (
        np.mean((f_B - mu_hat)[None, :] * (f_AB - f_A[None, :]), axis=1) / var_hat
    )  # (d,)
    total = 0.5 * np.mean((f_A[None, :] - f_AB) ** 2, axis=1) / var_hat  # (d,)
    second = np.zeros((d, d))
    if f_U.shape[0] > 0:
        # Centered cross-product of the pair designs: E[g^2] - E[g]^2 = Var(g),
        # estimated translation-invariantly (mu_hat absorbs any output offset).
        joint = (
            np.einsum("kn,kn->k", f_U - mu_hat, f_V - mu_hat) / n
        ) / var_hat  # (K,) Var(E[Y|Xi,Xj])/V, exact
        ii = pairs[:, 0]
        jj = pairs[:, 1]
        s_ij = joint - first[ii] - first[jj]
        second[ii, jj] = s_ij
        second[jj, ii] = s_ij
    m1 = float(np.sum(first))
    m2 = float(np.sum(np.triu(second, k=1)))
    r = 1.0 - m1 - m2
    return {
        "first": first,
        "total": total,
        "second": second,
        "m1": m1,
        "m2": m2,
        "r": r,
        "var_zero": False,
    }


def _sobol_joint_bootstrap(
    f_A: np.ndarray,
    f_B: np.ndarray,
    f_AB: np.ndarray,
    f_U: np.ndarray,
    f_V: np.ndarray,
    pairs: np.ndarray,
    *,
    seed: int | None,
    n_resamples: int,
    confidence: float,
) -> dict:
    """Shared-row bootstrap over ALL indices.

    ONE row-index resample per replicate recomputes S_i, S_Ti, S_ij and then
    M1, M2, R, so the percentile CIs preserve estimator covariance and the
    M1+M2+R=1 identity holds per replicate. Resampling reuses the already
    computed evaluations - no extra surrogate calls.

    Returns: first/total CIs (d, 2), per-pair CIs (K, 2) (used to scale analytic
    regression tolerances), and m1/m2/r CIs as (2,) bounds.
    """
    n = f_A.shape[0]
    d = f_AB.shape[0]
    k_pairs = f_U.shape[0]
    rng = np.random.default_rng(seed)
    boot_first = np.empty((n_resamples, d))
    boot_total = np.empty((n_resamples, d))
    boot_second = np.empty((n_resamples, k_pairs))
    boot_m1 = np.empty(n_resamples)
    boot_m2 = np.empty(n_resamples)
    boot_r = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        alg = _sobol_algebra(
            f_A[idx], f_B[idx], f_AB[:, idx], f_U[:, idx], f_V[:, idx], pairs
        )
        boot_first[b] = alg["first"]
        boot_total[b] = alg["total"]
        if k_pairs:
            boot_second[b] = alg["second"][pairs[:, 0], pairs[:, 1]]
        boot_m1[b] = alg["m1"]
        boot_m2[b] = alg["m2"]
        boot_r[b] = alg["r"]
    alpha = (1 - confidence) / 2
    lo, hi = 100 * alpha, 100 * (1 - alpha)

    def _bounds(x: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.percentile(x, lo, axis=0), np.percentile(x, hi, axis=0)], axis=-1
        )

    return {
        "first": _bounds(boot_first),  # (d, 2)
        "total": _bounds(boot_total),  # (d, 2)
        "second": _bounds(boot_second),  # (K, 2) in triu pair order
        "m1": np.array([np.percentile(boot_m1, lo), np.percentile(boot_m1, hi)]),
        "m2": np.array([np.percentile(boot_m2, lo), np.percentile(boot_m2, hi)]),
        "r": np.array([np.percentile(boot_r, lo), np.percentile(boot_r, hi)]),
    }


def evaluate_sobol_indices(
    run_dir: Path,
    PROCESSED_TRAINING_FILE: Path,
    input_vars: list[str],
    response_var: str,
    distributions: dict[str, dict],
    preprocessor,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute Sobol' first-order, total-order, and second-order sensitivity indices.

    Generates Saltelli A/B/C sample matrices locally (honouring per-input
    distributions via ``scipy.stats.rv_continuous.ppf``), evaluates all samples
    in ONE batch through ``evaluate_sumo()`` (surrogate-only, Dakota does not run
    ``variance_based_decomp`` itself), then applies ``_sobol_algebra`` -- the
    exact scipy.stats.sobol_indices saltelli_2010 algebra (pinned equal by the
    estimator-parity test) -- for first/total order plus the exact joint-pair
    second-order (pairwise interaction) estimator and order masses.

    Second-order estimator: exact joint-pair designs,
    S_ij = Var(E[Y|X_i,X_j])/V - S_i - S_j, exact for ANY d -- derivation,
    B26nc history and analytic benchmarks in ``_sobol_algebra``. Surrogate cost
    grows to n*(2 + d + d(d-1)) ~ n*d^2, the documented price of exact pairs
    (Saltelli 2010, Comput.Phys.Commun. 181(2), 259-270).

    Base sample count is the fixed ``SOBOL_BASE_SAMPLES`` constant (V36), NOT
    the frontend's shared UQ ``numSamples`` -- Sobol' has fundamentally
    different sample-cost scaling (multiplicative in ``d_varying``) than the
    other UQ views, so it uses its own well-established practical default.
    All confidence intervals (first/total/second plus the M1/M2/R order masses
    returned as ``sobolOrderContributions``) come from ONE shared bootstrap: each
    replicate resamples the existing evaluation rows once (no extra surrogate
    calls) and recomputes every index, preserving estimator covariance and the
    M1+M2+R=1 partition per replicate.

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
        "total_ci_high": float}}``), ``"sobolSecondOrder"``
        (``{varA: {varB: float}}`` symmetric over unordered pairs, no self-pair),
        and ``"sobolOrderContributions"`` (unique order masses ``first_order``=M1,
        ``second_order``=M2, ``third_and_higher``=R with bootstrap CIs and
        ``heuristic_noise_floor``; ``None`` when the sample output variance is
        zero -- the fractions are undefined there).
    """
    import math

    import pandas as pd
    from scipy.stats import lognorm, norm

    # NOTE: input_vars/distributions must stay in the caller's original
    # (unsanitized) form here -- preprocessor.input_variables is keyed by
    # original names and preprocessor.transform() looks samples up by those
    # same original column names (see propagate_manual_uq_with_uncertainty
    # above for the same reasoning). response_var is already the mapped
    # Dakota-safe name by the time it reaches this function.
    # df_varying[input_vars] needs list, not tuple, indexing
    input_vars = list(input_vars)

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

    # Build frozen scipy distributions with .ppf for each varying variable. The
    # scale map is the shared scale_distribution: a log-scale uniform is drawn
    # log-uniform in the caller's original units (V44ls) and a log-scale normal
    # becomes lognorm(s=σ, scale=e^μ) (V46rn — exp of an ln-space N(μ,σ)), the
    # surrogate's preprocessor re-applying the log downstream. Either way the
    # decomposition is taken over what the model actually sees.
    ppfs = {}
    for var in varying_vars:
        dist_info = distributions[var]
        dist_type = dist_info["distribution"]
        log_scale = resolve_log_scale(var, dist_info)
        if dist_type == "normal":
            ppfs[var] = (
                lognorm(
                    s=float(dist_info["std"]), scale=np.exp(float(dist_info["mean"]))
                )
                if log_scale
                else norm(loc=dist_info["mean"], scale=dist_info["std"])
            )
        elif dist_type == "uniform":
            ppfs[var] = scale_distribution(
                float(dist_info["min"]),
                float(dist_info["max"]),
                scale="log" if log_scale else "linear",
            )
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    # --- 2. Fixed base sample count, rounded up to next power of 2 (V36) ---
    if d_varying == 0:
        # All variables are constant — indices are trivially zero. Order masses
        # are NOT reported as (0,0,0): a zero-variance output has no variance to
        # partition, so the M1/M2/R fractions are undefined and the response
        # states that explicitly with null (the closure-to-1 statement is about
        # real variance partitions, not this degenerate case).
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
        return {
            "sobol": sobol,
            "sobolSecondOrder": {},
            "sobolOrderContributions": None,
        }

    n = 2 ** math.ceil(math.log2(max(SOBOL_BASE_SAMPLES, 2)))

    # --- 3. Saltelli A/B/C sampling + AB_i + exact pair (U/V) designs ---
    # Shared builders, also driven by the analytic benchmarks in
    # tests/test_sobol_indices.py, so tests exercise the shipped pipeline.
    A, B, C = _saltelli_abc([ppfs[var] for var in varying_vars], n, seed)
    AB, uv, pairs = _saltelli_pair_designs(A, B, C)
    K = pairs.shape[0]

    # --- 4. Concatenate into one big sample matrix, restore constant columns ---
    # Layout: A (n) + B (n) + AB_0..AB_{d-1} (d*n) + U^0..U^{K-1} (K*n) + V^0..V^{K-1} (K*n)
    all_samples_varying = np.vstack(
        [A, B, AB.reshape(-1, d_varying), uv.reshape(-1, d_varying)]
    )

    # Build DataFrame with varying variables only
    df_varying = pd.DataFrame(all_samples_varying, columns=pd.Index(varying_vars))

    # Add constant columns (fixed values for all rows)
    for var, val in constant_vars.items():
        df_varying[var] = val

    # Reorder columns to match original input_vars order
    df_samples = df_varying[input_vars]

    # --- 5. Transform and write processed samples, call evaluate_sumo ONCE ---
    SAMPLES_FILE = run_dir / "sobol_samples.csv"
    df_samples.to_csv(SAMPLES_FILE, index=False)

    df_samples_transformed = preprocessor.transform(df_samples)
    PROCESSED_SAMPLES_FILE = run_dir / "sobol_samples_processed.csv"
    df_samples_transformed.to_csv(PROCESSED_SAMPLES_FILE, sep=" ", index=False)

    mapped_input_vars = [
        preprocessor.input_variables[var].mapped_name for var in input_vars
    ]
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

    # --- 6. Split the single batch of predictions back into the design blocks ---
    all_preds = np.asarray(results[prediction_key])
    total_rows = n * (d_varying + 2 + 2 * K)
    if len(all_preds) != total_rows:
        raise ValueError(
            f"Expected {total_rows} predictions (n={n}, d_varying={d_varying}, pairs={K}), "
            f"got {len(all_preds)}."
        )

    idx = 0
    fA_flat = all_preds[idx : idx + n]
    idx += n
    fB_flat = all_preds[idx : idx + n]
    idx += n
    fAB_2d = np.empty((d_varying, n))
    for i in range(d_varying):
        fAB_2d[i] = all_preds[idx : idx + n]
        idx += n
    f_UV = all_preds[idx : idx + 2 * K * n].reshape(2 * K, n)
    f_U = f_UV[:K]
    f_V = f_UV[K:]

    # --- 7. Point estimates + shared-row bootstrap CIs ---
    # All CIs come from ONE bootstrap: one row-index resample per replicate
    # recomputes first/total/second AND the M1/M2/R masses, resampling the
    # already-computed evaluations -- no extra evaluate_sumo() calls, effectively
    # free. Single estimator for every d: the algebra IS the scipy
    # saltelli_2010 estimator (pinned by the estimator-parity test), so there is
    # no scipy call and no d==1 special case -- at d=1, AB_0 is the full B
    # sample and the Saltelli form yields S_1 ~ 1.
    alg = _sobol_algebra(fA_flat, fB_flat, fAB_2d, f_U, f_V, pairs)
    boot = _sobol_joint_bootstrap(
        fA_flat,
        fB_flat,
        fAB_2d,
        f_U,
        f_V,
        pairs,
        seed=seed,
        n_resamples=SOBOL_BOOTSTRAP_RESAMPLES,
        confidence=SOBOL_BOOTSTRAP_CONFIDENCE,
    )
    first_order = np.atleast_1d(alg["first"])  # shape (d_varying,)
    total_order = np.atleast_1d(alg["total"])  # shape (d_varying,)
    first_order_ci = boot["first"]  # (d_varying, 2) percentile bounds
    total_order_ci = boot["total"]  # (d_varying, 2)

    # --- 8. Second-order S_ij for every unordered pair ---
    # Exact joint-pair estimator S_ij = Var(E[Y|X_i,X_j])/V - S_i - S_j from the
    # U/V mixed designs (algebra + derivation in _sobol_algebra), replacing the
    # retired B26nc identity that inferred O(d^2) pairs from d first/total gaps
    # and provably collapsed (pair sums = 0 at d=4, negative beyond).
    sobol_second_order: dict[str, dict[str, float]] = {}
    if d_varying >= 2:
        for ii in range(d_varying):
            for jj in range(ii + 1, d_varying):
                s_ij = float(alg["second"][ii, jj])
                var_a = varying_vars[ii]
                var_b = varying_vars[jj]
                sobol_second_order.setdefault(var_a, {})[var_b] = s_ij
                sobol_second_order.setdefault(var_b, {})[var_a] = s_ij

    # --- 9. Order masses M1/M2/R + heuristic noise floor ---
    # M1 sums the DISPLAYED first-order point estimates; M2 sums each unordered
    # pair once; R = 1 - M1 - M2 closes the partition by construction (no
    # clamping; R's bootstrap CI covering 0 means "unresolved from sampling
    # noise", and the noise floor below is an explicitly rough comparator).
    # Zero sample variance (degenerate surrogate on these samples): variance
    # fractions are undefined -> report null, NOT silent (0,0,0) masses that
    # would contradict the closure-to-1 (same contract as the d_varying==0 path).
    order_contributions: dict[str, float] | None
    if alg["var_zero"]:
        order_contributions = None
    else:
        m1_mass = float(np.sum(first_order))
        m2_mass = float(np.sum(np.triu(alg["second"], k=1)))
        r_mass = 1.0 - m1_mass - m2_mass
        ci_half_widths = np.concatenate(
            [
                (first_order_ci[:, 1] - first_order_ci[:, 0]) / 2.0,
                (total_order_ci[:, 1] - total_order_ci[:, 0]) / 2.0,
            ]
        )
        order_contributions = {
            "first_order": m1_mass,
            "second_order": m2_mass,
            "third_and_higher": r_mass,
            "first_order_ci_low": float(boot["m1"][0]),
            "first_order_ci_high": float(boot["m1"][1]),
            "second_order_ci_low": float(boot["m2"][0]),
            "second_order_ci_high": float(boot["m2"][1]),
            "third_and_higher_ci_low": float(boot["r"][0]),
            "third_and_higher_ci_high": float(boot["r"][1]),
            "heuristic_noise_floor": float(np.median(ci_half_widths)),
        }

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
    for var_a, inner in sobol_second_order.items():
        for var_b, val in inner.items():
            if not np.isfinite(val):
                raise ValueError(
                    f"Second-order Sobol' index {var_a}:{var_b} is not finite: {val}"
                )
    if order_contributions is not None:
        for key, val in order_contributions.items():
            if not np.isfinite(val):
                raise ValueError(
                    f"Sobol' order contribution {key} is not finite: {val}"
                )

    return {
        "sobol": sobol,
        "sobolSecondOrder": sobol_second_order,
        "sobolOrderContributions": order_contributions,
    }


if __name__ == "__main__":
    _logger.info("Dakota evaluation module executed")
