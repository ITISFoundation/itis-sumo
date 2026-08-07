"""
Integration utilities for DataPreprocessor with existing ML workflow.

This module provides helper functions to integrate the DataPreprocessor class
with the existing _create_training_file_from_jobs function and other workflow components.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import ValidationError

from itis_sumo.preprocess.data_preprocessor import DataPreprocessor
from itis_sumo.preprocess.models import JobVariableSelection, required_completed_jobs

_logger = logging.getLogger(__name__)


class VariableStats(TypedDict):
    count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    median: float | None
    range: float | None
    cv: float | None


def create_training_file_with_preprocessor(
    jobs: list[dict[str, Any]],
    input_vars: list[str],
    output_response: list[str],
    preprocessor: DataPreprocessor,
    run_dir: Path,
) -> tuple[Path, DataPreprocessor]:
    """
    Create a training file from jobs using the DataPreprocessor for transformation.

    This function extends the existing _create_training_file_from_jobs functionality
    by applying data preprocessing transformations.

    Args:
        jobs: List of job dictionaries with 'inputs', 'outputs', and 'status' keys
        input_vars: List of input variable names
        output_response: Output variable name(s)
        preprocessor: Fitted DataPreprocessor instance
        folder_name: Name of the folder to create for output files

    Returns:
        Tuple of (training_file_path, fitted_preprocessor)
    """
    output_response_list = (
        [output_response] if isinstance(output_response, str) else output_response
    )

    try:
        validated_selection = JobVariableSelection.model_validate(
            {
                "jobs": jobs,
                "input_vars": input_vars,
                "output_vars": output_response_list,
                "minimum_completed_jobs": required_completed_jobs(input_vars),
            }
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    df_jobs = pd.DataFrame(validated_selection.to_records())
    _logger.info(f"Created DataFrame with shape: {df_jobs.shape}")

    # Apply preprocessing
    df_transformed = preprocessor.fit_transform(df_jobs)
    _logger.info(f"Transformed DataFrame with shape: {df_transformed.shape}")
    _logger.info(f"Transformed columns: {list(df_transformed.columns)}")

    # Save original data
    original_file = run_dir / "df_jobs_original.csv"
    df_jobs.to_csv(original_file, index=False)

    # Save transformed data
    training_file = run_dir / "df_jobs_transformed.csv"
    df_transformed.to_csv(training_file, index=False)

    # Save preprocessor configuration
    config_file = run_dir / "preprocessor_config.json"
    preprocessor.save_config(config_file)

    _logger.info(f"Training files created in: {run_dir}")
    _logger.info(f"Original data: {original_file}")
    _logger.info(f"Transformed data: {training_file}")
    _logger.info(f"Preprocessor config: {config_file}")

    return training_file, preprocessor


def setup_preprocessor_from_config(
    input_vars: list[str],
    output_response: list[str],
    input_normalizations: dict[str, str] | None = None,
    output_normalizations: dict[str, str] | None = None,
    input_sign_switches: list[str] | None = None,
    output_sign_switches: list[str] | None = None,
    # Variable filtering parameters
    include_inputs: list[str] | None = None,
    exclude_inputs: list[str] | None = None,
    include_outputs: list[str] | None = None,
    exclude_outputs: list[str] | None = None,
    input_patterns: list[str] | None = None,
    output_patterns: list[str] | None = None,
    input_predicate: Callable[[str, Any], bool] | None = None,
    output_predicate: Callable[[str, Any], bool] | None = None,
) -> DataPreprocessor:
    """
    Create and configure a DataPreprocessor instance using the new separated workflow.

    Args:
        input_vars: List of input variable names
        output_response: Output variable name(s)
        input_normalizations: Dict mapping input var names to normalization methods
        output_normalizations: Dict mapping output var names to normalization methods
        input_sign_switches: List of input vars to switch signs
        output_sign_switches: List of output vars to switch signs
        include_inputs: List of input variable names to include (exact matches)
        exclude_inputs: List of input variable names to exclude (exact matches)
        include_outputs: List of output variable names to include (exact matches)
        exclude_outputs: List of output variable names to exclude (exact matches)
        input_patterns: List of regex patterns for input variable names to include
        output_patterns: List of regex patterns for output variable names to include
        input_predicate: Custom function to filter input variables (name, config) -> bool
        output_predicate: Custom function to filter output variables (name, config) -> bool

    Returns:
        Configured DataPreprocessor instance
    """
    preprocessor = DataPreprocessor()

    # Step 1: Set up basic variable mappings
    preprocessor.setup_variables(input_vars=input_vars, output_vars=output_response)

    # Step 2: Configure normalization (optional)
    if input_normalizations or output_normalizations:
        preprocessor.setup_normalization(
            input_normalizations=input_normalizations,
            output_normalizations=output_normalizations,
        )

    # Step 3: Configure sign switching (optional)
    if input_sign_switches or output_sign_switches:
        preprocessor.setup_sign_switching(
            input_sign_switches=input_sign_switches,
            output_sign_switches=output_sign_switches,
        )

    # Step 4: Apply variable filtering (optional)
    if any(
        [
            include_inputs,
            exclude_inputs,
            include_outputs,
            exclude_outputs,
            input_patterns,
            output_patterns,
            input_predicate,
            output_predicate,
        ]
    ):
        preprocessor.filter_variables(
            include_inputs=include_inputs,
            exclude_inputs=exclude_inputs,
            include_outputs=include_outputs,
            exclude_outputs=exclude_outputs,
            input_patterns=input_patterns,
            output_patterns=output_patterns,
            input_predicate=input_predicate,
            output_predicate=output_predicate,
        )

    return preprocessor


def load_and_inverse_transform_results(
    results: dict[str, list[float]] | pd.DataFrame | list[dict[str, float]],
    config_file_path: str | Path,
) -> (
    dict[str, float]
    | list[dict[str, float]]
    | dict[str, list[float]]
    | list[dict[str, list[float]]]
):
    """
    Load a preprocessor configuration and inverse transform algorithm results.

    Args:
        results: Results from ML algorithms (with mapped variable names)
        config_file_path: Path to the preprocessor configuration file

    Returns:
        Results with original variable names and scales
    """
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_config(config_file_path)

    # Handle different input formats
    if isinstance(results, dict):
        return preprocessor.inverse_transform(results)
    elif isinstance(results, pd.DataFrame):
        inverse_results = []
        for _, row in results.iterrows():
            inverse_results.append(preprocessor.inverse_transform(row.to_dict()))
        return inverse_results
    elif isinstance(results, list):
        return [preprocessor.inverse_transform(result) for result in results]
    else:
        raise TypeError(f"Unsupported results format: {type(results)}")


def get_preprocessing_summary(config_file_path: str | Path) -> dict[str, Any]:
    """
    Get a summary of preprocessing configuration from a saved config file.

    Args:
        config_file_path: Path to the preprocessor configuration file

    Returns:
        Summary dictionary
    """
    preprocessor = DataPreprocessor()
    preprocessor.load_config(config_file_path)
    return preprocessor.get_summary()


def create_filtered_preprocessor(
    base_preprocessor: DataPreprocessor,
    include_inputs: list[str] | None = None,
    exclude_inputs: list[str] | None = None,
    include_outputs: list[str] | None = None,
    exclude_outputs: list[str] | None = None,
    input_patterns: list[str] | None = None,
    output_patterns: list[str] | None = None,
    input_predicate: Callable[[str, Any], bool] | None = None,
    output_predicate: Callable[[str, Any], bool] | None = None,
) -> DataPreprocessor:
    """
    Create a new preprocessor with filtered variables from an existing one.
    This preserves the original preprocessor while creating a filtered copy.

    Args:
        base_preprocessor: The original preprocessor to filter from
        include_inputs: List of input variable names to include (exact matches)
        exclude_inputs: List of input variable names to exclude (exact matches)
        include_outputs: List of output variable names to include (exact matches)
        exclude_outputs: List of output variable names to exclude (exact matches)
        input_patterns: List of regex patterns for input variable names to include
        output_patterns: List of regex patterns for output variable names to include
        input_predicate: Custom function to filter input variables (name, config) -> bool
        output_predicate: Custom function to filter output variables (name, config) -> bool

    Returns:
        New DataPreprocessor instance with filtered variables
    """
    import copy

    # Create a deep copy of the base preprocessor
    filtered_preprocessor = DataPreprocessor()
    filtered_preprocessor.input_variables = copy.deepcopy(base_preprocessor.input_variables)
    filtered_preprocessor.output_variables = copy.deepcopy(base_preprocessor.output_variables)
    filtered_preprocessor._is_fitted = base_preprocessor._is_fitted

    # Apply filtering
    filtered_preprocessor.filter_variables(
        include_inputs=include_inputs,
        exclude_inputs=exclude_inputs,
        include_outputs=include_outputs,
        exclude_outputs=exclude_outputs,
        input_patterns=input_patterns,
        output_patterns=output_patterns,
        input_predicate=input_predicate,
        output_predicate=output_predicate,
    )

    return filtered_preprocessor


def get_variable_statistics(
    jobs: list[dict[str, Any]], variable_names: list[str], variable_type: str = "input"
) -> dict[str, VariableStats]:
    """
    Get basic statistics for variables from job data.
    Useful for deciding which variables to filter based on their characteristics.

    Args:
        jobs: List of job dictionaries
        variable_names: List of variable names to analyze
        variable_type: Type of variables ("input" or "output")

    Returns:
        Dictionary mapping variable names to their statistics
    """
    import numpy as np

    completed_jobs = [
        job for job in jobs if job.get("status", "").lower() in ["completed", "success"]
    ]

    if len(completed_jobs) == 0:
        raise ValueError("No completed jobs found for statistics calculation")

    stats: dict[str, VariableStats] = {}

    for var_name in variable_names:
        values = []
        data_key = "inputs" if variable_type == "input" else "outputs"

        for job in completed_jobs:
            if data_key in job and var_name in job[data_key]:
                try:
                    values.append(float(job[data_key][var_name]))
                except (ValueError, TypeError):
                    continue

        var_stats: VariableStats
        if values:
            values = np.array(values)
            var_stats = VariableStats(
                count=len(values),
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                median=float(np.median(values)),
                range=float(np.max(values) - np.min(values)),
                cv=float(np.std(values) / np.mean(values))
                if np.mean(values) != 0
                else float("inf"),
            )
        else:
            var_stats = VariableStats(
                count=0,
                mean=None,
                std=None,
                min=None,
                max=None,
                median=None,
                range=None,
                cv=None,
            )
        stats[var_name] = var_stats

    return stats


def filter_variables_by_statistics(
    jobs: list[dict[str, Any]],
    input_vars: list[str],
    output_vars: list[str],
    min_cv: float | None = None,
    max_cv: float | None = None,
    min_range: float | None = None,
    max_range: float | None = None,
    require_complete_data: bool = True,
) -> dict[str, list[str]]:
    """
    Filter variables based on their statistical properties.

    Args:
        jobs: List of job dictionaries
        input_vars: List of input variable names to consider
        output_vars: List of output variable names to consider
        min_cv: Minimum coefficient of variation (std/mean) to include
        max_cv: Maximum coefficient of variation to include
        min_range: Minimum range (max-min) to include
        max_range: Maximum range to include
        require_complete_data: If True, only include variables with data in all jobs

    Returns:
        Dictionary with 'inputs' and 'outputs' keys containing filtered variable lists
    """
    input_stats = get_variable_statistics(jobs, input_vars, "input")
    output_stats = get_variable_statistics(jobs, output_vars, "output")

    def passes_filter(stats: VariableStats) -> bool:
        if stats["count"] == 0:
            return False

        if require_complete_data and stats["count"] < len(
            [job for job in jobs if job.get("status", "").lower() in ["completed", "success"]]
        ):
            return False

        if min_cv is not None and (stats["cv"] is None or stats["cv"] < min_cv):
            return False

        if max_cv is not None and (stats["cv"] is None or stats["cv"] > max_cv):
            return False

        if min_range is not None and (stats["range"] is None or stats["range"] < min_range):
            return False

        return max_range is None or (
            stats["range"] is not None and stats["range"] <= max_range
        )

    filtered_inputs = [var for var, stats in input_stats.items() if passes_filter(stats)]
    filtered_outputs = [var for var, stats in output_stats.items() if passes_filter(stats)]

    _logger.info(
        f"Statistical filtering: {len(filtered_inputs)}/{len(input_vars)} inputs, {len(filtered_outputs)}/{len(output_vars)} outputs"
    )

    return {"inputs": filtered_inputs, "outputs": filtered_outputs}
