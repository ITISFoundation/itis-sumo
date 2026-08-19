"""
Data Preprocessor Class for ML Workflows

This module provides a comprehensive data preprocessing class that handles:
- Variable mapping to standardized names (x1, x2, ... for inputs; y1, y2, ... for outputs)
- Data normalization and denormalization
- Sign switching and restoration
- Configuration persistence for reproducibility

The class is designed to work with the existing _create_training_file_from_jobs function
and maintain compatibility with the ML workflow pipeline.
"""

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


@dataclass
class VariableConfig:
    """Configuration for a single variable transformation."""

    original_name: str
    mapped_name: str
    normalize: bool = False
    switch_sign: bool = False
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    normalization_method: str = "z_score"


@dataclass
class PreprocessorConfig:
    """Complete configuration for the data preprocessor."""

    input_variables: dict[str, VariableConfig]
    output_variables: dict[str, VariableConfig]
    created_timestamp: str
    version: str = "1.0"


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for ML workflows.

    This class handles variable mapping, normalization, and sign switching
    while maintaining the ability to reverse all transformations.
    """

    def __init__(self):
        self.input_variables: dict[str, VariableConfig] = {}
        self.output_variables: dict[str, VariableConfig] = {}
        self._is_fitted = False

    def setup_variables(self, input_vars: list[str], output_vars: list[str]) -> None:
        """
        Set up the basic variable mappings to avoid issues with variable names.

        Args:
            input_vars: List of input variable names
            output_vars: List of output variable names
        """
        self.input_variables = self._setup_variable_group(input_vars, "x")
        self.output_variables = self._setup_variable_group(output_vars, "y")

        _logger.info(
            f"Set up {len(self.input_variables)} input variables and {len(self.output_variables)} output variables"
        )

    def setup_normalization(
        self,
        input_normalizations: dict[str, str] | None = None,
        output_normalizations: dict[str, str] | None = None,
    ) -> None:
        """
        Configure normalization for variables.

        Args:
            input_normalizations: Dict mapping input var names to normalization methods
            output_normalizations: Dict mapping output var names to normalization methods
        """
        if input_normalizations:
            self._configure_normalizations(
                self.input_variables, input_normalizations, "input"
            )
            _logger.info(
                f"Configured normalization for {len(input_normalizations)} input variables"
            )

        if output_normalizations:
            self._configure_normalizations(
                self.output_variables, output_normalizations, "output"
            )
            _logger.info(
                f"Configured normalization for {len(output_normalizations)} output variables"
            )

    def setup_sign_switching(
        self,
        input_sign_switches: list[str] | None = None,
        output_sign_switches: list[str] | None = None,
    ) -> None:
        """
        Configure sign switching for variables.

        Args:
            input_sign_switches: List of input vars to switch signs
            output_sign_switches: List of output vars to switch signs
        """
        input_sign_switches = input_sign_switches or []
        output_sign_switches = output_sign_switches or []

        self._configure_sign_switches(
            self.input_variables, input_sign_switches, "input"
        )
        self._configure_sign_switches(
            self.output_variables, output_sign_switches, "output"
        )

        _logger.info(
            f"Configured sign switching for {len(input_sign_switches)} input and {len(output_sign_switches)} output variables"
        )

    def _setup_variable_group(
        self, var_names: list[str], prefix: str
    ) -> dict[str, VariableConfig]:
        """
        Helper function to set up a group of variables (inputs or outputs).
        """
        variables = {}
        for i, var_name in enumerate(var_names):
            mapped_name = f"{prefix}{i + 1}"
            variables[var_name] = VariableConfig(
                original_name=var_name,
                mapped_name=mapped_name,
                normalize=False,
                switch_sign=False,
                normalization_method="none",
            )
        return variables

    def _configure_normalizations(
        self,
        variables: dict[str, VariableConfig],
        normalizations: dict[str, str],
        var_type: str,
    ) -> None:
        for var_name, norm_method in normalizations.items():
            if var_name in variables:
                variables[var_name].normalize = True
                variables[var_name].normalization_method = norm_method
            else:
                _logger.warning(
                    f"{var_type.capitalize()} variable {var_name} not found in setup variables"
                )

    def _configure_sign_switches(
        self,
        variables: dict[str, VariableConfig],
        sign_switches: list[str],
        var_type: str,
    ) -> None:
        for var_name in sign_switches:
            if var_name in variables:
                variables[var_name].switch_sign = True
            else:
                _logger.warning(
                    f"{var_type.capitalize()} variable {var_name} not found in setup variables"
                )

    def _fit_variable_group(
        self,
        data: pd.DataFrame,
        variables: dict[str, VariableConfig],
        var_type: str,
    ) -> None:
        for var_name, config in variables.items():
            if var_name not in data.columns:
                _logger.warning(
                    f"{var_type.capitalize()} variable {var_name} not found in data"
                )
                continue

            values = np.array(data[var_name].values, dtype=float)
            if config.switch_sign:
                values = -values

            if config.normalize and config.normalization_method != "none":
                if config.normalization_method == "z_score":
                    config.mean = float(np.mean(values))
                    config.std = float(np.std(values))
                elif config.normalization_method == "min_max":
                    config.min_val = float(np.min(values))
                    config.max_val = float(np.max(values))

    def _transform_variable_group(
        self,
        data: pd.DataFrame,
        variables: dict[str, VariableConfig],
        transformed_data: dict[str, np.ndarray],
        var_type: str,
    ) -> dict[str, np.ndarray]:
        for var_name, config in variables.items():
            if var_name not in data.columns:
                _logger.warning(
                    f"{var_type.capitalize()} variable {var_name} not found in data during transform"
                )
                continue

            values = np.array(data[var_name].values, dtype=float).copy()

            if config.switch_sign:
                values = -values

            if config.normalize and config.normalization_method != "none":
                values = self._normalize_values(values, config)

            transformed_data[config.mapped_name] = values

        return transformed_data

    def fit(self, data: pd.DataFrame | list[dict[str, Any]]) -> "DataPreprocessor":
        """
        Fit the preprocessor to the data to compute normalization parameters.
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)

        self._fit_variable_group(data, self.input_variables, "input")
        self._fit_variable_group(data, self.output_variables, "output")

        self._is_fitted = True
        _logger.info("Preprocessor fitted to data")
        return self

    def transform(self, data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """
        Transform the data using the fitted preprocessor.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")

        if isinstance(data, list):
            data = pd.DataFrame(data)

        transformed_data: dict[str, np.ndarray] = {}
        transformed_data = self._transform_variable_group(
            data, self.input_variables, transformed_data, "input"
        )
        transformed_data = self._transform_variable_group(
            data, self.output_variables, transformed_data, "output"
        )

        return pd.DataFrame(transformed_data)

    @staticmethod
    def _ndarray_to_variable_dict(
        arr: np.ndarray, all_vars: list[VariableConfig]
    ) -> dict[str, list[float]]:
        data_dict: dict[str, list[float]] = {}
        if arr.ndim == 1:
            for i, var in enumerate(all_vars):
                if i < len(arr):
                    data_dict[var.mapped_name] = [float(arr[i])]
        else:
            for i, var in enumerate(all_vars):
                if i < arr.shape[1]:
                    data_dict[var.mapped_name] = arr[:, i].tolist()
        return data_dict

    def inverse_transform(
        self, data: pd.DataFrame | Mapping[str, list[float] | float] | np.ndarray
    ) -> dict[str, list[float]]:
        """
        Inverse transform the data back to original scale and variable names.
        """
        if not self._is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before inverse transforming data"
            )

        parsed_data: dict[str, list[float]]
        if isinstance(data, np.ndarray):
            all_vars = list(self.input_variables.values()) + list(
                self.output_variables.values()
            )
            parsed_data = self._ndarray_to_variable_dict(data, all_vars)
        elif isinstance(data, pd.DataFrame):
            data_dict: dict[str, list[float]] = {}
            for col in data.columns:
                col_data = data[col].values
                if len(col_data) == 1 and not isinstance(
                    col_data[0], (list, np.ndarray)
                ):
                    data_dict[col] = [float(col_data[0])]
                elif isinstance(col_data[0], (list, np.ndarray)):
                    data_dict[col] = [
                        float(val)
                        for sublist in col_data
                        for val in (
                            sublist
                            if isinstance(sublist, (list, np.ndarray))
                            else [sublist]
                        )
                    ]
                else:
                    data_dict[col] = col_data.tolist()
            parsed_data = data_dict
        else:
            data_dict = {}
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    data_dict[key] = [float(v) for v in np.asarray(value)]
                else:
                    data_dict[key] = [float(value)]
            parsed_data = data_dict

        result = {}

        for var_name, config in self.input_variables.items():
            if config.mapped_name in parsed_data:
                value = parsed_data[config.mapped_name]

                if config.normalize and config.normalization_method != "none":
                    value = self._denormalize_value(value, config)

                if config.switch_sign:
                    if isinstance(value, list):
                        value = [-v for v in value]
                    else:
                        value = -value

                if not isinstance(value, list):
                    value = [value]

                result[var_name] = value

        for var_name, config in self.output_variables.items():
            if config.mapped_name in parsed_data:
                value = parsed_data[config.mapped_name]

                if config.normalize and config.normalization_method != "none":
                    value = self._denormalize_value(value, config)

                if config.switch_sign:
                    if isinstance(value, list):
                        value = [-v for v in value]
                    else:
                        value = -value

                if not isinstance(value, list):
                    value = [value]

                result[var_name] = value

        return result

    def _normalize_values(
        self, values: np.ndarray, config: VariableConfig
    ) -> np.ndarray:
        if config.normalization_method == "z_score":
            if config.std is None or config.mean is None or config.std == 0:
                _logger.warning(
                    f"Invalid parameters mean = {config.mean} and std = {config.std} for z_score normalization of {config.original_name}, skipping normalization"
                )
                return values
            normalized_values = (values - config.mean) / config.std
            return normalized_values
        if config.normalization_method == "min_max":
            if (
                config.max_val is None
                or config.min_val is None
                or config.max_val == config.min_val
            ):
                _logger.warning(
                    f"Invalid parameters min = {config.min_val} and max = {config.max_val} for min_max normalization of {config.original_name}, skipping normalization"
                )
                return values
            normalized_values = (values - config.min_val) / (
                config.max_val - config.min_val
            )
            return normalized_values
        return values

    def _denormalize_value(
        self, value: float | list[float] | np.ndarray, config: VariableConfig
    ) -> float | list[float]:
        if isinstance(value, list):
            values_array = np.array(value, dtype=float)
        elif isinstance(value, np.ndarray):
            values_array = np.asarray(value, dtype=float)
        else:
            if config.normalization_method == "z_score":
                if config.std is None or config.mean is None:
                    _logger.warning(
                        f"Invalid parameters for z_score denormalization of {config.original_name}"
                    )
                    return value
                return value * config.std + config.mean
            if config.normalization_method == "min_max":
                if config.max_val is None or config.min_val is None:
                    _logger.warning(
                        f"Invalid parameters for min_max denormalization of {config.original_name}"
                    )
                    return value
                return value * (config.max_val - config.min_val) + config.min_val
            return value

        if config.normalization_method == "z_score":
            if config.std is None or config.mean is None:
                _logger.warning(
                    f"Invalid parameters for z_score denormalization of {config.original_name}"
                )
                return values_array.tolist()
            denormalized = values_array * config.std + config.mean
        elif config.normalization_method == "min_max":
            if config.max_val is None or config.min_val is None:
                _logger.warning(
                    f"Invalid parameters for min_max denormalization of {config.original_name}"
                )
                return values_array.tolist()
            denormalized = (
                values_array * (config.max_val - config.min_val) + config.min_val
            )
        else:
            denormalized = values_array

        return denormalized.tolist()

    def get_variable_mapping(self) -> dict[str, str]:
        mapping = {}
        for var_name, config in self.input_variables.items():
            mapping[var_name] = config.mapped_name
        for var_name, config in self.output_variables.items():
            mapping[var_name] = config.mapped_name
        return mapping

    def get_inverse_mapping(self) -> dict[str, str]:
        mapping = {}
        for var_name, config in self.input_variables.items():
            mapping[config.mapped_name] = var_name
        for var_name, config in self.output_variables.items():
            mapping[config.mapped_name] = var_name
        return mapping

    def save_config(self, file_path: str | Path) -> None:
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before saving config")

        from datetime import datetime, timezone

        config = PreprocessorConfig(
            input_variables={
                name: config for name, config in self.input_variables.items()
            },
            output_variables={
                name: config for name, config in self.output_variables.items()
            },
            created_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        config_dict = asdict(config)

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        config_dict = convert_numpy_types(config_dict)

        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        _logger.info(f"Configuration saved to {file_path}")

    def load_config(self, file_path: str | Path) -> "DataPreprocessor":
        with open(file_path) as f:
            config_dict = json.load(f)

        self.input_variables = {}
        for name, var_config in config_dict["input_variables"].items():
            self.input_variables[name] = VariableConfig(**var_config)

        self.output_variables = {}
        for name, var_config in config_dict["output_variables"].items():
            self.output_variables[name] = VariableConfig(**var_config)

        self._is_fitted = True
        _logger.info(f"Configuration loaded from {file_path}")
        return self

    def fit_transform(self, data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def filter_variables(
        self,
        include_inputs: list[str] | None = None,
        exclude_inputs: list[str] | None = None,
        include_outputs: list[str] | None = None,
        exclude_outputs: list[str] | None = None,
        input_patterns: list[str] | None = None,
        output_patterns: list[str] | None = None,
        input_predicate: Callable[[str, VariableConfig], bool] | None = None,
        output_predicate: Callable[[str, VariableConfig], bool] | None = None,
    ) -> "DataPreprocessor":
        filtered_inputs = self._filter_variable_group(
            self.input_variables,
            include_list=include_inputs,
            exclude_list=exclude_inputs,
            patterns=input_patterns,
            predicate=input_predicate,
            var_type="input",
        )

        filtered_outputs = self._filter_variable_group(
            self.output_variables,
            include_list=include_outputs,
            exclude_list=exclude_outputs,
            patterns=output_patterns,
            predicate=output_predicate,
            var_type="output",
        )

        self.input_variables = filtered_inputs
        self.output_variables = filtered_outputs
        self._remap_filtered_variables()

        _logger.info(
            f"Filtered variables: {len(self.input_variables)} inputs, {len(self.output_variables)} outputs"
        )

        return self

    def _filter_variable_group(
        self,
        variables: dict[str, VariableConfig],
        include_list: list[str] | None = None,
        exclude_list: list[str] | None = None,
        patterns: list[str] | None = None,
        predicate: Callable[[str, VariableConfig], bool] | None = None,
        var_type: str = "variable",
    ) -> dict[str, VariableConfig]:
        filtered_vars = {}

        for var_name, config in variables.items():
            should_include = True

            if include_list is not None:
                should_include = var_name in include_list
            elif exclude_list is not None:
                should_include = var_name not in exclude_list

            if should_include and patterns is not None:
                pattern_match = any(
                    re.search(pattern, var_name) for pattern in patterns
                )
                should_include = pattern_match

            if should_include and predicate is not None:
                should_include = predicate(var_name, config)

            if should_include:
                filtered_vars[var_name] = config
            else:
                _logger.debug(f"Filtered out {var_type} variable: {var_name}")

        return filtered_vars

    def _remap_filtered_variables(self) -> None:
        for i, (_, config) in enumerate(self.input_variables.items()):
            config.mapped_name = f"x{i + 1}"

        for i, (_, config) in enumerate(self.output_variables.items()):
            config.mapped_name = f"y{i + 1}"

    def filter_by_names(
        self,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        exclude: bool = False,
    ) -> "DataPreprocessor":
        if exclude:
            return self.filter_variables(
                exclude_inputs=input_names, exclude_outputs=output_names
            )
        return self.filter_variables(
            include_inputs=input_names, include_outputs=output_names
        )

    def filter_by_patterns(
        self,
        input_patterns: list[str] | None = None,
        output_patterns: list[str] | None = None,
    ) -> "DataPreprocessor":
        return self.filter_variables(
            input_patterns=input_patterns, output_patterns=output_patterns
        )

    def filter_normalized_only(self) -> "DataPreprocessor":
        return self.filter_variables(
            input_predicate=lambda name, config: config.normalize,
            output_predicate=lambda name, config: config.normalize,
        )

    def filter_non_normalized_only(self) -> "DataPreprocessor":
        return self.filter_variables(
            input_predicate=lambda name, config: not config.normalize,
            output_predicate=lambda name, config: not config.normalize,
        )

    def get_filtered_variable_names(self) -> dict[str, list[str]]:
        return {
            "inputs": list(self.input_variables.keys()),
            "outputs": list(self.output_variables.keys()),
        }

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "fitted": self._is_fitted,
            "n_input_variables": len(self.input_variables),
            "n_output_variables": len(self.output_variables),
            "input_variables": {},
            "output_variables": {},
        }

        for name, config in self.input_variables.items():
            summary["input_variables"][name] = {
                "mapped_name": config.mapped_name,
                "normalize": config.normalize,
                "normalization_method": config.normalization_method,
                "switch_sign": config.switch_sign,
            }

        for name, config in self.output_variables.items():
            summary["output_variables"][name] = {
                "mapped_name": config.mapped_name,
                "normalize": config.normalize,
                "normalization_method": config.normalization_method,
                "switch_sign": config.switch_sign,
            }

        return summary
