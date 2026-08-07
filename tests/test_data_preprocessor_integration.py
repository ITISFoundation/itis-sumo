import pandas as pd
import pytest

from itis_sumo.preprocess import (
    DataPreprocessor,
    create_filtered_preprocessor,
    create_training_file_with_preprocessor,
    filter_variables_by_statistics,
    get_preprocessing_summary,
    get_variable_statistics,
    load_and_inverse_transform_results,
    setup_preprocessor_from_config,
)


def _sample_jobs():
    return [
        {
            "status": "completed",
            "inputs": {"x": 1.0, "z": 10.0},
            "outputs": {"y": 2.0, "other": 4.0},
        },
        {
            "status": "success",
            "inputs": {"x": 2.0, "z": 11.0},
            "outputs": {"y": 4.0, "other": 5.0},
        },
        {
            "status": "completed",
            "inputs": {"x": 3.0, "z": 12.0},
            "outputs": {"y": 6.0, "other": 6.0},
        },
        {
            "status": "completed",
            "inputs": {"x": 4.0, "z": 13.0},
            "outputs": {"y": 8.0, "other": 7.0},
        },
        {
            "status": "completed",
            "inputs": {"x": 5.0, "z": 14.0},
            "outputs": {"y": 10.0, "other": 8.0},
        },
    ]


class TestDataPreprocessorIntegration:
    def test_create_training_file_with_preprocessor_writes_expected_files(self, tmp_path):
        preprocessor = DataPreprocessor()
        preprocessor.setup_variables(["x"], ["y"])
        training_file, fitted_preprocessor = create_training_file_with_preprocessor(
            _sample_jobs(),
            input_vars=["x"],
            output_response=["y"],
            preprocessor=preprocessor,
            run_dir=tmp_path / "run",
        )

        assert training_file.exists()
        assert (tmp_path / "run" / "df_jobs_original.csv").exists()
        assert (tmp_path / "run" / "preprocessor_config.json").exists()
        assert fitted_preprocessor.get_variable_mapping() == {"x": "x1", "y": "y1"}

    def test_create_training_file_with_preprocessor_requires_minimum_samples(self, tmp_path):
        preprocessor = DataPreprocessor()
        preprocessor.setup_variables(["x"], ["y"])

        with pytest.raises(ValueError, match="At least 5 samples"):
            create_training_file_with_preprocessor(
                _sample_jobs()[:4],
                input_vars=["x"],
                output_response=["y"],
                preprocessor=preprocessor,
                run_dir=tmp_path / "run",
            )

    def test_setup_preprocessor_from_config_applies_filters(self):
        preprocessor = setup_preprocessor_from_config(
            input_vars=["x", "z"],
            output_response=["y", "other"],
            input_normalizations={"x": "z_score"},
            output_sign_switches=["other"],
            include_inputs=["x"],
            include_outputs=["other"],
        )

        assert list(preprocessor.input_variables.keys()) == ["x"]
        assert list(preprocessor.output_variables.keys()) == ["other"]
        assert preprocessor.input_variables["x"].normalize is True
        assert preprocessor.output_variables["other"].switch_sign is True

    def test_load_and_inverse_transform_results_supports_multiple_shapes(self, tmp_path):
        config_path = tmp_path / "config.json"
        preprocessor = DataPreprocessor()
        preprocessor.setup_variables(["x"], ["y"])
        preprocessor.fit(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
        preprocessor.save_config(config_path)

        assert load_and_inverse_transform_results({"x1": [1.0], "y1": [3.0]}, config_path) == {
            "x": [1.0],
            "y": [3.0],
        }
        assert load_and_inverse_transform_results(
            [{"x1": [1.0], "y1": [3.0]}],
            config_path,
        ) == [{"x": [1.0], "y": [3.0]}]

    def test_get_preprocessing_summary_and_filtered_copy(self, tmp_path):
        config_path = tmp_path / "config.json"
        base = DataPreprocessor()
        base.setup_variables(["x", "z"], ["y"])
        base.fit(pd.DataFrame({"x": [1.0], "z": [2.0], "y": [3.0]}))
        base.save_config(config_path)

        summary = get_preprocessing_summary(config_path)
        filtered = create_filtered_preprocessor(base, include_inputs=["z"])

        assert summary["n_input_variables"] == 2
        assert filtered.get_variable_mapping() == {"z": "x1", "y": "y1"}
        assert base.get_variable_mapping() == {"x": "x1", "z": "x2", "y": "y1"}

    def test_variable_statistics_and_filtering(self):
        jobs = _sample_jobs()

        stats = get_variable_statistics(jobs, ["x"], "input")
        filtered = filter_variables_by_statistics(
            jobs,
            input_vars=["x", "z"],
            output_vars=["y", "other"],
            min_range=3.0,
        )

        assert stats["x"]["count"] == 5
        assert stats["x"]["range"] == 4.0
        assert filtered["inputs"] == ["x", "z"]
        assert filtered["outputs"] == ["y", "other"]
