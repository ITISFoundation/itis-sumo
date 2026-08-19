"""Tests for mmux_flaskapi.dakota.funs_data_processing pure helper functions."""

import re

import numpy as np
import pandas as pd
import pytest

from itis_sumo.data.funs_data_processing import (
    _filter_data,
    _parse_data,
    create_manual_uq_samples,
    get_bounds_uniform_distribution,
    get_bounds_uniform_distributions,
    get_non_dominated_indices,
    get_results,
    get_variable_names,
    is_dominated,
    load_data,
    sanitize_varnames,
)

# --- sanitize_varnames -------------------------------------------------------


def test_sanitize_varnames_string():
    assert sanitize_varnames("my var name") == "my_var_name"
    assert sanitize_varnames("weird!chars@here") == "weird_chars_here"


def test_sanitize_varnames_list():
    assert sanitize_varnames(["a b", "c-d", "e f g"]) == ["a_b", "c-d", "e_f_g"]


def test_sanitize_varnames_preserves_hyphen():
    """Regression test: the sanitize regex character class `[^0-9a-zA-Z_*-+/]`
    used to place "-" between "*" and "+", forming an unintended char-range
    (matching "*" and "+" only) instead of a literal hyphen, silently replacing
    real hyphens with underscores (e.g. breaks the default `-AFpeak` output key
    used by `get_results`)."""
    assert sanitize_varnames("-AFpeak") == "-AFpeak"
    assert sanitize_varnames("a-b-c") == "a-b-c"


def test_sanitize_varnames_dict():
    result = sanitize_varnames({"a b": 1, "c d": 2})
    assert result == {"a_b": 1, "c_d": 2}


def test_sanitize_varnames_nested_dict():
    result = sanitize_varnames({"outer key": {"inner key": 1}})
    assert result == {"outer_key": {"inner_key": 1}}


def test_sanitize_varnames_dataframe():
    df = pd.DataFrame({"col a": [1], "col b": [2]})
    result = sanitize_varnames(df)
    assert list(result.columns) == ["col_a", "col_b"]
    # original untouched
    assert list(df.columns) == ["col a", "col b"]


def test_sanitize_varnames_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported input type"):
        sanitize_varnames(12345)


# --- get_bounds_uniform_distribution(s) --------------------------------------


def test_get_bounds_uniform_distribution_valid():
    dist = {"distribution": "uniform", "min": 0.0, "max": 10.0}
    assert get_bounds_uniform_distribution("x", dist) == (0.0, 10.0)


def test_get_bounds_uniform_distribution_non_uniform_raises():
    dist = {"distribution": "normal", "mean": 0.0, "std": 1.0}
    with pytest.raises(ValueError, match="Non-uniform distribution"):
        get_bounds_uniform_distribution("x", dist)


def test_get_bounds_uniform_distribution_missing_bounds_raises():
    dist = {"distribution": "uniform", "min": 0.0}
    with pytest.raises(ValueError, match="Bounds for variable"):
        get_bounds_uniform_distribution("x", dist)


def test_get_bounds_uniform_distribution_invalid_bounds_raises():
    dist = {"distribution": "uniform", "min": 10.0, "max": 0.0}
    with pytest.raises(ValueError, match="Invalid bounds"):
        get_bounds_uniform_distribution("x", dist)


def test_get_bounds_uniform_distributions_multiple_vars():
    distributions = {
        "x": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "y": {"distribution": "uniform", "min": -5.0, "max": 5.0},
    }
    lower, upper = get_bounds_uniform_distributions(["x", "y"], distributions)
    assert lower == [0.0, -5.0]
    assert upper == [1.0, 5.0]


def test_get_bounds_uniform_distributions_missing_var_raises():
    with pytest.raises(ValueError, match="is not defined"):
        get_bounds_uniform_distributions(["missing"], {})


# --- is_dominated / get_non_dominated_indices --------------------------------


def test_is_dominated_true_when_another_point_dominates():
    point = np.array([1.0, 1.0])
    others = np.array([[2.0, 2.0], [0.0, 0.0]])
    assert is_dominated(point, others) is True


def test_is_dominated_false_when_no_point_dominates():
    point = np.array([0.0, 0.0])
    others = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert is_dominated(point, others) is False


def test_get_non_dominated_indices_minimization():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    # point 1 (a=2,b=2) is dominated by neither in a min/min sense trade-off (Pareto front)
    indices = get_non_dominated_indices(df, ["a", "b"])
    assert set(indices) == {0, 1, 2}


def test_get_non_dominated_indices_with_optimization_modes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    # minimize a, maximize b -> point 2 (a=3,b=3) is best on b but worst on a;
    # point 0 (a=1,b=1) is best on a but worst on b -> both on the Pareto front,
    # point 1 (a=2,b=2) is dominated by neither too since a min/b max create a trade-off
    indices = get_non_dominated_indices(
        df, ["a", "b"], optimization_modes=["min", "max"]
    )
    assert set(indices) == {0, 1, 2}


def test_get_non_dominated_indices_invalid_mode_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not recognized"):
        get_non_dominated_indices(df, ["a"], optimization_modes=["bogus"])


def test_get_non_dominated_indices_mismatched_modes_length_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="must match"):
        get_non_dominated_indices(df, ["a", "b"], optimization_modes=["min"])


# --- create_manual_uq_samples -------------------------------------------------


def test_create_manual_uq_samples_normal():
    samples = create_manual_uq_samples(
        ["x"],
        {"x": {"distribution": "normal", "mean": 0.0, "std": 1.0}},
        num_samples=5,
        seed=1,
    )
    assert len(samples["x"]) == 5


def test_create_manual_uq_samples_uniform():
    samples = create_manual_uq_samples(
        ["x"],
        {"x": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
        num_samples=5,
        seed=1,
    )
    assert len(samples["x"]) == 5
    assert all(0.0 <= v <= 1.0 for v in samples["x"])


def test_create_manual_uq_samples_constant():
    samples = create_manual_uq_samples(
        ["x"], {"x": {"distribution": "constant", "value": 42}}, num_samples=3, seed=1
    )
    assert samples["x"] == [42.0, 42.0, 42.0]


def test_create_manual_uq_samples_unsupported_distribution_raises():
    with pytest.raises(ValueError, match="Unsupported distribution type"):
        create_manual_uq_samples(
            ["x"], {"x": {"distribution": "lognormal"}}, num_samples=3, seed=1
        )


class TestCreateManualUqSamplesSeedReproducibility:
    """B12/V27: `seed` must actually control reproducibility of generated samples."""

    def test_same_seed_produces_identical_samples_normal(self):
        distributions = {"x1": {"distribution": "normal", "mean": 0.0, "std": 1.0}}
        samples_a = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=123
        )
        samples_b = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=123
        )
        assert samples_a["x1"] == samples_b["x1"]

    def test_same_seed_produces_identical_samples_uniform(self):
        distributions = {"x1": {"distribution": "uniform", "min": -1.0, "max": 1.0}}
        samples_a = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=7
        )
        samples_b = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=7
        )
        assert samples_a["x1"] == samples_b["x1"]

    def test_different_seeds_produce_different_samples(self):
        distributions = {"x1": {"distribution": "normal", "mean": 0.0, "std": 1.0}}
        samples_a = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=1
        )
        samples_b = create_manual_uq_samples(
            ["x1"], distributions, num_samples=50, seed=2
        )
        assert samples_a["x1"] != samples_b["x1"]

    def test_same_seed_produces_identical_samples_mixed_distributions(self):
        """T17 (PR #487 review): a single seeded call spanning both distribution
        types must reproduce identically, not just each type in isolation."""
        distributions = {
            "x1": {"distribution": "normal", "mean": 0.0, "std": 1.0},
            "x2": {"distribution": "uniform", "min": -1.0, "max": 1.0},
        }
        samples_a = create_manual_uq_samples(
            ["x1", "x2"], distributions, num_samples=50, seed=42
        )
        samples_b = create_manual_uq_samples(
            ["x1", "x2"], distributions, num_samples=50, seed=42
        )
        assert samples_a["x1"] == samples_b["x1"]
        assert samples_a["x2"] == samples_b["x2"]


class TestLoadDataMalformedFile:
    """B20/V38: a `.dat`/`.txt` file whose header column count doesn't match a data
    row's column count must raise a diagnostic ValueError naming the file, line
    number, and both column counts - not pandas' bare "X columns passed, passed
    data had Y columns" (which carries no file/line context)."""

    def test_row_with_fewer_columns_than_header_raises_with_context(self, tmp_path):
        malformed_file = tmp_path / "predictions.dat"
        malformed_file.write_text(
            "eval_id interface x1 x2 x3\n1 NO_ID 0.1 0.2 0.3\n2 NO_ID 0.4 0.5\n"
        )
        with pytest.raises(ValueError, match=r"header \(line 1\) has 5 columns"):
            load_data(malformed_file)

    def test_error_message_identifies_offending_line_number(self, tmp_path):
        malformed_file = tmp_path / "predictions.dat"
        malformed_file.write_text(
            "eval_id interface x1 x2 x3\n1 NO_ID 0.1 0.2 0.3\n2 NO_ID 0.4 0.5\n"
        )
        with pytest.raises(ValueError, match="line 3 has 4 columns"):
            load_data(malformed_file)

    def test_error_message_includes_raw_context_lines(self, tmp_path):
        """B21: the raw (untokenized) previous/offending/next lines must be included
        so a future occurrence can confirm/refute a Dakota tabular line-wrapping
        cause without needing filesystem access to the (possibly already-cleaned-up)
        run directory."""
        malformed_file = tmp_path / "predictions.dat"
        malformed_file.write_text(
            "eval_id interface x1 x2 x3\n1 NO_ID 0.1 0.2 0.3\n2 NO_ID 0.4 0.5\n3 NO_ID 0.7 0.8 0.9\n"
        )
        with pytest.raises(ValueError, match=r"Raw context \(total 4 lines\)"):
            load_data(malformed_file)
        with pytest.raises(
            ValueError, match=re.escape("line 3 (offending): '2 NO_ID 0.4 0.5'")
        ):
            load_data(malformed_file)

    def test_well_formed_file_still_loads_correctly(self, tmp_path):
        good_file = tmp_path / "predictions.dat"
        good_file.write_text(
            "eval_id interface x1 x2 x3\n1 NO_ID 0.1 0.2 0.3\n2 NO_ID 0.4 0.5 0.6\n"
        )
        df = load_data(good_file)
        assert len(df) == 2
        assert list(df.columns) == ["eval_id", "interface", "x1", "x2", "x3"]


class TestLoadDataHealOrDropMalformedRow:
    """B22 (2026-07-15): confirmed real Dakota tabular-writer defect - some rows
    duplicate the interface/leading-variable prefix before finishing the row, and
    never repeat `_eval_id`. `on_malformed_row="heal_or_drop"` recovers what it can
    (inferring `_eval_id` as the row's own sequential position) and drops+warns on
    anything it can't reconstruct, instead of raising and failing the whole caller."""

    HEADER = "%eval_id interface x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 y1\n"

    def test_heals_real_captured_corrupted_row(self, tmp_path):
        # Byte-for-byte the row captured from a real production predictions.dat
        # (B22): the interface+x1..x9 prefix written twice (last x9 digit noisy),
        # then x10/x11/y1 written once, `_eval_id` never repeated.
        corrupted_line = (
            "1        APPROX_INTERFACE_1 0.564657       0.0685889      0.163073       "
            "0.347383       0.127338       0.000826683    0.565321       0.210969       "
            "0.554521        APPROX_INTERFACE_1 0.564657       0.0685889      0.163073       "
            "0.347383       0.127338       0.000826683    0.565321       0.210969       "
            "0.554525       0.174912       488.687        0.3218000105   \n"
        )
        f = tmp_path / "predictions.dat"
        f.write_text(self.HEADER + corrupted_line)
        warnings: list[str] = []
        df = load_data(f, on_malformed_row="heal_or_drop", warnings=warnings)
        assert len(df) == 1
        assert df.iloc[0]["_eval_id"] == "1"
        assert df.iloc[0]["interface"] == "APPROX_INTERFACE_1"
        assert float(df.iloc[0]["x9"]) == pytest.approx(
            0.554525
        )  # 2nd (final) write wins
        assert float(df.iloc[0]["x10"]) == pytest.approx(0.174912)
        assert float(df.iloc[0]["x11"]) == pytest.approx(488.687)
        assert float(df.iloc[0]["y1"]) == pytest.approx(0.3218000105)
        assert any("Recovered corrupted row" in w for w in warnings)

    def test_drops_unhealable_row_and_warns_instead_of_raising(self, tmp_path):
        f = tmp_path / "predictions.dat"
        f.write_text(
            self.HEADER + "1 APPROX_INTERFACE_1 0.1 0.2 0.3\n"
        )  # too short to heal
        warnings: list[str] = []
        df = load_data(f, on_malformed_row="heal_or_drop", warnings=warnings)
        assert len(df) == 0
        assert any("Dropped unrecoverable malformed row" in w for w in warnings)

    def test_default_still_raises_for_the_same_unhealable_row(self, tmp_path):
        f = tmp_path / "predictions.dat"
        f.write_text(self.HEADER + "1 APPROX_INTERFACE_1 0.1 0.2 0.3\n")
        with pytest.raises(ValueError, match="Malformed data file"):
            load_data(f)  # on_malformed_row defaults to "raise"

    def test_well_formed_rows_unaffected_by_heal_or_drop_mode(self, tmp_path):
        f = tmp_path / "predictions.dat"
        f.write_text(
            self.HEADER
            + "1 APPROX_INTERFACE_1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2\n"
        )
        df = load_data(f, on_malformed_row="heal_or_drop")
        assert len(df) == 1


# --- _parse_data / load_data / get_results / get_variable_names -------------


def test_parse_data_space_delimited(tmp_path):
    file = tmp_path / "data.dat"
    file.write_text("col1 col2\n1 2\n3 4\n")
    parsed = _parse_data(file)
    assert parsed == [["col1", "col2"], ["1", "2"], ["3", "4"]]


def test_load_data_dat_file(tmp_path):
    file = tmp_path / "data.dat"
    file.write_text("x y\n1 2\n3 4\n")
    df = load_data(file)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2


def test_load_data_csv_file(tmp_path):
    file = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(file, index=False)
    df = load_data(file)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2


def test_load_data_unsupported_extension_raises(tmp_path):
    file = tmp_path / "data.xyz"
    file.write_text("nonsense")
    with pytest.raises(ValueError, match="not a DAT / TXT / JSON / CSV file"):
        load_data(file)


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(AssertionError, match="does not exist"):
        load_data(tmp_path / "missing.csv")


def test_get_results_extracts_column_as_float_array(tmp_path):
    file = tmp_path / "predictions.dat"
    file.write_text("-AFpeak other\n1.5 9\n2.5 9\n")
    result = get_results(file)
    np.testing.assert_array_equal(result, np.array([1.5, 2.5]))


def test_get_variable_names_dat(tmp_path):
    file = tmp_path / "data.dat"
    # NOTE: .dat parsing splits on whitespace, so header columns are space-delimited
    file.write_text("col_a col_b\n1 2\n")
    assert get_variable_names(file) == ["col_a", "col_b"]


def test_get_variable_names_csv(tmp_path):
    file = tmp_path / "data.csv"
    pd.DataFrame({"col a": [1], "col b": [2]}).to_csv(file, index=False)
    assert get_variable_names(file) == ["col_a", "col_b"]


def test_get_variable_names_unsupported_extension_raises(tmp_path):
    file = tmp_path / "data.xyz"
    file.write_text("nonsense")
    with pytest.raises(ValueError, match="not a DAT / TXT / JSON / CSV file"):
        get_variable_names(file)


def test_get_variable_names_missing_file_raises(tmp_path):
    with pytest.raises(AssertionError, match="does not exist"):
        get_variable_names(tmp_path / "missing.dat")


# --- _filter_data -------------------------------------------------------------


def test_filter_data_keep_idxs():
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    result = _filter_data(df, keep_idxs=[0, 2])
    assert list(result.index) == [0, 2]


def test_filter_data_filter_n_samples():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    result = _filter_data(df, filter_N_samples=2)
    assert len(result) == 2


def test_filter_data_filter_highest_n():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = _filter_data(df, filter_highest_N=2)
    # top 2 values (5, 4) are dropped by `.iloc[filter_highest_N:]` after sorting descending
    assert len(result) == 3
    assert 5 not in result["a"].values
    assert 4 not in result["a"].values


def test_filter_data_both_filters_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(AssertionError, match="only one of"):
        _filter_data(df, filter_highest_N=1, filter_N_samples=1)
