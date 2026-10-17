"""Tests for the Dakota JSON input seam (SPEC §R4, T46xx).

Two tiers:

* pure unit (no engine) — the emitter's study validates against the wheel's
  native `dakota.spec.DakotaStudy` oracle, round-trips through
  `model_validate`/`model_dump(mode="json")`, and carries the R5 interface-cache
  shim (SPEC §R5);
* `@pytest.mark.integration` — the JSON-built study is executed on the real
  engine IN A WORKER CHILD PROCESS (Dakota's bindings hold a process-global
  problem DB with no reset hook; `dakota_object` isolates the same way) and
  its predictions are compared to the *NIDR* path's predictions for the
  identical build/evaluation inputs, so the seam is proven equivalent to the
  default path rather than merely schema-valid.
"""

import copy
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest
from dakota.spec import DakotaStudy
from pydantic import ValidationError

from itis_sumo.config import (
    create_sumo_evaluation_json,
    start_dakota_json_file,
    validate_json_study,
)
from itis_sumo.evaluate import evaluate_sumo


@pytest.fixture
def surrogate_files(tmp_path):
    """A tiny training design + evaluation points in the `*processed*` naming the
    NIDR path uses (so `has_eval_id_column` infers to False on both paths)."""
    x_train = np.linspace(0.0, 1.0, 15)
    y_train = 2.0 * x_train + 1.0
    train_file = tmp_path / "train_processed.txt"
    pd.DataFrame({"x": x_train, "y": y_train}).to_csv(train_file, sep=" ", index=False)
    eval_file = tmp_path / "eval.txt"
    pd.DataFrame({"x": [0.05, 0.3, 0.55, 0.72, 0.95]}).to_csv(
        eval_file, sep=" ", index=False
    )
    return train_file, eval_file


class TestJsonSeamEmitter:
    """Pure unit: no engine is invoked, only the `dakota.spec` oracle."""

    def test_emitted_study_validates_and_round_trips(self, surrogate_files):
        train_file, eval_file = surrogate_files
        payload = create_sumo_evaluation_json(
            build_file=train_file,
            samples_file=eval_file,
            input_variables=["x"],
            output_responses=["y"],
        )

        # json-serializable (dakota.environment.study accepts this dict as-is)
        assert json.loads(json.dumps(payload)) == payload

        # constructs native models, round-trips stably
        model = DakotaStudy.model_validate(copy.deepcopy(payload))
        assert model.model_dump(mode="json", exclude_none=True) == payload

        # the emitted payload is exactly what validate_json_study canonicalizes
        assert validate_json_study(payload) == payload

    def test_start_dakota_json_file_carries_the_r5_shim(self):
        study = start_dakota_json_file()
        shim_model = study["model"][0]["single"]
        assert shim_model["id_model"] == "R5_WA_MODEL"
        assert shim_model["interface_pointer"] == "R5_WA_INTERFACE"

        shim_interface = study["interface"][0]
        assert shim_interface["id_interface"] == "R5_WA_INTERFACE"
        drivers = shim_interface["analysis_drivers"]
        assert drivers["drivers"] == ["true"]
        assert drivers["interface_type"] == {"fork": {}}

    def test_surrogate_points_its_truth_model_at_the_shim(self, surrogate_files):
        train_file, eval_file = surrogate_files
        payload = create_sumo_evaluation_json(
            build_file=train_file,
            samples_file=eval_file,
            input_variables=["x"],
            output_responses=["y"],
        )
        global_approx = payload["model"][1]["surrogate"]["category"]["global_approx"]
        # SPEC §R5: JSON nests the NIDR model-level `truth_model_pointer` here.
        assert global_approx["build_data"] == {"truth_model_pointer": "R5_WA_MODEL"}
        assert "truth_model_pointer" not in payload["model"][1]["surrogate"]

    def test_surrogate_build_points_import_is_declared(self, surrogate_files):
        train_file, eval_file = surrogate_files
        payload = create_sumo_evaluation_json(
            build_file=train_file,
            samples_file=eval_file,
            input_variables=["x"],
            output_responses=["y"],
        )
        global_approx = payload["model"][1]["surrogate"]["category"]["global_approx"]
        imported = global_approx["import_build_points_file"]
        assert imported["filename"] == str(train_file.resolve())
        assert imported["format"]["custom_annotated"]["header"] == {
            "use_variable_labels": True
        }
        # `*processed*` training file -> no eval_id column (mirrors NIDR inference)
        assert "eval_id" not in imported["format"]["custom_annotated"]

    def test_explicit_has_eval_id_column_switches_the_format(self, surrogate_files):
        train_file, eval_file = surrogate_files
        payload = create_sumo_evaluation_json(
            build_file=train_file,
            samples_file=eval_file,
            input_variables=["x"],
            output_responses=["y"],
            has_eval_id_column=True,
        )
        imported = payload["model"][1]["surrogate"]["category"]["global_approx"][
            "import_build_points_file"
        ]
        assert imported["format"]["custom_annotated"]["eval_id"] is True
        assert DakotaStudy.model_validate(copy.deepcopy(payload))

    @pytest.mark.parametrize("archive", ["text_archive", "binary_archive"])
    def test_import_and_export_variants_validate(self, surrogate_files, archive):
        train_file, eval_file = surrogate_files

        def emit(**kwargs):
            return create_sumo_evaluation_json(
                build_file=train_file,
                samples_file=eval_file,
                input_variables=["x"],
                output_responses=["y"],
                export_import_format=archive,
                **kwargs,
            )

        exported_gp = emit(sumo_export_name="export")["model"][1]["surrogate"][
            "category"
        ]["global_approx"]["type"]["gaussian_process"]
        assert exported_gp["gp_implementation"]["surfpack"]["export_model"] == {
            "filename_prefix": "export",
            "formats": {archive: True},
        }

        imported_gp = emit(sumo_import_name="export")["model"][1]["surrogate"][
            "category"
        ]["global_approx"]["type"]["gaussian_process"]
        assert imported_gp["gp_implementation"]["surfpack"]["import_model"] == {
            "filename_prefix": "export",
            "import_format": {archive: True},
        }

    def test_import_and_export_are_mutually_exclusive(self, surrogate_files):
        train_file, eval_file = surrogate_files
        with pytest.raises(ValueError, match="not both"):
            create_sumo_evaluation_json(
                build_file=train_file,
                samples_file=eval_file,
                input_variables=["x"],
                output_responses=["y"],
                sumo_import_name="in",
                sumo_export_name="out",
            )

    def test_unknown_archive_format_is_rejected(self, surrogate_files):
        train_file, eval_file = surrogate_files
        with pytest.raises(ValueError, match="export_import_format"):
            create_sumo_evaluation_json(
                build_file=train_file,
                samples_file=eval_file,
                input_variables=["x"],
                output_responses=["y"],
                export_import_format="algebraic_file",  # not a re-loadable archive (R10)
            )

    def test_validate_json_study_rejects_a_malformed_study(self):
        with pytest.raises(ValidationError):
            validate_json_study({"environment": {}, "method": []})


def _read_predictions(path, response):
    """Parse a Dakota tabular output file (`%eval_id interface <vars> <response>`)."""
    df = pd.read_csv(path, sep=r"\s+")
    return df[response].to_numpy(dtype=float)


def _execute_json_study(payload: dict, run_dir: str) -> None:
    """Run the JSON study in a dedicated child process.

    Dakota's Python bindings register definitions in a process-global
    problem DB and the wheel ships no reset hook — the production path runs
    Dakota through worker processes (`dakota_object`) precisely because of
    that. An in-process `study()` call here would leak the R5-shim
    definitions into later tests' engine runs through fork inheritance.
    Explicit `mp_context` per V50gm (fork where available, spawn otherwise;
    the child re-imports everything it needs, so both work).
    """
    import os

    import dakota.environment as dakenv

    os.chdir(run_dir)
    study = dakenv.study(callbacks={}, input_json=payload)
    study.execute()


@pytest.mark.integration
def test_json_evaluation_matches_nidr_prediction(surrogate_files, tmp_path):
    """The JSON seam must reproduce the default NIDR path exactly.

    Identical training design + evaluation points; NIDR via `evaluate_sumo`
    (the default path), JSON via `create_sumo_evaluation_json` +
    `dakota.environment.study(input_json=...)` in a child process (see
    `_execute_json_study`). Both include the R5 shim, so this also proves the
    workaround survives the JSON nesting (SPEC §R5)."""

    train_file, eval_file = surrogate_files

    nidr_run = tmp_path / "nidr_run"
    nidr_run.mkdir()
    nidr = evaluate_sumo(nidr_run, train_file, eval_file, ["x"], "y")

    payload = create_sumo_evaluation_json(
        build_file=train_file,
        samples_file=eval_file,
        input_variables=["x"],
        output_responses=["y"],
    )
    json_run = tmp_path / "json_run"
    json_run.mkdir()
    # explicit start method, never the ambient default (V50gm), mirroring
    # core/dakota_object.py's _ENGINE_MP_CONTEXT selection
    child = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context("spawn")
    ).Process(target=_execute_json_study, args=(payload, str(json_run)))
    child.start()
    child.join(timeout=300)
    # study() construction in the child is the R5 proof: without a populated
    # interface_cache it throws `IndexError: map::at` before execute() (SPEC §R5)
    assert child.exitcode == 0, (
        f"JSON study run failed in child process (exit {child.exitcode})"
    )
    json_pred = _read_predictions(json_run / "predictions.dat", "y")

    nidr_pred = np.asarray(nidr["y_hat"], dtype=float)
    assert json_pred.shape == nidr_pred.shape
    # bit-identical in practice; 1e-12 is the tightest meaningful bar
    np.testing.assert_allclose(json_pred, nidr_pred, atol=1e-12)

    # GP predictive-variance output survived the JSON path too
    assert (json_run / "variances.dat").is_file()
