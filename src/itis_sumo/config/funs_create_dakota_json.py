### JSON study-input seam for Dakota (T16mo rung 2 / SPEC §R4).
###
### This module mirrors config/funs_create_dakota_conf.py but emits Dakota's native
### JSON study input (dakota.json schema) instead of NIDR strings. The produced dict is
### fed to Dakota via dakenv.study(callbacks=..., input_json=...) -- available on
### itis-dakota >= 6.24 (the wheel we pinned in rung 2).
###
### Validation oracle: itis-dakota/dakota/src/dakota.json (the same schema Dakota's own
### materializer uses). We validate emitted dicts with `jsonschema` before handing them to
### Dakota; Dakota validates again at runtime.
import json
from pathlib import Path


def start_dakota_json_file(
    top_method_pointer: str | None = None,
    results_file_name: str | None = None,
) -> dict:
    """Base DakotaStudy dict: environment + the §R5 interface-cache shim.

    T16mo rung 2 / SPEC §R5: itis-dakota 6.24's study() ctor unconditionally calls
    Interface::interface_cache(problem_db) and throws map::at for pure data-fit surrogate
    confs (no interface constructed). The shim appends an otherwise-unused `single` model
    backed by a no-op `fork` interface so the cache gets an entry. Mirrors
    add_r5_interface_cache_workaround() in funs_create_dakota_conf.py.
    """
    if results_file_name is None:
        results_file_name = "results.dat"
    environment: dict = {"tabular_data": {"tabular_data_file": results_file_name}}
    if top_method_pointer is not None:
        environment["top_method_pointer"] = top_method_pointer
    return {
        "environment": environment,
        "model": [
            {
                "single": {
                    "id_model": "R5_WA_MODEL",
                    "interface_pointer": "R5_WA_INTERFACE",
                    "variables_pointer": "VARIABLES",
                    "responses_pointer": "RESPONSES",
                }
            }
        ],
        "interface": [
            {
                "id_interface": "R5_WA_INTERFACE",
                "analysis_drivers": {
                    "drivers": ["true"],
                    "interface_type": {"fork": {}},
                },
            }
        ],
    }


def add_surrogate_model(
    id_model: str = "SURR_MODEL",
    surrogate_type: str = "gaussian_process surfpack",
    training_samples_file: str | None = None,
    *,
    has_eval_id_column: bool,
) -> dict:
    """Surrogate model block (JSON). Mirrors add_surrogate_model()."""
    # itis-sumo only builds gaussian_process surfpack surrogates today.
    if surrogate_type != "gaussian_process surfpack":
        raise ValueError(f"JSON seam only supports 'gaussian_process surfpack', got {surrogate_type!r}")
    if training_samples_file is None:
        raise ValueError("training samples file is required for the JSON seam (sampling-built surrogates not yet supported)")
    global_cfg: dict = {
        "type": {
            "gaussian_process": {
                "gp_implementation": {"surfpack": {}},
                "export_approx_variance_file": {"filename": "variances.dat"},
            }
        },
        "import_build_points_file": {
            "filename": str(training_samples_file),
            "format": {
                "custom_annotated": {
                    "header": {"use_variable_labels": True},
                    "eval_id": bool(has_eval_id_column),
                }
            },
        },
        "export_approx_points_file": {"filename": "predictions.dat"},
    }
    return {
        "surrogate": {
            "id_model": id_model,
            "category": {"global_approx": global_cfg},
        },
        "interface_pointer": "R5_WA_INTERFACE",
    }


def add_evaluation_method(
    input_file: str,
    model_pointer: str = "SURR_MODEL",
    includes_eval_id: bool = False,
) -> dict:
    """Evaluation + list_parameter_study method block (JSON). Mirrors add_evaluation_method()."""
    return {
        "list_parameter_study": {
            "id_method": "EVALUATION",
            "model_pointer": model_pointer,
            "source": {
                "import_points_file": {
                    "filename": str(input_file),
                    "format": {
                        "custom_annotated": {
                            "eval_id": {} if includes_eval_id else None,
                        }
                    },
                }
            },
        }
    }


def add_continuous_variables(
    variables: list[str],
    id_variables: str = "VARIABLES",
    initial_points: list[float] | None = None,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
) -> dict:
    cd: dict = {"count": len(variables), "descriptors": list(variables)}
    if initial_points is not None:
        cd["initial_point"] = list(initial_points)
    if lower_bounds is not None:
        cd["lower_bounds"] = list(lower_bounds)
    if upper_bounds is not None:
        cd["upper_bounds"] = list(upper_bounds)
    return {"id_variables": id_variables, "continuous_design": cd}


def add_responses(descriptors: list[str]) -> dict:
    return {
        "id_responses": "RESPONSES",
        "descriptors": list(descriptors),
        "response_type": {"objective_functions": {"count": len(descriptors)}},
        "gradient_type": {"no_gradients": True},
        "hessian_type": {"no_hessians": True},
    }


def create_sumo_evaluation_json(
    build_file: Path,
    samples_file: Path,
    input_variables: list[str],
    output_responses: list[str],
    has_eval_id_column: bool | None = None,
) -> dict:
    """JSON equivalent of create_sumo_evaluation_conffile().

    Builds a surrogate from `build_file` points and evaluates it at `samples_file`.
    Returns a DakotaStudy dict ready for dakenv.study(input_json=...).
    """
    training_samples_file = str(Path(build_file).resolve())
    if has_eval_id_column is None:
        has_eval_id_column = "processed" not in training_samples_file
    study = start_dakota_json_file()
    study["model"].append(
        add_surrogate_model(
            training_samples_file=training_samples_file,
            has_eval_id_column=has_eval_id_column,
        )
    )
    study.setdefault("method", []).append(
        add_evaluation_method(str(Path(samples_file).resolve()))
    )
    study.setdefault("variables", []).append(
        add_continuous_variables(variables=input_variables)
    )
    study.setdefault("responses", []).append(add_responses(output_responses))
    return study


def validate_against_schema(study: dict, schema_path: str | Path) -> None:
    """Validate a DakotaStudy dict against Dakota's own JSON schema (authoritative oracle)."""
    import jsonschema

    schema = json.loads(Path(schema_path).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=study, schema=schema)
