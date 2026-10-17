"""Dakota JSON study-input seam for the surrogate-evaluate workflow.

SPEC §R4 / T46xx: the v1 config path stays the NIDR string composers
(`config.nidr`, invariant V5ui). This module is the
**additive, parallel** builder for the same workflows, emitting Dakota's native
JSON study input (`dakota.environment.study(input_json=...)`, Dakota 6.24+,
`tests/simple/simple.json` in the itis-dakota checkout) instead of NIDR text.

Oracle (DAKOTA-STUBS.md): the pinned `itis-dakota` wheel ships `dakota.spec`,
the typed, authoritative pydantic model tree for Dakota's JSON study format
(`dakota.spec.DakotaStudy`, generated from `dakota/src/dakota.json`). This
module builds a plain dict (mirroring the NIDR composers' assembly style so the
two paths read side by side), then validates it by *constructing* those models
and serializing with `model_dump(mode="json")`: a malformed study fails here,
with the engine's own schema error, instead of mid-run in Dakota.

R5 interface-cache workaround (SPEC §R5, B16mo): same semantics as the NIDR shim
`add_r5_interface_cache_workaround()`, expressed in JSON. That shim appends an
otherwise-unused `single` model (`R5_WA_MODEL`) wired to a no-op `fork` interface
(`analysis_drivers = 'true'`) and points every data-fit surrogate at it via
`truth_model_pointer`, so the study ctor's `Interface::interface_cache` lookup
finds an entry. In NIDR the truth pointer is a model-level keyword; in the JSON
schema it is *not* — it lives at `category.global_approx.build_data.
truth_model_pointer` (see `dakota/src/dakota.input.nspec`, where the
`dace_method_pointer | truth_model_pointer` alternative is the `build_data`
union). The nesting differs; the materialized Dakota DB key is the same
(`model.surrogate.truth_model_pointer`, `ProblemDescDB.cpp`). Verified
end-to-end: the emitted JSON study builds and evaluates bit-identically to the
NIDR conf (`tests/test_dakota_json_seam.py`) and the dummy interface is
constructed but never executed (`R5_WA_INTERFACE: 0 total`).

Expressiveness: **no gap found** for this path. `model`, `interface`,
`analysis_drivers` (fork), the `truth_model_pointer` pointer, the
`gaussian_process surfpack` surrogate (build-points import, model
import/export archives), the `list_parameter_study` evaluation method, variables
and responses are all expressible in the shipped model tree.

Scope today: the surrogate-evaluate path only (build / import / export a
surrogate + evaluate at given points). The sibling NIDR workflows (UQ
propagation, cross-validation, MOGA, adaptive sampling) and the other surrogate
implementations (e.g. the `dakota` GP, a sampling-built surrogate) are not
ported here yet — they land as further parallel incremental builders. The seam
therefore keeps a narrower signature than its NIDR twin and does not accept
those arguments at all, so a caller can never get a study that silently lacks
the behaviour they asked for.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dakota.spec import DakotaStudy

_logger = logging.getLogger(__name__)


def _add_r5_interface_cache_workaround_json() -> dict[str, Any]:
    """JSON twin of `add_r5_interface_cache_workaround()` (SPEC §R5).

    A `single` model backed by a no-op `fork` interface (`analysis_drivers =
    'true'`), declared but never referenced by any method/iterator, so the
    interface is constructed (populating Dakota's `interface_cache`) but never
    executed. Returns the `model`/`interface` fragments to merge into a study.
    """
    return {
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


def start_dakota_json_file(
    top_method_pointer: str | None = None,
    results_file_name: str | None = None,
) -> dict[str, Any]:
    """JSON twin of `start_dakota_file()`: the base study (environment block) plus
    the R5 interface-cache workaround (SPEC §R5). `results_file_name` is written
    as Dakota's tabular data file, matching the NIDR default."""
    if results_file_name is None:
        results_file_name = "results.dat"
    environment: dict[str, Any] = {
        "tabular_data": {"tabular_data_file": results_file_name}
    }
    if top_method_pointer is not None:
        environment["top_method_pointer"] = top_method_pointer
    study: dict[str, Any] = {"environment": environment}
    study.update(_add_r5_interface_cache_workaround_json())
    return study


def _add_surrogate_model_json(
    training_samples_file: str,
    *,
    has_eval_id_column: bool,
    id_model: str = "SURR_MODEL",
    sumo_import_name: str | None = None,
    sumo_export_name: str | None = None,
    export_import_format: str = "text_archive",
) -> dict[str, Any]:
    """JSON twin of `add_surrogate_model()` for the build/import/export global
    Gaussian-process surrogate (`gaussian_process surfpack`, the only
    implementation the evaluate/export workflow builds; other implementations,
    e.g. the `dakota` GP, are a later increment — see module docstring). The R5
    shim pointer is attached through `build_data.truth_model_pointer`.
    """
    if export_import_format not in ("text_archive", "binary_archive"):
        raise ValueError(
            "export_import_format must be 'text_archive' or 'binary_archive', "
            f"got {export_import_format!r}"
        )
    if sumo_import_name and sumo_export_name:
        raise ValueError(
            "a surrogate is either imported or exported, not both "
            f"(got import={sumo_import_name!r}, export={sumo_export_name!r})"
        )

    # gaussian_process surfpack block, mirroring the NIDR composer: the model
    # archive keywords plus the predictive-variance output file.
    impl_config: dict[str, Any] = {}
    if sumo_import_name:
        impl_config["import_model"] = {
            "filename_prefix": sumo_import_name,
            "import_format": {export_import_format: True},
        }
    elif sumo_export_name:
        impl_config["export_model"] = {
            "filename_prefix": sumo_export_name,
            "formats": {export_import_format: True},
        }
    gp_config: dict[str, Any] = {
        "gp_implementation": {"surfpack": impl_config},
        "export_approx_variance_file": {"filename": "variances.dat"},
    }

    custom_annotated: dict[str, Any] = {"header": {"use_variable_labels": True}}
    if has_eval_id_column:
        custom_annotated["eval_id"] = True

    global_approx: dict[str, Any] = {
        "type": {"gaussian_process": gp_config},
        # SPEC §R5: the R5 shim pointer (NIDR: `truth_model_pointer` at model level).
        "build_data": {"truth_model_pointer": "R5_WA_MODEL"},
        "import_build_points_file": {
            "filename": training_samples_file,
            "format": {"custom_annotated": custom_annotated},
        },
        "export_approx_points_file": {"filename": "predictions.dat"},
    }
    return {
        "surrogate": {
            "id_model": id_model,
            "category": {"global_approx": global_approx},
        }
    }


def _add_evaluation_method_json(
    input_file: str,
    model_pointer: str = "SURR_MODEL",
) -> dict[str, Any]:
    """JSON twin of `add_evaluation_method()`: a `list_parameter_study` that
    evaluates the surrogate at the points in `input_file` (inputs-only file)."""
    return {
        "list_parameter_study": {
            "id_method": "EVALUATION",
            "model_pointer": model_pointer,
            "source": {
                "import_points_file": {
                    "filename": input_file,
                    "format": {"custom_annotated": {"header": True}},
                }
            },
        }
    }


def _add_continuous_variables_json(
    variables: list[str],
    id_variables: str = "VARIABLES",
) -> dict[str, Any]:
    """JSON twin of `add_continuous_variables()` for the evaluate path
    (continuous design variables + descriptors)."""
    return {
        "id_variables": id_variables,
        "continuous_design": {"count": len(variables), "descriptors": list(variables)},
    }


def _add_responses_json(descriptors: list[str]) -> dict[str, Any]:
    """JSON twin of `add_responses()`: objective-function responses, no
    gradients/hessians."""
    descriptors = descriptors if isinstance(descriptors, list) else [descriptors]
    return {
        "id_responses": "RESPONSES",
        "descriptors": list(descriptors),
        "response_type": {"objective_functions": {"count": len(descriptors)}},
        "gradient_type": {"no_gradients": True},
        "hessian_type": {"no_hessians": True},
    }


def _infer_has_eval_id_column(training_samples_file: str) -> bool:
    """Reuse the NIDR path's inference (`infer_has_eval_id_column_from_filename`,
    V15zx) rather than restating the `"processed"` filename convention here.

    Imported lazily through the package's public surface: this module is itself
    wired up by `itis_sumo.config/__init__.py`, so a top-level import of the
    package would be a circular import whose success depends on module init
    order.
    """
    from itis_sumo.config import infer_has_eval_id_column_from_filename

    return infer_has_eval_id_column_from_filename(training_samples_file)


def validate_json_study(study: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a Dakota JSON study against the engine's native `dakota.spec`
    model tree (the authoritative oracle, DAKOTA-STUBS.md) and return its
    canonical, JSON-serializable form (`model_dump(mode="json")`, `None` values
    dropped — Dakota's materializer takes presence-only keywords, explicit nulls
    are noise).

    Raises `pydantic.ValidationError` on a malformed study. The input mapping is
    not mutated (the `dakota.spec` before-validators replace nested dicts with
    model instances in place, so it is deep-copied first).
    """
    model = DakotaStudy.model_validate(copy.deepcopy(dict(study)))
    return model.model_dump(mode="json", exclude_none=True)


def create_sumo_evaluation_json(
    build_file: Path,
    samples_file: Path,
    input_variables: list[str],
    output_responses: list[str],
    sumo_import_name: str | None = None,
    sumo_export_name: str | None = None,
    export_import_format: str = "text_archive",
    has_eval_id_column: bool | None = None,
) -> dict[str, Any]:
    """JSON twin of `create_sumo_evaluation_conffile()`: build (or import) a
    surrogate from `build_file` and evaluate it at the points in `samples_file`.

    Returns a validated, JSON-serializable `DakotaStudy` dict ready for
    `dakota.environment.study(callbacks={}, input_json=...)`. `has_eval_id_column`
    falls back to the NIDR filename inference when omitted (V15zx).
    """
    training_samples_file = str(Path(build_file).resolve())
    if has_eval_id_column is None:
        has_eval_id_column = _infer_has_eval_id_column(training_samples_file)

    study = start_dakota_json_file()
    study["model"].append(
        _add_surrogate_model_json(
            training_samples_file,
            has_eval_id_column=has_eval_id_column,
            sumo_import_name=sumo_import_name,
            sumo_export_name=sumo_export_name,
            export_import_format=export_import_format,
        )
    )
    study["method"] = [_add_evaluation_method_json(str(Path(samples_file).resolve()))]
    study["variables"] = [_add_continuous_variables_json(input_variables)]
    study["responses"] = [_add_responses_json(output_responses)]

    _logger.info(
        "Built Dakota JSON study for surrogate evaluation (build=%s, samples=%s)",
        training_samples_file,
        samples_file,
    )
    return validate_json_study(study)
