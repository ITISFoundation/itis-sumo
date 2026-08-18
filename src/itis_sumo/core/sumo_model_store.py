"""Persistence layer for exported SuMo (surrogate model) artifacts (E1).

Single canonical models directory (V13: rejects the mmux/vite draft branch's
per-run `run_dir.parent / "models"` split), keyed by a server-generated
`sumo_model_id` (V12, uuid4 hex) -- never a caller-supplied filename/prefix,
which would allow path traversal.

Each export writes Dakota's archive files alongside a `{id}.metadata.json`
sidecar (V10) recording the verbatim surrogate-model conf block, ordered
input descriptors, output descriptor, and export format, plus a copy of the
real training-data file, kept for reference/debuggability. Re-import (V11)
reuses the ordered descriptors to validate the caller's variables match what
the model was trained on.

Dakota's `import_model` conf block still requires an `import_build_points_file`
keyword to satisfy the surrogate constructor (R2), but the loaded GP is fully
reconstructed from the archive file alone -- the training file's *values* are
never read back into it (verified empirically, R9, and in Dakota's own source:
`SurfpackApproximation::import_model` calls `SurfpackInterface::LoadModel`
directly, bypassing `approxData`). That means the stored training-data copy is
not technically required for a correct reload -- so if it's ever missing (e.g.
a model exported before this persistence was added, or the file was deleted
out-of-band), `stage_model_for_import` falls back to synthesizing a
header-only placeholder from the sidecar's descriptors instead of crashing,
logging a loud warning since that model's original training data is then
unrecoverable.
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

from pydantic import BaseModel

_logger = logging.getLogger(__name__)

MODELS_DIR_ENV_VAR = "ITIS_SUMO_MODELS_DIR"
_DEFAULT_MODELS_DIR = Path.cwd() / "sumo_models"


class SumoModelMetadata(BaseModel):
    """Sidecar contents for one exported SuMo model (V10)."""

    sumo_model_id: str
    surrogate_conf_block: str
    input_descriptors: list[str]
    output_descriptor: str
    export_format: str


def get_models_dir() -> Path:
    """Return the single env-overridable models-dir source (V13)."""
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    models_dir = Path(override) if override else _DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _metadata_path(models_dir: Path, sumo_model_id: str) -> Path:
    return models_dir / f"{sumo_model_id}.metadata.json"


def _training_file_name(sumo_model_id: str) -> str:
    # Must keep the substring "processed" in the name: `add_surrogate_model`
    # (funs_create_dakota_conf.py) infers whether the file carries an eval_id
    # column from this filename convention, not from an explicit flag.
    return f"{sumo_model_id}.processed_training.dat"


def store_exported_model(
    run_dir: Path,
    training_file: Path,
    surrogate_conf_block: str,
    input_descriptors: list[str],
    output_descriptor: str,
    export_prefix: str,
    export_format: str = "text_archive",
) -> str:
    """Move Dakota's `export_prefix.*` archive files out of `run_dir` into the
    canonical models dir under a fresh `sumo_model_id`, alongside a metadata
    sidecar (V10-V13) and a copy of `training_file`, kept for reference (V10).

    Returns the new `sumo_model_id`.
    """
    models_dir = get_models_dir()
    sumo_model_id = uuid.uuid4().hex

    exported_files = sorted(run_dir.glob(f"{export_prefix}.*"))
    if not exported_files:
        raise FileNotFoundError(
            f"No exported model files found in {run_dir} with prefix '{export_prefix}' "
            f"(format={export_format}); Dakota may not have run export_model"
        )
    for f in exported_files:
        suffix = f.name[len(export_prefix) :]  # keeps '.{response}.{ext}'
        shutil.copy(f, models_dir / f"{sumo_model_id}{suffix}")
    shutil.copy(training_file, models_dir / _training_file_name(sumo_model_id))

    metadata = SumoModelMetadata(
        sumo_model_id=sumo_model_id,
        surrogate_conf_block=surrogate_conf_block,
        input_descriptors=list(input_descriptors),
        output_descriptor=output_descriptor,
        export_format=export_format,
    )
    _metadata_path(models_dir, sumo_model_id).write_text(
        metadata.model_dump_json(indent=2)
    )
    _logger.info(
        "Exported SuMo model %s (%d archive files)", sumo_model_id, len(exported_files)
    )
    return sumo_model_id


def load_model_metadata(sumo_model_id: str) -> SumoModelMetadata:
    """Read back a model's metadata sidecar; raises if the id is unknown."""
    models_dir = get_models_dir()
    path = _metadata_path(models_dir, sumo_model_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No SuMo model found for id '{sumo_model_id}' in {models_dir}"
        )
    return SumoModelMetadata.model_validate_json(path.read_text())


def stage_model_for_import(
    sumo_model_id: str, run_dir: Path
) -> tuple[SumoModelMetadata, Path]:
    """Copy a stored model's archive files, and its stored training-data
    file, into `run_dir` to satisfy Dakota's `import_build_points_file`
    keyword.

    If the stored training-data file is missing (e.g. a model exported
    before this persistence was added, or the file was deleted out-of-band),
    falls back to synthesizing a header-only placeholder from the sidecar's
    descriptors instead of crashing, and logs a loud warning -- this is safe
    because Dakota's import fully reconstructs the surrogate from the
    archive; the points file's row *values* are never read back, only its
    column descriptors need to match by name (R9).

    Returns the model's metadata and the path of the staged points file.
    Archive files keep the `sumo_model_id` as their filename prefix, matching
    Dakota's `import_model filename_prefix=` expectation.
    """
    models_dir = get_models_dir()
    metadata = load_model_metadata(sumo_model_id)

    training_file_name = _training_file_name(sumo_model_id)
    archive_files = [
        f
        for f in sorted(models_dir.glob(f"{sumo_model_id}.*"))
        if f.name not in (f"{sumo_model_id}.metadata.json", training_file_name)
    ]
    if not archive_files:
        raise FileNotFoundError(
            f"SuMo model '{sumo_model_id}' has a metadata sidecar but no archive files in {models_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    for f in archive_files:
        shutil.copy(f, run_dir / f.name)

    stored_training_file = models_dir / training_file_name
    staged_training_file = run_dir / training_file_name
    if stored_training_file.is_file():
        shutil.copy(stored_training_file, staged_training_file)
    else:
        _logger.warning(
            "SuMo model '%s' has no stored training-data file in %s; "
            "synthesizing a header-only placeholder for Dakota's "
            "import_build_points_file. Predictions are unaffected (R9), but "
            "this model's original training data is unavailable.",
            sumo_model_id,
            models_dir,
        )
        descriptors = [*metadata.input_descriptors, metadata.output_descriptor]
        staged_training_file.write_text(" ".join(descriptors) + "\n")

    return metadata, staged_training_file
