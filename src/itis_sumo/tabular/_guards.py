"""Internal input guards shared by the public workflow modules."""

from __future__ import annotations

from collections.abc import Iterable

from itis_sumo.errors import SumoInputError
from itis_sumo.types import (
    PreprocessingSpec,
)


def _reject_unused_overrides(
    preprocessing: PreprocessingSpec | None, columns: Iterable[str]
) -> None:
    """Refuse scale overrides for columns this call never touches (V47st).

    The session-based workflows reject unknown overrides at ``_session.py``;
    the table-mode/sampler entry points must honor the same exact-cover
    contract, so a misspelled log-scale column fails loud instead of silently
    returning linear results.
    """
    unknown = sorted(
        set((preprocessing or PreprocessingSpec()).overrides) - set(columns)
    )
    if unknown:
        raise SumoInputError(
            f"Preprocessing overrides given for columns that are not in play: {unknown}"
        )
