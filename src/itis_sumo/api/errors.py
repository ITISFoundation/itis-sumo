"""Stable error taxonomy for the itis-sumo consumer API (SPEC V23er).

Every error that escapes ``itis_sumo.api`` is a :class:`SumoError`. Consumers
classify failures by catching one of the three subclasses -- an HTTP service maps
them onto 400 / 422 / 500 -- instead of matching on message text.
"""

from __future__ import annotations

from pathlib import Path


class SumoError(Exception):
    """Base class for every error crossing the ``itis_sumo.api`` boundary."""


class SumoInputError(SumoError):
    """The supplied samples or configuration are unusable.

    The caller can fix this by sending different data or configuration.
    """


class SumoResultError(SumoError):
    """The run finished, but did not produce the values that were requested.

    Nothing about the request was wrong, so retrying it unchanged will not help.
    """


class SumoEngineError(SumoError):
    """The Dakota engine itself failed.

    Dakota reports failures opaquely -- ``Dakota aborted: Unknown error 250``,
    ``IndexError: map::at`` -- and the only tractable evidence is the run
    directory that produced them. That directory is therefore kept alive when a
    run fails (it is discarded on success), and its path travels with this error
    alongside the tail of Dakota's own stderr. See SPEC V24af.
    """

    def __init__(
        self,
        message: str,
        *,
        run_dir: Path | None = None,
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.run_dir = run_dir
        self.stderr_tail = stderr_tail

    def __str__(self) -> str:
        text = super().__str__()
        if self.run_dir is not None:
            text += f"\nRun directory preserved at: {self.run_dir}"
        if self.stderr_tail:
            text += f"\nDakota stderr (tail):\n{self.stderr_tail}"
        return text
