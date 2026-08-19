import concurrent.futures
import contextlib
import logging
import os
from pathlib import Path

import dakota.environment as dakenv

from itis_sumo.core import wiofiles as wio

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(path: Path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


# Static function to execute Dakota - needs to be at module level to be picklable
def _dak_exec_static(conf: str) -> tuple[str | None, str | None]:
    """Static version of dak_exec that can be pickled for multiprocessing."""
    study = None
    stdoutstr, stderrstr = None, None
    exec_error: Exception | None = None
    with wio.capture_to_file(stdout="./stdout", stderr="./stderr") as (stdout, stderr):
        try:
            # NIDR input-file parsing (e.g. rejecting `seed = 0`) happens during
            # study construction, not just study.execute() - construct it inside
            # this same capture+try block so those aborts are captured/enriched
            # too, not just execution-time ones (flaskapi/SPEC.md B17).
            study = dakenv.study(callback=None, input_string=conf)  # type: ignore
            study.execute()
        except Exception as exc:  # noqa: BLE001 - re-raised below with captured diagnostics
            exec_error = exc
    # Always read back captured stdout/stderr, even on failure - Dakota's own
    # exception (e.g. "Dakota aborted: Unknown error 254") carries no detail,
    # but the captured stdout/stderr usually contains the real NIDR/input-file
    # error message that caused the abort (flaskapi/SPEC.md B16).
    with open(stdout) as outf, open(stderr) as errf:
        stdoutstr = outf.read()
        stderrstr = errf.read()
    del study
    if exec_error is not None:
        diagnostics = "\n".join(
            s.strip() for s in (stdoutstr, stderrstr) if s and s.strip()
        )
        message = f"Dakota execution failed: {exec_error}"
        if diagnostics:
            message = f"{message}\n--- Dakota stdout/stderr ---\n{diagnostics}"
        raise RuntimeError(message) from exec_error
    return stdoutstr, stderrstr


class DakotaObject:
    def __init__(self) -> None:
        logger.info("DakotaObject created")

    def run(self, dakota_conf: str, output_dir: Path) -> None:
        logger.info("Starting Dakota run")
        with working_directory(output_dir):
            # Create a picklable version of the callback
            stdout, stderr = self.future_exec(conf=dakota_conf)
            logger.info("Dakota run finished")
            with (
                open("dakota_stdout.txt", "w") as f_out,
                open("dakota_stderr.txt", "w") as f_err,
            ):
                if stdout:
                    f_out.write(stdout)
                if stderr:
                    f_err.write(stderr)
            if stderr:
                logger.warning("Dakota produced stderr output: %s", stderr)

    def future_exec(self, conf):
        # Use the static function directly rather than the instance method
        with concurrent.futures.ProcessPoolExecutor(1) as pool:
            future = pool.submit(_dak_exec_static, conf)
        return future.result()
