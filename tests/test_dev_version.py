import io
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from packaging.version import Version

_spec = spec_from_file_location(
    "dev_version", Path(__file__).parents[1] / "scripts/dev_version.py"
)
dev_version = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dev_version)


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_next_version_advances_past_published_testpypi_release(monkeypatch):
    monkeypatch.setattr(dev_version, "current_version", lambda: "0.1.0a1.dev1")
    monkeypatch.setattr(
        dev_version,
        "existing_dev_versions",
        lambda base: [Version("0.1.0a1.dev1")],
    )
    response = JsonResponse(
        json.dumps({"releases": {"0.1.0a1.dev1": [{"filename": "wheel"}]}}).encode()
    )
    monkeypatch.setattr(dev_version, "urlopen", lambda *args, **kwargs: response)

    assert dev_version.next_version() == "0.1.0a1.dev2"
