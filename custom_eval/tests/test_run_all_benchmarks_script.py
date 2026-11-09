import importlib.util
import sys
from pathlib import Path


def test_full_limits_cover_all_supported_benchmarks() -> None:
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_all_benchmarks.py"
        spec = importlib.util.spec_from_file_location("run_all_benchmarks", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    assert module.FULL_LIMITS == {
        "textvqa": 5000, "chartqa": 2500, "ocrbench": 1000, "mme": 2374
    }
