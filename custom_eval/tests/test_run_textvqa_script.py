import importlib.util
from pathlib import Path


def test_run_textvqa_script_can_be_imported() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_textvqa.py"
    spec = importlib.util.spec_from_file_location("run_textvqa", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
