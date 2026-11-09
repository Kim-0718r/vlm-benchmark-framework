import importlib.util
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompts_are_benchmark_specific() -> None:
    module = load_script()
    assert "yes or no" in module.prompt_for("mme", "Question?")
    assert "single word or phrase" in module.prompt_for("chartqa", "Question?")


def test_controlled_profile_uses_one_crop_and_common_generation_budget() -> None:
    module = load_script()

    qwen = module.create_model("qwen2vl_2b", 64, profile="controlled")
    internvl = module.create_model("internvl2_5_2b", 64, profile="controlled")
    flamingo = module.create_model("openflamingo_3b_langinstruct", 64, profile="controlled")
    smol = module.create_model("smolvlm2_2_2b", 64, profile="controlled")

    assert qwen.min_pixels == qwen.max_pixels == 256 * 28 * 28
    assert internvl.max_num == 1
    assert flamingo.num_beams == 1
    assert flamingo.precision == "bf16"
    assert smol.single_crop is True
    assert {
        qwen.max_new_tokens,
        internvl.max_new_tokens,
        flamingo.max_new_tokens,
        smol.max_new_tokens,
    } == {64}
