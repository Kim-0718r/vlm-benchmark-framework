import pytest

from vlm_eval.models import OpenFlamingoAdapter


def test_openflamingo_defaults_to_official_langinstruct_checkpoint() -> None:
    adapter = OpenFlamingoAdapter()
    assert adapter.model_id == "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct"
    assert adapter.name == "openflamingo_3b_langinstruct"
    assert adapter.num_beams == 1
    assert adapter.precision == "fp16"


def test_openflamingo_validates_generation_settings() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        OpenFlamingoAdapter(max_new_tokens=0)
    with pytest.raises(ValueError, match="num_beams"):
        OpenFlamingoAdapter(num_beams=0)
    with pytest.raises(ValueError, match="precision"):
        OpenFlamingoAdapter(precision="fp32")
