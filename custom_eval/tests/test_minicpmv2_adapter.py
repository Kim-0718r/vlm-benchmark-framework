from types import SimpleNamespace

import pytest

from vlm_eval import EvalSample
from vlm_eval.models import MiniCPMV2Adapter


class FakeModel:
    def __init__(self) -> None:
        self.chat_kwargs = None

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return "  dakota digital  ", [{"role": "assistant"}], {"num_beams": 3}


class FakeTokenizer:
    pass


def test_minicpmv2_builds_chat_request_and_extracts_answer() -> None:
    image = SimpleNamespace(convert=lambda mode: image)
    adapter = MiniCPMV2Adapter(max_new_tokens=32, max_input_length=1024)
    adapter.model = FakeModel()
    adapter.tokenizer = FakeTokenizer()
    sample = EvalSample("textvqa", "val", "1", "What brand?", media=[image])

    answer = adapter._generate_text(sample=sample, prompt=sample.question)

    assert answer == "dakota digital"
    assert adapter.model.chat_kwargs["image"] is image
    assert adapter.model.chat_kwargs["msgs"] == [
        {"role": "user", "content": "What brand?"}
    ]
    assert adapter.model.chat_kwargs["context"] is None
    assert adapter.model.chat_kwargs["tokenizer"] is adapter.tokenizer
    assert adapter.model.chat_kwargs["max_new_tokens"] == 32
    assert adapter.model.chat_kwargs["max_inp_length"] == 1024
    assert adapter.model.chat_kwargs["sampling"] is False
    assert adapter.model.chat_kwargs["num_beams"] == 1


def test_minicpmv2_rejects_missing_image() -> None:
    adapter = MiniCPMV2Adapter()
    adapter.model = FakeModel()
    adapter.tokenizer = FakeTokenizer()
    sample = EvalSample("textvqa", "val", "1", "What brand?")

    with pytest.raises(ValueError, match="has no image"):
        adapter._generate_text(sample=sample, prompt=sample.question)


def test_minicpmv2_validates_settings() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        MiniCPMV2Adapter(max_new_tokens=0)
    with pytest.raises(ValueError, match="max_input_length"):
        MiniCPMV2Adapter(max_input_length=0)
