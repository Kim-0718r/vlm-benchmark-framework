from types import SimpleNamespace

import pytest
import torch

from vlm_eval import EvalSample
from vlm_eval.models import Qwen2VLAdapter


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.tensor([[10, 20, 30]]))
        self.input_ids = self["input_ids"]
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered prompt"

    def __call__(self, *, text, images, padding, return_tensors):
        self.images = images
        return self.inputs

    def batch_decode(self, token_ids, **kwargs):
        assert token_ids[0].tolist() == [40, 50]
        return ["  dakota  "]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generation_kwargs = None

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 20, 30, 40, 50]])


def test_qwen_adapter_builds_multimodal_request_and_decodes_answer() -> None:
    image = object()
    adapter = Qwen2VLAdapter(max_new_tokens=32)
    adapter.model = FakeModel()
    adapter.processor = FakeProcessor()
    adapter._loaded = True
    sample = EvalSample("textvqa", "val", "1", "What brand?", media=[image])

    output = adapter.generate(sample)

    assert output.text == "dakota"
    assert output.error is None
    assert adapter.processor.messages[0]["content"][0]["image"] is image
    assert adapter.processor.images == [image]
    assert adapter.model.generation_kwargs["max_new_tokens"] == 32
    assert adapter.model.generation_kwargs["do_sample"] is False
    assert adapter.model.generation_kwargs["num_beams"] == 1
    assert adapter.model.generation_kwargs["use_cache"] is True


def test_qwen_adapter_records_missing_image_as_sample_error() -> None:
    adapter = Qwen2VLAdapter()
    adapter.model = FakeModel()
    adapter.processor = FakeProcessor()
    adapter._loaded = True
    sample = EvalSample("textvqa", "val", "1", "What brand?")

    output = adapter.generate(sample)

    assert output.succeeded is False
    assert "has no image" in output.error


def test_qwen_adapter_validates_settings() -> None:
    with pytest.raises(ValueError, match="pixel"):
        Qwen2VLAdapter(min_pixels=100, max_pixels=10)
    with pytest.raises(ValueError, match="max_new_tokens"):
        Qwen2VLAdapter(max_new_tokens=0)
