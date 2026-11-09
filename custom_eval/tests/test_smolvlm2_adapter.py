from types import SimpleNamespace

import pytest
import torch

from vlm_eval import EvalSample
from vlm_eval.models import SmolVLM2Adapter


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(
            input_ids=torch.tensor([[10, 20, 30]]),
            pixel_values=torch.ones((1, 3, 4, 4)),
        )
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.messages = None
        self.images = None
        self.template_kwargs = None
        self.video_processor = SimpleNamespace(num_frames=4, fps=2)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, *, text, images, return_tensors):
        self.images = images
        return self.inputs

    def batch_decode(self, token_ids, **kwargs):
        assert token_ids[0].tolist() == [40, 50]
        return ["  dakota digital  "]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generation_kwargs = None

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 20, 30, 40, 50]])


def test_smolvlm2_builds_multimodal_request_and_decodes_answer() -> None:
    image = SimpleNamespace(convert=lambda mode: image)
    adapter = SmolVLM2Adapter(max_new_tokens=32)
    adapter.model = FakeModel()
    adapter.processor = FakeProcessor()
    sample = EvalSample("textvqa", "val", "1", "What brand?", media=[image])

    answer = adapter._generate_text(sample=sample, prompt=sample.question)

    assert answer == "dakota digital"
    assert adapter.processor.messages[0]["content"][0] == {"type": "image"}
    assert adapter.processor.template_kwargs["processor_kwargs"] == {
        "num_frames": 4,
        "fps": 2,
    }
    assert adapter.processor.images == [image]
    assert adapter.model.generation_kwargs["max_new_tokens"] == 32
    assert adapter.model.generation_kwargs["do_sample"] is False
    assert adapter.model.generation_kwargs["num_beams"] == 1
    assert adapter.model.generation_kwargs["use_cache"] is True


def test_smolvlm2_rejects_missing_image() -> None:
    adapter = SmolVLM2Adapter()
    adapter.model = FakeModel()
    adapter.processor = FakeProcessor()
    sample = EvalSample("textvqa", "val", "1", "What brand?")

    with pytest.raises(ValueError, match="has no image"):
        adapter._generate_text(sample=sample, prompt=sample.question)


def test_smolvlm2_validates_settings() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        SmolVLM2Adapter(max_new_tokens=0)
    with pytest.raises(ValueError, match="attn_implementation"):
        SmolVLM2Adapter(attn_implementation="")
