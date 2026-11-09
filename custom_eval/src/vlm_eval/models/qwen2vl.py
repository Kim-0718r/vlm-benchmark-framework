from __future__ import annotations

from typing import Any

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample


class Qwen2VLAdapter(BaseModelAdapter):
    """Qwen2-VL image-question adapter for single-image benchmark samples."""

    def __init__(
        self,
        *,
        name: str = "qwen2vl_2b",
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        max_new_tokens: int = 128,
        device_map: str = "auto",
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("pixel limits must satisfy 0 < min_pixels <= max_pixels")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.model: Any | None = None
        self.processor: Any | None = None

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2VLAdapter requires a CUDA-enabled PyTorch installation")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model.eval()

    def _unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _before_generate(self) -> None:
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def _after_generate(self) -> float | None:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**2)

    def _generate_text(self, *, sample: EvalSample, prompt: str) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen2-VL model resources are not loaded")
        if not sample.media:
            raise ValueError(f"Sample {sample.sample_id} has no image")

        image = sample.media[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[rendered_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        import torch

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not answers:
            raise RuntimeError("Qwen2-VL returned no decoded answer")
        return answers[0].strip()
