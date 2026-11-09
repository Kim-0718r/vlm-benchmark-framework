from __future__ import annotations

from typing import Any

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample


class SmolVLM2Adapter(BaseModelAdapter):
    """SmolVLM2 single-image adapter for the shared benchmark runner."""

    def __init__(
        self,
        *,
        name: str = "smolvlm2_2_2b",
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        max_new_tokens: int = 64,
        attn_implementation: str = "sdpa",
        single_crop: bool = False,
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not attn_implementation.strip():
            raise ValueError("attn_implementation must not be empty")
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.single_crop = single_crop
        self.model: Any | None = None
        self.processor: Any | None = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("SmolVLM2Adapter requires a CUDA-enabled PyTorch installation")

        processor_kwargs = {}
        if self.single_crop:
            # Keep the native 384px SigLIP crop while disabling the default
            # multi-tile path whose longest edge is 1920px.
            processor_kwargs = {
                "size": {"longest_edge": 384},
                "max_image_size": {"longest_edge": 384},
            }
        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            _attn_implementation=self.attn_implementation,
        ).eval().to("cuda")

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
            raise RuntimeError("SmolVLM2 model resources are not loaded")
        if not sample.media:
            raise ValueError(f"Sample {sample.sample_id} has no image")

        import torch

        image = sample.media[0]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # SmolVLMProcessor injects these video defaults into **kwargs when
            # processor_kwargs is empty, which emits one warning per image.
            # Supplying them explicitly keeps image-only benchmark logs clean.
            processor_kwargs={
                "num_frames": self.processor.video_processor.num_frames,
                "fps": self.processor.video_processor.fps,
            },
        )
        inputs = self.processor(
            text=[rendered_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        answer_ids = generated_ids[:, input_length:]
        answers = self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not answers:
            raise RuntimeError("SmolVLM2 returned no decoded answer")
        return answers[0].strip()
