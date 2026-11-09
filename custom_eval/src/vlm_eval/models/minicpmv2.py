from __future__ import annotations

from typing import Any

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample


class MiniCPMV2Adapter(BaseModelAdapter):
    """MiniCPM-V 2.0 single-image adapter for the shared benchmark runner."""

    def __init__(
        self,
        *,
        name: str = "minicpm_v_2",
        model_id: str = "openbmb/MiniCPM-V-2",
        max_new_tokens: int = 64,
        max_input_length: int = 2048,
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if max_input_length < 1:
            raise ValueError("max_input_length must be at least 1")
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPMV2Adapter requires a CUDA-enabled PyTorch installation")

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().to(device="cuda", dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def _unload(self) -> None:
        self.model = None
        self.tokenizer = None
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
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MiniCPM-V 2.0 model resources are not loaded")
        if not sample.media:
            raise ValueError(f"Sample {sample.sample_id} has no image")

        import torch

        image = sample.media[0]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        messages = [{"role": "user", "content": prompt}]

        with torch.inference_mode():
            result = self.model.chat(
                image=image,
                msgs=messages,
                context=None,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                max_inp_length=self.max_input_length,
                sampling=False,
                num_beams=1,
            )

        answer = result[0] if isinstance(result, tuple) else result
        if not isinstance(answer, str):
            raise TypeError("MiniCPM-V 2.0 returned a non-string answer")
        return answer.strip()
