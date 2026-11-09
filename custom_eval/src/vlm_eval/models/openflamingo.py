from __future__ import annotations

from typing import Any

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample


class OpenFlamingoAdapter(BaseModelAdapter):
    """OpenFlamingo 3B adapter using the MPT-1B Dolly language backbone."""

    def __init__(
        self,
        *,
        name: str = "openflamingo_3b_langinstruct",
        model_id: str = "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        precision: str = "fp16",
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        if max_new_tokens < 1 or num_beams < 1:
            raise ValueError("max_new_tokens and num_beams must be at least 1")
        if precision not in {"fp16", "bf16"}:
            raise ValueError("precision must be 'fp16' or 'bf16'")
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.precision = precision
        self.model: Any | None = None
        self.image_processor: Any | None = None
        self.tokenizer: Any | None = None

    def _load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms

        if not torch.cuda.is_available():
            raise RuntimeError("OpenFlamingoAdapter requires CUDA")
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            cross_attn_every_n_layers=1,
        )
        checkpoint_path = hf_hub_download(self.model_id, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint, strict=False)
        model_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.model = self.model.to(device="cuda", dtype=model_dtype).eval()
        self.tokenizer.padding_side = "left"

    def _unload(self) -> None:
        self.model = self.image_processor = self.tokenizer = None
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
        if self.model is None or self.image_processor is None or self.tokenizer is None:
            raise RuntimeError("OpenFlamingo resources are not loaded")
        if not sample.media:
            raise ValueError(f"Sample {sample.sample_id} has no image")

        import torch

        image = sample.media[0].convert("RGB") if hasattr(sample.media[0], "convert") else sample.media[0]
        vision_x = self.image_processor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        model_dtype = next(self.model.parameters()).dtype
        text = f"<image>Question: {prompt}\nAnswer:"
        tokenized = self.tokenizer([text], return_tensors="pt")
        input_length = tokenized["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                vision_x=vision_x.to(device="cuda", dtype=model_dtype),
                lang_x=tokenized["input_ids"].cuda(),
                attention_mask=tokenized["attention_mask"].cuda(),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                use_cache=True,
            )
        answer = self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=False)
        return answer.split("<|endofchunk|>", 1)[0].replace("<|endoftext|>", "").strip()
