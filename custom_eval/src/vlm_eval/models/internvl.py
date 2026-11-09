from typing import Any
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample

class InternVLAdapter(BaseModelAdapter):
    def __init__(
        self,
        *,
        name: str = "internvl2_5_2b",
        model_id: str = "OpenGVLab/InternVL2_5-2B",
        input_size: int = 448,
        max_num: int = 4,
        max_new_tokens: int = 64,
    ) -> None:
        super().__init__(name=name, model_id=model_id)
        if input_size < 1:
            raise ValueError("input size 1보다 작음")
        if max_num < 1:
            raise ValueError("max_num 1보다 작음")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens 1보다 작음")
        self.input_size = input_size
        self.max_num = max_num
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None

    def _load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("InternVLAdapter requires a CUDA-enabled PyTorch installation")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype = torch.bfloat16,
            low_cpu_mem_usage =True,
            use_flash_attn = False,
            trust_remote_code =True
        ).eval().cuda()
        self.tokenizer= AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            use_fast=False)

    def _prepare_image(self, image: Any) -> torch.Tensor:
        image = image.convert("RGB")
        transform = build_transform(self.input_size)
        images = dynamic_preprocess(
            image,
            image_size=self.input_size,
            max_num=self.max_num,
            use_thumbnail=True,
        )
        pixel_values = torch.stack([transform(tile) for tile in images])
        return pixel_values.to(device="cuda", dtype=torch.bfloat16)

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
            raise RuntimeError("InternVL model resources are not loaded")
        if not sample.media:
            raise ValueError(f"Sample {sample.sample_id} has no image")

        pixel_values = self._prepare_image(sample.media[0])
        question = f"<image>\\n{prompt}"
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }

        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )

        if not isinstance(response, str):
            raise TypeError("InternVL returned a non-string response")
        return response.strip()

        
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
