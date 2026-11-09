from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path


model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    dtype="auto", device_map="auto"
)

min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=min_pixels,
    max_pixels=max_pixels,

)

image_path = Path("08_count.png").resolve()
question = "사진에 보이는 원은 모두 몇개인가요?"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(image_path),
            },
            {"type": "text",
             "text": question}

        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt = True,
)
print(prompt)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)
print("입력 변환 완료")
print(inputs.keys())

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
  )

print("모델 답변:")
print(output_text[0])