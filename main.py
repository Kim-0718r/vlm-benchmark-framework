import torch

# CUDA 사용 가능 여부 및 BF16 지원 여부 확인
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    print("BF16 지원 여부:", torch.cuda.is_bf16_supported())
else:
    print("CUDA를 사용할 수 없습니다.")
