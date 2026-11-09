# OpenFlamingo 3B 실행 안내

사용 모델은 `openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct`입니다.
Qwen2-VL과 의존성 버전이 달라서 반드시 `.venv-openflamingo` 환경에서 실행합니다.

## 환경 활성화 및 확인

프로젝트 루트에서 PowerShell로 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-openflamingo\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

정상이면 `2.0.1+cu118 True`가 출력됩니다.

## 먼저 2개만 시험

첫 실행에는 모델 가중치가 다운로드되므로 시간이 걸립니다.

```powershell
python .\custom_eval\scripts\run_benchmark.py --benchmark textvqa --model openflamingo_3b_langinstruct --limit 2
```

결과는 다음 파일에 한 문제씩 즉시 저장됩니다.

```text
outputs/textvqa/openflamingo_3b_langinstruct/predictions.jsonl
```

## 전체 벤치마크 실행

```powershell
python .\custom_eval\scripts\run_all_benchmarks.py --model openflamingo_3b_langinstruct
```

TextVQA, ChartQA, OCRBench, MME 순서로 실행됩니다. 실행이 중단돼도 같은 명령을
다시 실행하면 이미 저장된 sample ID는 건너뛰고 이어서 진행합니다.

최종 비교 리포트는 아래에 생성됩니다.

```text
outputs/reports/summary.csv
outputs/reports/summary.md
```

## 환경을 다시 만들어야 할 때

Python 3.11로 가상환경을 만든 뒤 다음 순서로 설치합니다.

```powershell
py -3.11 -m venv .venv-openflamingo
.\.venv-openflamingo\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r .\custom_eval\requirements-openflamingo.txt
python -m pip install --no-deps git+https://github.com/mlfoundations/open_flamingo.git@main
python -m pip install -e .\custom_eval
```

