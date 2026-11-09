# VLM 벤치마크 실험 프레임워크

이 프로젝트는 서로 다른 Vision-Language Model(VLM)을 동일한 데이터와 채점 방식으로 비교하기 위한 개인 실험 프레임워크다.

최종 실험 보고서는 [VLM 추론 결과 보고서_김동현.hwp](docs/VLM%20추론%20결과%20보고서_김동현.hwp)에서 확인할 수 있다. 개별 문항의 원본 추론 결과는 용량과 재현 가능성을 고려해 Git 저장소에서 제외하고, `outputs/reports`의 요약 결과만 공개한다.

현재 공통으로 실행하는 벤치마크는 다음 네 가지다.

| 벤치마크 | 전체 문항 | 주로 확인하는 능력 |
|---|---:|---|
| TextVQA | 5,000 | 자연 이미지 안의 문자 이해 및 질의응답 |
| ChartQA | 2,500 | 차트·그래프 이해와 수치 추론 |
| OCRBench | 1,000 | 장면 문자, 문서, 수식 등의 OCR 능력 |
| MME | 2,374 | 인지 및 추론 능력 |

## 1. 프로젝트 구조

```text
vlm_experiment/
├─ .venv-custom-eval/       # Qwen2-VL 및 공통 개발 환경
├─ .venv-openflamingo/      # OpenFlamingo 전용 환경
├─ .venv-internvl/          # InternVL 전용 환경
├─ .venv-smolvlm/           # SmolVLM2 전용 환경
├─ .venv-minicpmv2/         # MiniCPM-V-2 전용 환경
├─ custom_eval/
│  ├─ requirements-*.txt    # 모델별 의존성
│  ├─ scripts/
│  │  ├─ run_benchmark.py       # 벤치마크 하나 실행
│  │  ├─ run_all_benchmarks.py  # 네 벤치마크 연속 실행 및 보고서 생성
│  │  └─ generate_report.py     # 저장된 결과로 보고서 재생성
│  ├─ src/vlm_eval/
│  │  ├─ benchmarks/        # 데이터셋을 EvalSample 형식으로 변환
│  │  ├─ models/            # 모델별 어댑터
│  │  ├─ scoring/           # 벤치마크별 채점 방식
│  │  ├─ storage/           # JSONL 저장 및 재개 상태
│  │  ├─ schemas.py         # 공통 입출력 자료형
│  │  ├─ runner.py          # 추론 반복 실행
│  │  └─ reporting.py       # 비교 보고서 생성
│  └─ tests/                # 자동 테스트
└─ outputs/
   ├─ <benchmark>/<model>/predictions.jsonl
   └─ reports/model_comparison.{md,csv,json}
```

모델마다 별도의 가상환경을 사용하는 이유는 모델별로 요구하는 PyTorch와 Transformers 버전이 다르기 때문이다. GPU와 NVIDIA 드라이버는 컴퓨터 전체에서 공유하지만, CUDA 지원 PyTorch와 Python 라이브러리는 가상환경마다 별도로 설치한다.

## 2. VS Code에서 가상환경 사용하는 방법

이 문서의 명령어는 모두 **VS Code 통합 터미널**에 입력한다. VS Code 메뉴에서 `터미널(Terminal) → 새 터미널(New Terminal)`을 선택한다. 터미널 오른쪽 위 프로필은 현재 사용 중인 `PowerShell`로 둔다.

먼저 VS Code에서 `vlm_experiment` 폴더를 연다. 통합 터미널의 현재 위치가 다음 프로젝트 루트인지 확인한다.

```text
...\vlm_experiment
```

VS Code 통합 터미널에서 Python 3.11로 새 환경을 만든다.

```powershell
py -3.11 -m venv .venv-모델이름
```

생성한 환경을 VS Code의 Python 인터프리터로 선택한다.

1. `Ctrl+Shift+P`를 누른다.
2. `Python: Select Interpreter`를 검색한다.
3. `Enter interpreter path`를 선택한다.
4. `.venv-모델이름\Scripts\python.exe`를 선택한다.
5. 기존 터미널을 휴지통 버튼으로 닫고 `새 터미널`을 연다.

VS Code 설정에서 자동 활성화가 켜져 있으면 새 터미널 앞에 `(.venv-모델이름)`이 자동으로 표시된다. 자동으로 표시되지 않을 때만 통합 터미널에서 다음 두 명령을 실행한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-모델이름\Scripts\Activate.ps1
```

VS Code 통합 터미널이 선택한 환경을 사용 중인지 확인한다.

```powershell
python --version
python -m pip --version
```

터미널 앞에 가상환경 이름이 나오고, pip 경로가 해당 가상환경 내부를 가리켜야 한다. VS Code 화면 오른쪽 아래의 Python 버전도 같은 가상환경을 가리켜야 한다. `pip` 대신 `python -m pip`를 사용하면 VS Code에서 현재 선택한 Python에 설치한다는 점이 명확해진다.

## 3. CUDA용 PyTorch 설치 원리

`https://pytorch.org`는 안내 웹사이트이며 pip 패키지 저장소가 아니다. CUDA 12.1 wheel 저장소는 다음 주소다.

```text
https://download.pytorch.org/whl/cu121
```

예를 들어 MiniCPM-V-2의 공식 버전에 맞는 GPU용 설치 명령은 다음과 같다. `.venv-minicpmv2`가 선택된 VS Code 통합 터미널에 입력한다.

```powershell
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

설치 후 반드시 실제 CUDA 연결을 확인한다.

```powershell
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

정상적인 MiniCPM-V-2 환경의 예상 결과는 다음과 같다.

```text
torch: 2.1.2+cu121
torchvision: 0.16.2+cu121
cuda available: True
torch cuda: 12.1
gpu: NVIDIA GeForce RTX 4060 Ti
```

CUDA Toolkit을 가상환경마다 새로 설치하는 것이 아니다. PyTorch wheel에 실행에 필요한 CUDA 라이브러리가 포함되며, 시스템의 NVIDIA 드라이버를 사용한다.

## 4. 새 모델 추가 전체 절차

### 4.1 모델을 먼저 확정한다

이름이 비슷한 후속 모델과 혼동하지 않도록 다음 정보를 기록한다.

- Hugging Face 공식 모델 ID
- 파라미터 규모
- 공식 지원 Python, PyTorch, Transformers 버전
- 공식 입력 형식과 추론 메서드
- BF16/FP16 지원 여부
- `trust_remote_code=True` 필요 여부
- 예상 VRAM과 라이선스

버전은 블로그나 임의 예제보다 공식 모델 카드의 Usage 항목을 우선한다.

### 4.2 모델 전용 requirements 파일을 만든다

파일 이름은 다음 규칙을 사용한다.

```text
custom_eval/requirements-<모델키>.txt
```

일반적으로 다음 두 묶음이 필요하다.

- 모델 의존성: `transformers`, `accelerate`, `timm`, `sentencepiece` 등
- 프레임워크 의존성: `datasets`, `pyarrow`, `pandas`, `numpy`, `Pillow`

PyTorch와 torchvision은 CUDA wheel 저장소에서 별도로 설치하므로 requirements 파일에서 제외한다.

requirements 설치와 프레임워크 연결:

```powershell
python -m pip install -r .\custom_eval\requirements-<모델키>.txt
python -m pip install -e .\custom_eval
python -m pip check
```

`-e` 설치는 현재 `custom_eval/src` 코드를 가상환경에서 import할 수 있게 한다. 소스 코드를 수정해도 패키지를 매번 재설치할 필요가 없다.

### 4.3 공식 예제로 단독 smoke test를 한다

어댑터부터 작성하지 말고 공식 모델 카드의 최소 예제를 별도 스크립트로 먼저 검증한다.

확인 항목:

1. 모델과 tokenizer/processor가 다운로드되는가?
2. CUDA에 올라가는가?
3. 실제 이미지 한 장을 RGB로 읽는가?
4. 한 질문에 문자열 답변을 생성하는가?
5. VRAM 부족이나 dtype 오류가 없는가?

공식 예제가 실패하면 프레임워크 어댑터를 작성해도 원인 구분이 어렵다.

### 4.4 공통 어댑터를 구현한다

새 파일:

```text
custom_eval/src/vlm_eval/models/<모델키>.py
```

모든 모델 어댑터는 `BaseModelAdapter`를 상속하며 다음 역할을 구현한다.

- `__init__`: 모델 ID, 생성 토큰 수, 해상도 같은 설정 보관
- `_load`: 모델과 tokenizer/processor를 한 번 로드
- `_generate_text`: `EvalSample` 이미지와 공통 prompt를 모델 고유 형식으로 변환하고 문자열 반환
- `_unload`: 모델 참조 제거 및 CUDA 캐시 정리
- `_before_generate`, `_after_generate`: 지연 시간과 최대 VRAM 측정 지원

필수 원칙:

- `sample.media`가 없으면 명확한 오류를 발생시킨다.
- 입력 이미지는 RGB로 변환한다.
- 벤치마크 비교에서는 샘플링을 끄고 결정적 생성을 사용한다.
- 최종 반환값은 앞뒤 공백이 제거된 문자열이어야 한다.
- 모델마다 공식 chat template과 입력 방식을 그대로 사용한다.
- 모델 오류는 runner가 샘플 단위로 기록할 수 있게 한다.

### 4.5 모델을 세 곳에 등록한다

어댑터 작성만으로는 실행 명령에서 모델을 찾을 수 없다. 다음 위치를 모두 수정한다.

1. `custom_eval/src/vlm_eval/models/__init__.py`
   - 새 Adapter import
   - `__all__`에 클래스 추가
2. `custom_eval/scripts/run_benchmark.py`
   - Adapter import
   - `--model` choices에 모델 키 추가
   - 동일한 모델 키를 처리하는 생성 분기 추가
3. `custom_eval/scripts/run_all_benchmarks.py`
   - Adapter import
   - `--model` choices에 모델 키 추가
   - 동일한 모델 키를 처리하는 생성 분기 추가

세 파일에서 모델 키의 대소문자와 철자가 완전히 같아야 한다.

### 4.6 어댑터 단위 테스트를 작성한다

파일:

```text
custom_eval/tests/test_<모델키>_adapter.py
```

실제 대형 모델을 다운로드하지 않고 FakeModel과 FakeTokenizer/Processor를 사용한다. 최소한 다음을 검사한다.

- 이미지와 prompt가 올바른 모델 입력 형식으로 전달되는가?
- 생성 결과를 문자열로 반환하는가?
- 이미지가 없을 때 오류가 발생하는가?
- `max_new_tokens`와 샘플링 설정이 올바른가?
- 잘못된 초기 설정을 거부하는가?

전체 테스트 실행:

```powershell
python -m pytest .\custom_eval\tests -q --basetemp .\custom_eval\.tmp-pytest -p no:cacheprovider
```

`--basetemp`는 Windows 시스템 임시 폴더 권한 문제를 피하기 위해 프로젝트 내부 임시 폴더를 사용한다.

### 4.7 실제 데이터로 단계별 검증한다

먼저 TextVQA 한 문제를 실행한다.

```powershell
python .\custom_eval\scripts\run_benchmark.py --benchmark textvqa --model <모델키> --limit 1 --output .\outputs\pilot_<모델키>_textvqa.jsonl
```

다음으로 네 벤치마크를 두 문제씩 실행한다.

```powershell
python .\custom_eval\scripts\run_all_benchmarks.py --model <모델키> --limit 2 --outputs .\outputs\pilot_<모델키>_all --report-dir .\outputs\pilot_<모델키>_reports
```

통과 기준:

- 네 벤치마크 모두 실행됨
- `error=None`
- 출력이 비어 있지 않음
- VRAM이 8GB 범위 안에 있음
- JSONL과 보고서 세 종류가 생성됨

2문제 점수가 낮은 것은 모델 오류를 의미하지 않는다. 파일럿의 목적은 실행 경로와 채점 연결을 확인하는 것이다.

### 4.8 전체 벤치마크를 실행한다

```powershell
python .\custom_eval\scripts\run_all_benchmarks.py --model <모델키> --outputs .\outputs --report-dir .\outputs\reports
```

실행 순서는 다음과 같다.

```text
TextVQA → ChartQA → OCRBench → MME
```

전체 문항 수는 10,874개이므로 수 시간이 걸릴 수 있다. 게임이나 다른 GPU 작업을 동시에 실행하지 않는다. 모니터만 끄는 것은 괜찮지만 Windows 절전 모드로 들어가면 추론도 멈춘다.

### 4.9 중단 후 이어서 실행한다

전체 실행은 각 답변을 즉시 다음 위치에 추가한다.

```text
outputs/<benchmark>/<model>/predictions.jsonl
```

중간에 터미널이 종료돼도 같은 전체 실행 명령을 다시 실행하면 저장된 행 수를 확인하고 다음 데이터부터 이어간다. 기존 결과를 삭제하거나 다른 출력 폴더를 지정하면 처음부터 시작한다.

이전에 오류로 저장된 샘플까지 다시 시도하려면 `--retry-errors`를 추가한다. 오류 샘플을 그대로 두고 빠르게 재개할 때는 기본 설정을 사용한다.

### 4.10 결과를 확인한다

샘플별 원본 결과:

```text
outputs/<benchmark>/<model>/predictions.jsonl
```

전체 비교 보고서:

```text
outputs/reports/model_comparison.md
outputs/reports/model_comparison.csv
outputs/reports/model_comparison.json
```

Markdown은 사람이 읽기 좋고, CSV는 Excel 분석에 적합하며, JSON은 후속 프로그램 처리에 적합하다.

## 5. MiniCPM-V-2 추가 기록

### 모델 정보

- 공식 모델 ID: `openbmb/MiniCPM-V-2`
- 모델 규모: 약 2.8B~3B
- 프레임워크 모델 키 예정: `minicpm_v_2`
- 어댑터 클래스 예정: `MiniCPMV2Adapter`
- 어댑터 파일 예정: `custom_eval/src/vlm_eval/models/minicpmv2.py`
- 공식 방식: `AutoModel`, `AutoTokenizer`, `trust_remote_code=True`, `model.chat(...)`
- 현재 GPU: RTX 4060 Ti 8GB
- 전용 환경: `.venv-minicpmv2`, Python 3.11.9

### 의존성 파일

`custom_eval/requirements-minicpmv2.txt`:

```text
Pillow==10.1.0
timm==0.9.10
transformers==4.36.0
sentencepiece==0.1.99
peft==0.7.1
accelerate==0.34.2
datasets==3.6.0
pyarrow==18.1.0
pandas>=2.2, <3
numpy==1.26.4
```

### 환경 설치 순서

VS Code 오른쪽 아래에서 `.venv-minicpmv2` 인터프리터를 선택하고 새 통합 터미널을 연다. 터미널 앞에 `(.venv-minicpmv2)`가 보이지 않을 때만 먼저 다음 두 명령으로 활성화한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-minicpmv2\Scripts\Activate.ps1
```

`(.venv-minicpmv2)`가 확인되면 나머지 설치 명령을 실행한다.

```powershell
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r .\custom_eval\requirements-minicpmv2.txt
python -m pip install -e .\custom_eval
python -m pip check
```

### 어댑터 구현 시 MiniCPM 전용 확인 사항

- 모델과 tokenizer 모두 `trust_remote_code=True`로 로드한다.
- RTX 4060 Ti에서는 BF16을 우선 사용한다.
- 입력 이미지를 `PIL.Image`의 RGB로 전달한다.
- `msgs`는 공식 모델 카드 형식을 따른다.
- `model.chat`에는 image, msgs, tokenizer를 모델이 요구하는 위치와 이름으로 전달한다.
- 공정한 벤치마크 비교를 위해 sampling을 끈다.
- `model.chat` 반환값이 tuple이면 실제 답변 문자열만 추출한다.
- 답변에 prompt나 특수 토큰이 섞이지 않았는지 1문제 파일럿에서 확인한다.

### 현재 진행 상태

- [x] `.venv-minicpmv2` 생성
- [x] Python 3.11.9 및 pip 경로 확인
- [x] `requirements-minicpmv2.txt` 작성 및 검토
- [x] CUDA용 torch 및 torchvision 설치
- [x] requirements 설치 및 editable 설치
- [x] CUDA와 라이브러리 버전 확인
- [x] 실제 모델 로딩 및 smoke test
- [x] `MiniCPMV2Adapter` 구현
- [x] 모델 export 및 실행 스크립트 등록
- [x] 어댑터 단위 테스트
- [x] TextVQA 1문제 파일럿
- [x] 네 벤치마크 2문제 파일럿
- [ ] 전체 벤치마크 실행

## 6. 자주 발생했던 문제

### 가상환경 이름은 보이는데 패키지가 없다고 나옴

다른 가상환경이 활성화됐을 가능성이 크다. `python -m pip --version`의 경로를 확인한다.

### `Torch not compiled with CUDA enabled`

CPU용 PyTorch가 설치된 상태다. 올바른 가상환경에서 CUDA wheel 저장소를 지정해 다시 설치한다.

### PowerShell에서 Activate.ps1을 실행할 수 없음

현재 터미널에만 적용되는 다음 설정을 먼저 사용한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### `ModuleNotFoundError: datasets` 또는 `vlm_eval`

모델 전용 requirements 설치와 `python -m pip install -e .\custom_eval`이 모두 필요하다.

### 모델이 CPU를 사용하는 것처럼 느림

`torch.cuda.is_available()`을 확인하고, 로그의 `vram` 값이 계속 기록되는지 확인한다. 첫 실행은 모델 다운로드와 로딩 때문에 오래 걸린다.

### CUDA OOM

다른 GPU 프로그램을 종료하고 다시 실행한다. 필요하면 해당 어댑터의 이미지 타일 수, 입력 해상도 또는 생성 토큰 수를 줄이되, 모델 간 비교 조건이 달라졌음을 결과에 기록한다.

### 한 문항에서 매우 오래 멈춤

고해상도나 특이한 종횡비 이미지가 과도한 이미지 토큰·타일을 만들 수 있다. 현재 GPU 사용량을 확인하고, 어댑터의 이미지 전처리 제한이 실제로 적용되는지 검사한다.

### 결과를 처음부터 다시 돌리고 싶음

기존 `outputs/<benchmark>/<model>` 폴더를 백업한 뒤 새 출력 경로를 지정하는 방법이 안전하다. 결과 삭제는 복구하기 어려우므로 실행 전에 경로를 반드시 확인한다.

## 7. 새 모델 추가 완료 체크리스트

- [ ] 정확한 모델 ID와 라이선스를 확인했다.
- [ ] 모델 전용 가상환경을 만들었다.
- [ ] CUDA용 PyTorch가 `True`로 확인됐다.
- [ ] 모델 전용 requirements를 고정했다.
- [ ] 프레임워크를 editable로 설치했다.
- [ ] 공식 단독 예제로 이미지 한 장 추론에 성공했다.
- [ ] 공통 BaseModelAdapter를 구현했다.
- [ ] 모델을 export 및 두 실행 스크립트에 등록했다.
- [ ] 모델 키 철자가 모든 파일에서 동일하다.
- [ ] 단위 테스트가 통과했다.
- [ ] TextVQA 1문제 파일럿이 통과했다.
- [ ] 네 벤치마크 2문제 파일럿이 통과했다.
- [ ] 전체 결과가 공통 outputs 구조에 저장된다.
- [ ] 비교 보고서에서 새 모델 행을 확인했다.

## 8. 네 모델 공정 비교 프로필

기존 `native` 결과는 각 모델의 권장 이미지 전처리를 사용한다. 정확도 비교에는
유용하지만 Qwen은 256~512 visual token, InternVL은 최대 4개 타일과 thumbnail,
SmolVLM2는 multi-tile, OpenFlamingo는 고정 224px를 사용하므로 latency 조건은
동일하지 않다.

`--profile controlled`는 다음 실행 조건을 공통으로 고정한다.

- batch size 1
- `max_new_tokens=64`
- sampling 비활성화
- greedy decoding (`num_beams=1`)
- KV Cache 활성화
- 모델 전체를 BF16 CUDA 추론으로 실행
- 모델별 공식 prompt/chat template 유지
- 이미지 crop은 모델마다 1개만 사용

이미지 해상도는 모델의 학습 규격 때문에 완전히 같은 값으로 강제하지 않는다.
Controlled 프로필의 입력은 Qwen 448px 상당(256 visual tokens), InternVL 448px,
SmolVLM2 384px, OpenFlamingo 224px이다. 해상도까지 같은 값으로 바꾸면 일부 모델은
학습 분포 밖 입력 또는 connector shape 오류가 생기므로 오히려 모델 비교가 왜곡된다.

Controlled 결과는 기존 결과를 덮어쓰지 않고 다음 위치에 저장된다.

```text
outputs/controlled/<benchmark>/<model>/predictions.jsonl
outputs/reports/controlled/model_comparison.md
```

각 모델의 전용 가상환경에서 전체 controlled 벤치마크를 실행한다.

```powershell
.\.venv-custom-eval\Scripts\python.exe .\custom_eval\scripts\run_all_benchmarks.py --model qwen2vl_2b --profile controlled
.\.venv-openflamingo\Scripts\python.exe .\custom_eval\scripts\run_all_benchmarks.py --model openflamingo_3b_langinstruct --profile controlled
.\.venv-internvl\Scripts\python.exe .\custom_eval\scripts\run_all_benchmarks.py --model internvl2_5_2b --profile controlled
.\.venv-smolvlm\Scripts\python.exe .\custom_eval\scripts\run_all_benchmarks.py --model smolvlm2_2_2b --profile controlled
```

먼저 `--limit 10`을 붙여 네 모델 모두 파일럿을 완료한 뒤 전체 실행하는 것을 권장한다.
중단 후 같은 명령을 다시 실행하면 저장된 문항 다음부터 이어서 처리한다.
