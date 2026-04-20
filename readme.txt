CCEpro - Stable Diffusion 1.4 Concept Erasure Project
=====================================================

1. 프로젝트 목적
----------------
이 프로젝트는 Stable Diffusion 1.4 기반 모델에서 특정 개념을 제거하는 실험을 수행하기 위한 코드베이스이다.

현재 목표:
- SD 1.4 베이스라인 로드
- 학습용 데이터셋 메타데이터(JSON) 기반 학습 루프 실행
- 체크포인트 저장
- 중간 샘플 이미지 저장

주의:
- 현재 baseline 코드는 "프롬프트 필터링"이나 "reverse guidance inference"가 아니라,
  이미지/프롬프트 데이터를 이용해 UNet을 학습하는 기반 코드이다.
- 이후 continual learning / reverse guidance / associative suppression 등을 추가할 예정.

2. 권장 환경
------------
권장 이유:
기존 base/anaconda 환경은 numpy / urllib3 / diffusers 호환성 문제로 쉽게 꼬일 수 있으므로
프로젝트 전용 conda 환경을 새로 만드는 것을 권장한다.

권장 버전:
- Python 3.10
- numpy < 2
- urllib3 < 2

3. 새 환경 생성
---------------
(1) conda 환경 생성
    conda create -n ccepro python=3.10 -y

(2) 환경 활성화
    conda activate ccepro

4. PyTorch 설치
---------------
아래는 CUDA 12.1 기준 예시이다.
사용 중인 CUDA 버전이 다르면 해당 버전에 맞는 공식 wheel을 설치해야 한다.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

만약 GPU를 사용하지 않거나 CUDA를 모르면, CPU 또는 자신의 CUDA 버전에 맞게 설치한다.

5. 필수 패키지 설치
-------------------
    pip install "numpy<2" "urllib3<2" diffusers transformers accelerate safetensors pillow tqdm

6. 설치 확인
------------
아래 명령들이 모두 정상 동작해야 한다.

    python -c "import numpy; print('numpy', numpy.__version__)"
    python -c "import urllib3; print('urllib3', urllib3.__version__)"
    python -c "import torch; print('torch', torch.__version__)"
    python -c "import diffusers; print('diffusers ok')"
    python -c "from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler; print('diffusers modules ok')"
    python -c "import torchvision, PIL, tqdm; print('extra packages ok')"

7. 프로젝트 폴더 구조 예시
---------------------------
    CCEpro/
    ├─ train.py
    ├─ src/
    │  ├─ model_loader.py
    │  ├─ dataset.py
    │  ├─ losses.py
    │  ├─ trainer.py
    │  ├─ sampler.py
    │  └─ utils.py
    ├─ data/
    │  ├─ erase/
    │  ├─ retain/
    │  └─ metadata.json
    └─ outputs/

8. 데이터 준비
--------------
이미지 파일을 아래처럼 정리한다.

    data/
    ├─ erase/
    │  ├─ img_001.png
    │  ├─ img_002.png
    ├─ retain/
    │  ├─ img_101.png
    │  ├─ img_102.png
    └─ metadata.json

9. metadata.json 형식
---------------------
metadata.json은 직접 만들어야 한다.
학습에 사용할 이미지 경로, 프롬프트, mode를 정의한다.

예시:
[
  {
    "image_path": "data/erase/img_001.png",
    "prompt": "a painting in the style of Van Gogh",
    "mode": "erase"
  },
  {
    "image_path": "data/retain/img_101.png",
    "prompt": "a photo of a dog in a park",
    "mode": "retain"
  }
]

주의:
- image_path는 실제 파일 경로와 정확히 일치해야 한다.
- mode는 현재 "erase", "retain"을 사용한다.
- train.py를 실행하는 위치 기준 상대경로로 맞추는 것이 가장 편하다.

10. metadata.json 자동 생성 예시
--------------------------------
필요하면 아래 스크립트를 make_metadata.py로 저장해서 사용할 수 있다.

    import os
    import json

    data = []

    for fname in os.listdir("data/erase"):
        data.append({
            "image_path": f"data/erase/{fname}",
            "prompt": "a painting in the style of Van Gogh",
            "mode": "erase"
        })

    for fname in os.listdir("data/retain"):
        data.append({
            "image_path": f"data/retain/{fname}",
            "prompt": "a photo of a dog in a park",
            "mode": "retain"
        })

    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

실행:
    python make_metadata.py

11. 학습 실행 예시
------------------
GPU 사용 예시:
    python train.py \
      --metadata_json data/metadata.json \
      --output_dir outputs/base_train \
      --device cuda \
      --batch_size 2 \
      --num_epochs 1 \
      --learning_rate 1e-5

CPU 사용 예시:
    python train.py \
      --metadata_json data/metadata.json \
      --output_dir outputs/base_train \
      --device cpu \
      --batch_size 1 \
      --num_epochs 1 \
      --learning_rate 1e-5

12. 현재 baseline 코드가 하는 일
-------------------------------
현재 baseline은 다음을 수행한다.

- Stable Diffusion 1.4 구성요소 로드
- dataset / dataloader 구성
- VAE로 이미지 latent 인코딩
- noise 추가
- UNet noise prediction 학습
- mode에 따라 loss 분기
- 주기적으로 체크포인트 저장
- 주기적으로 sample 이미지 저장

즉, 현재 코드는 학습 코드이다.
프롬프트 필터링이나 reverse guidance inference 코드는 아직 포함되지 않았다.

13. 출력 결과 위치
------------------
예시:
    outputs/base_train/
    ├─ checkpoints/
    │  ├─ step_500/
    │  └─ step_1000/
    └─ samples/
       ├─ step_200/
       ├─ step_400/
       └─ ...

14. 자주 나는 에러와 해결법
--------------------------
(1) ModuleNotFoundError: No module named 'torchvision'
해결:
    pip install torchvision

(2) cannot import name 'DEFAULT_CIPHERS' from urllib3.util.ssl_
원인:
    urllib3 2.x 호환성 문제
해결:
    pip install "urllib3<2"

(3) numpy.dtype size changed, may indicate binary incompatibility
원인:
    numpy와 다른 C-extension 패키지 ABI 충돌
해결:
    새 conda 환경을 만들고 아래처럼 재설치
    conda create -n ccepro python=3.10 -y
    conda activate ccepro
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install "numpy<2" "urllib3<2" diffusers transformers accelerate safetensors pillow tqdm

15. 권장 작업 순서
------------------
1) 새 환경 생성
2) 패키지 설치
3) import 테스트
4) data/erase, data/retain 이미지 준비
5) data/metadata.json 생성
6) train.py 실행
7) outputs 안에 checkpoint / sample이 생성되는지 확인
8) baseline이 정상 동작하면 이후 reverse guidance / continual memory 추가

16. 다음 개발 예정
------------------
baseline 학습 코드가 정상 동작하면 이후 아래 기능을 추가할 수 있다.

- reverse guidance inference
- target concept + related terms suppression
- continual memory
- replay mode
- sequential concept accumulation
- continual learning evaluation

17. 빠른 재시작 체크리스트
--------------------------
프로젝트를 나중에 다시 시작할 때는 아래 순서대로 확인한다.

[ ] conda activate ccepro
[ ] python -c "import torch, diffusers, numpy, urllib3"
[ ] data/metadata.json 존재 확인
[ ] data/erase, data/retain 이미지 존재 확인
[ ] train.py 경로 확인
[ ] GPU 사용 가능 여부 확인
[ ] train.py 실행
[ ] outputs 폴더 생성 확인

18. 예시 전체 명령어 모음
-------------------------
환경 생성:
    conda create -n ccepro python=3.10 -y
    conda activate ccepro

PyTorch 설치 (CUDA 12.1 예시):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

기타 설치:
    pip install "numpy<2" "urllib3<2" diffusers transformers accelerate safetensors pillow tqdm

metadata 생성:
    python make_metadata.py

학습 실행:
    python train.py \
      --metadata_json data/metadata.json \
      --output_dir outputs/base_train \
      --device cuda \
      --batch_size 2 \
      --num_epochs 1 \
      --learning_rate 1e-5

끝.