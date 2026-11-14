# CLAUDE.md — RecurrentJSCC

이 저장소는 `Semantic_main`의 `separ` 브랜치(인코더/디코더 분리 DeepJSCC 베이스라인)에서 출발해
**RecurrentJSCC**(가중치 공유 반복 코어 + FiLM 깊이 조건화 + 정적 언롤링)를 깔끔하게 다시 구현하고,
학습 → 시뮬레이션 평가 → NPU(Rubik Pi 3) 실측까지 재현 가능한 파이프라인을 만드는 것이 목표다.
이전 RecurrentJSCC 구현은 버린다. 참고할 코드는 `separ` 브랜치뿐이다.

Claude Code는 GPU 서버에서 실행된다.

## 0. 작업 원칙

- 작업 브랜치는 `separ`에서 딴 `recurrent`. `separ`에는 직접 커밋하지 않는다.
- 논문에 들어갈 숫자는 전부 스크립트가 `results/` 아래에 쓴 JSON/CSV에서만 나온다. 손으로 옮겨 적거나 추정치를 채우지 않는다. 측정하지 못한 값은 비워 두고 못 했다고 보고한다.
- 장시간 학습(수 시간 이상)은 사용자 확인 후 `tmux`에서 시작한다. 그 전에 1 epoch 미만의 스모크 런으로 끝까지 도는지 확인한다.
- GPU를 쓰는 실행 전에 `nvidia-smi`로 다른 작업이 도는지 확인한다. 사용자가 띄운 프로세스를 죽이거나 그 실험 디렉터리를 수정하지 않는다.
- 권한·인증 오류가 나면 우회하지 말고 멈춘다. `sudo`가 필요한 일은 사용자에게 요청한다.
- §9의 [USER] 항목은 시도하지 말고, 필요한 시점에 무엇이 필요한지 정확히 요청한다.
- NPU 툴체인 관련 사실(§7의 "검증 필요")은 기억에 의존하지 말고 onnxruntime 문서와 보드에서의 실제 실행으로 확인한 뒤 진행한다.
- 스펙이 모호하면 §10의 미결 사항인지 보고, 해당하면 기본값으로 진행하되 보고에 명시한다.
- 요청받지 않은 기능을 추가하지 않는다.

## 1. 환경

기계는 두 대, Python 환경은 두 개다. 섞지 않는다.

| 환경 | 기계 | Python | 하는 일 | 위치 |
|---|---|---|---|---|
| train | GPU 서버 (RTX 5090, x86_64 Linux) | 3.12, uv | 학습, 평가, FLOPs, ONNX export, QDQ 양자화, 캘리브레이션 데이터, 번들, 결과 파싱, 그림 | 저장소 루트 |
| board | Rubik Pi 3 (QCS6490, aarch64, Ubuntu 24.04 LTS) | 3.12, uv 프로젝트 `recurrent-npu-jscc` | NPU용 컴파일(세션 생성 시 자동), 실행, 측정 | `bench/` |

보드 환경(사용자가 이미 구성): `numpy`, `pillow`, `onnxruntime-qnn 1.23.0`(로컬 aarch64 휠). `get_available_providers()`는 `['QNNExecutionProvider', 'CPUExecutionProvider']`를 돌려준다. 보드에 패키지를 추가해야 하면 먼저 사용자에게 묻는다.

- PyTorch는 **cu128 고정**(`pyproject.toml`의 `pytorch-cu128` 인덱스 유지). RTX 5090은 cu126 휠에서 지원되지 않으므로 cu126으로 내리지 않는다.
- 의존성 추가는 `uv add`, 개발용은 `uv add --dev`(pytest, ruff). `pip install`을 직접 쓰지 않는다. `uv.lock`은 커밋한다.
- 서버와 보드는 ONNX 파일과 npy 파일로만 주고받는다.
- **서버의 `onnxruntime`을 보드와 같은 `1.23.*`로 고정한다**(현재 `>=1.32.0`). 양자화 도구가 보드가 읽을 수 있는 형식만 내게 하기 위해서다. `onnx` 패키지도 그 버전과 호환되는 범위로 맞추고, export opset은 17을 유지한다. 번들에 넣는 모든 ONNX는 서버의 onnxruntime 1.23 CPU EP로 로드·실행되는지 테스트한다. 이 테스트가 통과하면 보드도 읽는다.
- 보드에는 PyTorch를 설치하지 않는다. 보드에서는 학습·양자화·변환을 하지 않는다. PSNR은 보드가 저장한 출력을 서버로 가져와 계산한다.
- 보드 주소는 `ubuntu@192.168.1.9`지만 사용자 로컬 PC와 같은 Wi-Fi의 사설 주소라 **GPU 서버에서 직접 닿지 않는다**. 접속 방법은 둘 중 하나다.
  - 역방향 터널: 사용자가 로컬 PC에서 `ssh -R 2222:192.168.1.9:22 <서버>`를 열어 두면 서버에서 `ssh -p 2222 ubuntu@localhost`로 보드에 들어간다. 접속 명령은 환경 변수 `BOARD_SSH`로 받는다.
  - 중계: 터널이 없으면 번들을 만들고 멈춘다. 사용자가 로컬 PC에서 `tools/relay.sh`(서버 → 로컬 → 보드 전송, 보드에서 실행, 결과를 서버로 회수)를 돌린다.
  - 보드 작업 전에 `BOARD_SSH`로 접속되는지 먼저 확인하고, 안 되면 중계 방식으로 넘어간다. 접속을 뚫으려고 다른 방법을 시도하지 않는다.

```sh
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python train.py --config_file configs/recurrent/joint_rel.yaml
uv run python evaluate.py logs/<exp> --iters all
uv run python export_onnx.py logs/<exp> --enc-iters 1 2 4 6 --dec-iters 1 2 4 6 --static 256 256
```

## 2. 베이스(separ)에서 유지할 것과 고칠 것

유지한다:
- 인코더/디코더가 별도 `nn.Module`이고 `nn.ModuleDict({'encoder', 'decoder'})`로 묶는 구조. 채널은 어느 모듈에도 속하지 않는다.
- 심볼 텐서 규약: `(B, C, h, w)` 실수 텐서, 인접한 채널 쌍이 복소 심볼의 실수부/허수부. rate는 앞쪽 채널만 남기는 prefix truncation, 단위는 spp(symbols per pixel). CBR = spp / 3.
- `jsonargparse` + YAML config, CLI override(`--model.dim 96` 형태).
- 실험 디렉터리 `logs/<experiment_name>/`: `config.yaml`, `logging.log`, `train/`·`valid/` TensorBoard, `latest_ckpt_ep*.pth.tar`, `best_ckpt_ep*.pth.tar`, `weights/Epoch_*.pth.tar`.
- 체크포인트 키: `epoch`, `state_dict`, `loss`, `optimizer`, `lr_scheduler`.

고친다(현재 코드의 문제):
- `utils/logging.py::load_ckpt`의 bare `except`가 로드 실패를 삼키고 처음부터 학습한다 → 실패 시 예외를 그대로 올린다.
- resume 시 `best_loss`가 초기화된다 → 체크포인트에 `best_loss` 저장/복원.
- `config.yaml`이 `!!python/object` 덤프다 → 순수 dict로 저장하고 `yaml.safe_load`로 읽히게 한다. git commit hash도 함께 남긴다.
- validation이 랜덤 리사이즈+랜덤 크롭이라 epoch마다 값이 흔들린다 → 학습 중 validation은 고정 center crop, 최종 평가는 `evaluate.py`에서 원본 해상도 전체 이미지.
- seed가 `torch`/`random`만 고정된다 → numpy, DataLoader worker까지 고정.
- `utils/parsing.py`가 모델 타입을 하드코딩한다 → 모델 레지스트리(`model/__init__.py`)로 교체.
- `AFModule`이 SNR을 Python float로 받는다 → 텐서 입력으로 바꿔 export 그래프의 입력이 되게 한다.
- `AWGNChannel`이 배치 전체의 평균 전력을 쓴다 → 샘플별 전력 정규화를 명시적 함수로 분리(`channel.py`).
- `from x import *`를 명시적 import로 바꾼다. 타입 힌트를 단다.
- 커밋된 `logs/BASELINE/`은 `recurrent` 브랜치에서 제거하고 `.gitignore`의 `*/logs/*`를 `logs/`로 고친다.

## 3. 디렉터리 구조(목표)

```
configs/
  baseline.yaml                 # 기존 DeepJSCC_lite (회귀 확인용)
  recurrent/*.yaml              # §6의 실험별 config
model/
  __init__.py                   # 레지스트리: build_model(cfg) -> ModuleDict
  blocks.py                     # ConvNeXtBlock, 정규화, AFModule, FiLM
  recurrent_jscc.py             # RecurrentEncoder / RecurrentDecoder
  deep_jscc.py                  # 기존 베이스라인
channel.py                      # power_normalize, awgn (모델 그래프 밖)
criterion.py
train.py  evaluate.py  export_onnx.py  run_inference.py
utils/                          # dataset, logging, parsing, progress_meter, gpus, complexity
tools/
  quantize.py                   # ONNX -> QDQ 양자화 ONNX (onnxruntime.quantization)
  make_calib.py                 # 캘리브레이션/벤치 입력 생성
  make_bundle.py                # 보드로 보낼 tar 묶음 생성
  collect_npu.py                # 보드 결과 파싱 -> results/npu/
  relay.sh                      # 사용자 로컬 PC에서 실행: 서버 <-> 보드 중계
  plots.py                      # 논문 그림/표 (results/만 읽는다)
bench/                          # 보드에서 실행. numpy + onnxruntime-qnn + pillow만 사용
tests/
results/                        # 스크립트 산출물(JSON/CSV). 커밋 대상
artifacts/                      # ONNX, QDQ ONNX, 캘리브레이션. git 제외
```

## 4. 모델 스펙

### 4.1 전체 구조
대칭 2-stage, 잠재 해상도 H/8.

- 인코더: `Conv(↓4)` → ConvNeXt 블록 N₁개(독립 가중치, 채널 96) → AF → `Conv(↓2)` → **코어 F_θ를 M₁회 반복**(채널 192) → AF → `1×1 Conv`로 심볼 채널 → prefix truncation(spp).
- 디코더: (그래프 밖에서 zero padding으로 채널 복원) → `1×1 Conv` → AF → **코어 F_θ를 M₂회 반복** → `ConvTranspose(↑2)` → AF → ConvNeXt 블록 N₁개 → `ConvTranspose(↑4)` → clamp.
- 인코더 코어와 디코더 코어는 서로 다른 파라미터다. 각 코어 안에서만 반복 간 공유한다.
- AF와 rate truncation은 코어 밖에 둔다(SNR/rate 적응과 깊이 적응의 분리).
- 전력 정규화, AWGN, zero padding, 크롭은 **모델 그래프 밖**에 둔다.

### 4.2 코어와 깊이
- 코어 F_θ = ConvNeXt 블록 **3개**(`model.core_blocks: 3`), shape-preserving.
- 반복 횟수 K ∈ **{1, 2, 4, 6}** → 깊이(= 3K) **3, 6, 12, 18**. config 키는 `model.iters: [1, 2, 4, 6]`.
- 인코더 K(M₁)와 디코더 K(M₂)는 독립. 조합 16개.
- `forward(..., iters: int)`에서 `iters`는 Python int이고 `for t in range(iters)`로 돈다. trace 시 자연히 정적 언롤링된다. `torch.jit.script`, 동적 루프, 데이터 의존 분기는 쓰지 않는다.
- K = 3, 5는 학습 집합에 없다. **held-out 깊이**로 평가에만 쓴다(§6-D).

### 4.3 FiLM 깊이 조건화 — `model.film: relative | absolute | none`
- `relative`: 입력 스칼라 s = t / K (t = 0..K−1).
- `absolute`: 입력 스칼라 s = t / K_max, K_max = max(`model.iters`). K와 무관하게 t에만 의존한다.
- `none`: 조건화 없음(ablation 기준선).
- 공통: 사인파 임베딩 `φ(s)` → 2층 MLP(코어당 1개, 반복 간 공유) → `[γ_t; β_t] ∈ R^{2C}`.
  적용 위치는 코어 내 각 ConvNeXt 블록의 정규화 직후 `h ← γ_t ⊙ h + β_t`. 한 반복 안의 3개 블록은 같은 (γ_t, β_t)를 쓴다.
- MLP 마지막 층은 γ=1, β=0에서 시작하도록 초기화(`γ = 1 + Δγ`, 마지막 층 zero init).
- 세 모드는 FiLM 클래스 하나의 `mode`로 구분하고 나머지 코드 경로는 동일해야 한다. export 시 K가 고정이면 γ, β는 상수로 접힌다.

### 4.4 NPU 친화 제약
- NCHW를 유지한다. ConvNeXt의 pointwise는 `permute + Linear`가 아니라 `1×1 Conv2d`로 구현해 Transpose를 만들지 않는다.
- 정규화는 `model.norm: layernorm2d | groupnorm1 | batchnorm`으로 교체 가능하게 만든다. 기본값은 M0b(§8) 결과로 정한다. BatchNorm은 공유 코어에서 반복별 통계가 필요하다는 점에 유의한다.
- 활성화(`gelu | relu | hardswish`)와 GRN 사용 여부도 config로 둔다. 같은 이유.
- NPU용 export 그래프에는 int64 입력, 동적 축, 데이터 의존 shape 연산이 없어야 한다.
- M0b에서 미지원 연산자가 나오면 대체안을 제시하고 사용자 결정을 기다린다. 임의로 구조를 바꾸지 않는다.

## 5. 학습

- 매 minibatch마다 M₁, M₂를 `model.iters`에서 **독립 균등** 샘플링, SNR은 `[min_SNRdB, max_SNRdB]` 균등, spp는 `[min_spp, spp]` 균등. 손실은 MSE.
- 단일 깊이 학습(ablation B)은 `train.iters`를 원소 하나짜리 집합으로 주는 것으로 처리한다. 별도 코드 경로를 만들지 않는다. `train.iters` 기본값은 `model.iters`.
- `model.share_core: true | false`. `false`면 같은 깊이를 독립 가중치로 쌓는다(ablation C). 단일 깊이 학습과 함께만 쓴다.
- validation은 매 epoch 대표 조합((1,1), (2,2), (4,4), (6,6))을 고정 SNR·spp로 돌려 조합별 PSNR을 TensorBoard에 남긴다. best 기준은 이들의 평균 loss.
- AMP와 `channels_last`는 옵션. 켜고 끈 결과가 다르지 않은지 스모크 런으로 확인한 뒤에만 기본값으로 한다.
- 데이터셋 경로, 크롭 크기, batch size, epoch 수, lr은 전부 config. 코드에 박지 않는다.
- 로그는 표준 `logging`으로 콘솔과 `logging.log`에 남긴다. `print`를 쓰지 않는다.

## 6. 실험 매트릭스

RTX 5090 1장이므로 아래 순서대로 돌린다. 실험명은 config 파일명과 일치시킨다.

| ID | config | 내용 | 답하려는 질문 |
|----|--------|------|---------------|
| A1 | `joint_rel` | 공유, joint, FiLM relative | 본 모델 |
| A2 | `joint_abs` | 공유, joint, FiLM absolute | t/K vs t |
| A3 | `joint_none` | 공유, joint, FiLM 없음 | 조건화의 순수 효과 |
| B  | `fixed_k{1,2,4,6}_rel` | 공유, 단일 K 학습 | joint 학습의 손실 |
| C  | `unshared_k{1,2,4,6}` | 비공유, 단일 깊이 | 공유의 손실(파라미터 대비) |
| D  | (학습 없음) | A1~A3를 K = 3, 5에서 평가 | 미학습 깊이 일반화 |

`evaluate.py`가 내는 것(`results/sim/<exp>.json`):
- 16개 (M₁, M₂) 조합 × SNR 스윕 × spp 스윕의 PSNR, MS-SSIM. Kodak 전체 이미지(8의 배수로 패딩 후 크롭), 노이즈 seed 고정, 반복 평균.
- 조합별 파라미터 수(공유 가중치는 1회)와 GFLOPs(실행된 반복은 전부), 인코더/디코더 분리 표기. 기준 입력 해상도는 config.
- 미러 쌍((1,2)↔(2,1) 등)의 인코더/디코더 비대칭 비교표.

평가 코드 안에서 그림을 그리지 않는다. 그림은 `tools/plots.py`가 `results/`를 읽어 그린다.

## 7. NPU 파이프라인 (Rubik Pi 3 / QCS6490)

보드에서는 **ONNX Runtime의 QNN Execution Provider**로 실행한다. 역할은 이렇게 나뉜다.

- 서버: 학습, ONNX export, **QDQ 양자화**(캘리브레이션 데이터와 연산이 서버에 있으므로). NPU 전용 형식으로의 변환 코드는 서버에 두지 않는다.
- 보드: QDQ ONNX를 받아 세션을 만들 때 EP가 HTP(NPU)용으로 컴파일한다. 컴파일 결과(컨텍스트 바이너리)는 보드에 캐시해 재로드 시간과 크기를 잰다.

Qualcomm SDK의 변환 도구는 쓰지 않으므로 SDK 다운로드와 Qualcomm 계정은 필요 없다.

| 단계 | 환경 | 진입점 | 내용 |
|---|---|---|---|
| 1 | train | `export_onnx.py --static H W` | 체크포인트(또는 `--random-init`) → 고정 K 인코더·디코더 ONNX. 인코더 4개, 디코더 4개 |
| 2 | train | `tools/make_calib.py` | 캘리브레이션 입력과 벤치 입력 생성. 디코더용은 인코더 출력 + AWGN으로 만든다 |
| 3 | train | `tools/quantize.py` | `onnxruntime.quantization`의 QNN용 전처리·QDQ 설정으로 양자화. 활성화 8-bit과 16-bit 두 벌 |
| 4 | train | `tools/make_bundle.py` | QDQ ONNX + 입력 + `bench/`를 묶어 scp로 보드에 전송 |
| 5 | board | `bench/run_all.py` (`BOARD_SSH` 또는 `tools/relay.sh`) | QNN EP(HTP)와 CPU EP로 실행, 출력과 측정값 저장 |
| 6 | train | `tools/collect_npu.py` | 결과 회수 → `results/npu/<exp>/<날짜>/` |

- static 모드 그래프의 입력은 `image(1,3,H,W)` / `received(1,C,H/8,W/8)` + `snr(1,1)` float뿐이다. 기존 동적 축 모드(PC 추론용)는 그대로 유지한다.
- 모든 export에 대해 ONNX Runtime 출력과 PyTorch 출력의 일치를 테스트한다.
- QDQ 모델의 PSNR을 서버의 CPU EP로 먼저 측정한다. 보드 결과와 비교할 기준이다.
- 보드에서 PSNR 체인(인코더 → 고정 시드 AWGN → 디코더)은 numpy로 직접 돌린다. 원시 출력도 npy로 저장해 서버에서 재계산할 수 있게 한다.
- 보드에서 실행은 `uv run`으로, 사용자의 `recurrent-npu-jscc` 프로젝트 환경을 쓴다.

보드 측정 항목:
1. 인코더·디코더 × K별 추론 지연: 워밍업 후 N회, 중앙값/p95. QNN EP(HTP)와 CPU EP 둘 다.
2. 세션 생성(컴파일 포함) 시간, 컨텍스트 캐시를 쓴 재로드 시간, 프로세스 피크 RSS.
3. 크기: K별 QDQ ONNX 크기, EP가 덤프한 컨텍스트 바이너리 크기.
4. 온디바이스 출력의 PSNR과 FP32 대비 하락폭(K별).
5. 측정 중 SoC 온도(`/sys/class/thermal`), 스로틀링 여부.
6. **노드가 실제로 QNN EP에 배정됐는지.** CPU fallback 노드 목록을 남긴다. CPU fallback을 금지하는 세션 옵션으로도 한 번 돌려 본다.
7. 전력은 사용자가 외부 전력계 값을 줄 때만 표에 넣는다.

반드시 답해야 하는 질문:
- **언롤링 후에도 가중치가 실제로 공유되는가.** ONNX initializer 중복 여부(QDQ 변환 후 포함), 컨텍스트 바이너리 크기의 K 의존성, 여러 K 모델이 컨텍스트·가중치를 공유할 수 있는지. 안 되면 안 된다고 수치와 함께 보고한다.
- **공유 가중치의 반복 적용에서 양자화 오차가 K에 따라 누적되는가.**
- **인코더 출력(전송 심볼)의 양자화 비트폭이 PSNR에 주는 영향.** 활성화 8-bit과 16-bit을 비교한다.

검증 필요(작성 시점의 기억 기반. onnxruntime 1.23 문서와 실제 실행으로 확인 전에는 사실로 취급하지 않는다):
- provider 목록에 `QNNExecutionProvider`가 있다는 것은 빌드에 포함됐다는 뜻일 뿐, HTP 백엔드가 실제로 뜬다는 보장이 아니다. 휠 안에 HTP 백엔드 라이브러리(`libQnnHtp.so`)와 DSP 쪽 skel 라이브러리가 들어 있는지, 세션이 그것으로 만들어지는지 M0a에서 확인한다.
- QNN EP provider option과 세션 옵션의 정확한 이름: 백엔드 지정, 성능 모드, 프로파일링 레벨, 컨텍스트 캐시 덤프·임베드, 컨텍스트 공유, CPU fallback 금지.
- QCS6490의 HTP는 양자화 모델만 지원하고 FP16은 지원하지 않을 가능성이 높다. 그렇다면 QAT가 필요할 수 있다.
- 16-bit 활성화 QDQ가 이 EP·칩 조합에서 지원되는지.
- QAIRT SDK가 필요해지는 경우: 휠에 백엔드 라이브러리가 없을 때, 또는 연산자별 프로파일이 EP만으로 안 나올 때. 이때는 Qualcomm 계정으로 SDK를 받아야 하므로 **멈추고 사용자에게 요청**한다.

## 8. 마일스톤

통과 조건을 만족하기 전에는 다음으로 넘어가지 않는다. 단, M0가 [USER] 작업을 기다리는 동안에는 M1을 먼저 진행한다.

| 단계 | 내용 | [USER] 선행 작업 | 통과 조건 |
|---|---|---|---|
| M0a | conv 몇 층짜리 더미 QDQ 모델로 툴체인 검증 | 터널 열기 또는 `relay.sh` 실행 | 더미 모델의 노드가 보드 HTP에 배정되어 실행된다 |
| M0b | 랜덤 초기화 RecurrentJSCC(K=1, 6)를 보드에 올린다 | 없음 | CPU fallback 목록과 지연이 나오고 norm/활성화/GRN 기본값이 확정된다 |
| M1 | §2 리팩터링, §4 구현, 테스트 | 없음 | §11 통과, `baseline.yaml` 회귀 통과 |
| M2 | 학습(§6 순서) | 학습 승인 | 각 실험의 best 체크포인트 |
| M3 | 시뮬레이션 평가 | 없음 | `results/sim/` |
| M4 | 양자화, NPU 실측 | 전력계(선택) | `results/npu/` |
| M5 | 그림/표 | 없음 | `tools/plots.py` 산출물 |

M0는 오래 학습한 뒤 연산자 미지원을 발견하는 일을 막기 위한 단계다. M0a는 툴체인 문제와 모델 문제를 분리하기 위한 것이고, 더미 모델은 통과 후 삭제한다.

## 9. 역할 분담

[CC] = GPU 서버에서 Claude Code가 수행. [USER] = 사용자가 직접 수행.

GPU 서버:
- [CC] 코드 전부(§2~§7), 테스트, lint, 스모크 런, 학습 실행·모니터링(승인 후), 평가, ONNX export, 캘리브레이션 데이터, QDQ 양자화, 서버에서의 양자화 정확도, 번들 생성·전송, 결과 회수·파싱, 그림/표, 문서 조사.
- [USER] `sudo`가 필요한 설치.
- [USER] 장시간 학습 시작 승인, git 원격·인증 설정.
- [USER] 연구 결정: 하이퍼파라미터, SNR/spp 범위, 구조 변경 여부, 결과 해석.

Rubik Pi 3:
- [CC] `bench/`와 `tools/relay.sh` 작성. 터널이 열려 있으면 번들 전송, `bench/run_all.py` 실행, 결과 회수까지 수행.
- [USER] 로컬 PC에서 역방향 터널을 열어 두거나, `tools/relay.sh`를 실행. 보드의 `recurrent-npu-jscc` 프로젝트 경로 전달.
- [USER] 보드 패키지 추가·교체, `sudo`나 물리 조작이 필요한 일, 재부팅.
- [USER] 방열(히트싱크·팬) 확보, 측정 중 보드에 다른 부하가 없도록 유지.
- [USER] 전력 측정: USB-C 전력계로 수동 계측하고 `results/npu/.../power_manual.csv`에 기록.
- [USER] USRP 테스트베드 실험.
- [USER] (§7의 조건에 해당할 때만) QAIRT SDK 다운로드.

## 10. 미결 사항(기본값으로 진행, 사용자 답이 오면 수정)

- 비공유 stage의 블록 수 N₁. 기본값 3.
- FiLM 적용 위치(§4.3)가 이전 구현과 같은지.
- 크롭 크기.
- NPU 고정 입력 해상도. 기본값 256×256.

확정된 사항:
- 학습 데이터는 DIV2K, 위치는 `~/datasets/DIV2K`(train과 val이 모두 있음. 하위 폴더 이름은 직접 확인한다). 학습 중 validation은 DIV2K val, 최종 평가는 Kodak.
- 보드 OS는 Ubuntu 24.04 LTS, 주소는 `ubuntu@192.168.1.9`(사용자 로컬 네트워크).
- 채널 폭: stage 1(H/4)은 96, stage 2(H/8, 코어)는 192. `model.dim: 96`, stage 2는 그 2배.
- 최대 spp는 0.5(H/8 격자에서 실수 심볼 채널 64개). `min_spp`는 기존 0.1 유지.

## 11. 테스트(최소)

- 모든 (M₁, M₂) 조합에서 forward/backward와 출력 shape, 8의 배수가 아닌 해상도 처리.
- 파라미터 수가 K와 무관하다(`share_core: true`).
- GFLOPs(K=2) − GFLOPs(K=1) = 코어 1회분.
- `film` 세 모드가 zero init 직후 같은 출력을 낸다.
- `relative`와 `absolute`가 K = K_max에서 같은 출력을 낸다.
- 체크포인트 저장 → 로드 → resume 시 `epoch`, `best_loss`, optimizer 상태가 이어진다.
- ONNX(동적/정적) vs PyTorch 출력 일치.
- `baseline.yaml`이 리팩터링 후에도 학습 1 step과 export를 통과한다.