# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Policy (Mandatory)

- 모든 응답과 설명은 한국어로만 작성한다.
- 코드에 대한 설명, 계획, 분석, 확인 메시지는 모두 한국어로 작성한다.
- 함수명, 변수명, 클래스명, 라이브러리명 등 코드 식별자는 영어를 그대로 사용한다.
- 코드 주석은 반드시 한국어로 작성한다.
- 영어 문장 형태의 설명은 사용하지 않는다.
- 사용자가 명시적으로 요청한 경우에만 영어 설명을 허용한다.

## 프로젝트 개요

CasCoD (Cascading Decomposed CoTs Distillation) — EMNLP 2024 본 학회 논문의 공식 구현체. 대형 LLM의 Chain-of-Thought(CoT) 추론 능력을 소형 학생 모델에 증류하는 프레임워크. Meta의 llama-recipes 기반으로 구축되었으며, 최소 편집 거리를 활용한 EDIT 증류 방법론도 함께 포함.

## 주요 명령어

### 학습

```bash
# CasCoD 및 기타 증류 방법
cd shell && ./run_distilled_cot.sh

# EDIT 방법 (3단계)
python edit_dis_precal_A.py          # 0단계: 편집 거리 가중치 사전 계산 → .pkl
cd shell && ./run_edit_step1_A.sh    # 1단계: 기본 CoT에 대한 SFT
cd shell && ./run_edit_step2_krsl.sh # 2단계: 핵심 추론 단계 학습 (KRSL)
```

모든 학습은 `torchrun` + FSDP(3~4 GPU)를 사용. 체크포인트는 `../slm/hf/`에 저장.

### 평가

```bash
cd shell && ./eval_distilled_cot.sh  # 파인튜닝된 모델 평가
cd shell && ./eval_vanilla.sh        # 기본 모델 평가 (0-shot, few-shot)
```

평가 스크립트에서 변경할 주요 파라미터: `saved_model_dir`, `train_dataset`, `test_dataset`.

### PEFT 변환

```bash
python batch_convert_peft.py  # PEFT 어댑터 .bin → HuggingFace 형식 변환
```

## 아키텍처

### 진입점

- **`finetuning.py`** — 학습 오케스트레이터. config/모델/데이터셋 로드, FSDP 설정, 방법별 학습 Tool로 디스패치.
- **`evaluation.py`** — 추론 및 평가. vLLM 배치 처리와 self-consistency 투표 지원.

### 학습 메서드 디스패치

`finetuning.py` → `src/utils/train_utils.py:train()`에서 데이터셋 config에 따라 방법별 Tool 클래스를 선택:

| Tool 클래스 | 방법론 | 데이터셋 Config |
|---|---|---|
| `LLMCMTTool` | CasCoD (본 논문 핵심 기여) | `bbh_llmcmt_dataset` |
| `MyKRSLTool` | EDIT 2단계 (선호도 학습) | `bbh_krsl_dataset` |
| `LLMSCOTTTool` | SCOTT 베이스라인 | `bbh_llmscott_dataset` |
| `LLMMTTool` | 멀티태스크 변형 (MT-CoT, MT-Ra, MT-Re) | `bbh_llmmt*_dataset` |
| `LLMSTTool` | 표준 CoT | `bbh_llmst_dataset` |
| `LLMWeightSTTool` | 가중치 CoT | `bbh_llmweightst_dataset` |

각 Tool 클래스는 `src/utils/method_utils.py`에서 자체 `forward()` / 손실 함수를 정의.

### Config 시스템

`src/configs/`에 Dataclass 기반 설정:
- **`training.py`** — 하이퍼파라미터, 손실 가중치 (alpha, gama, krsl_alpha, krsl_beta), PEFT/FSDP 토글
- **`datasets.py`** — 데이터셋 경로, 최대 길이, KRSL 전용 파라미터. 각 증류 방법마다 고유 config 보유.
- **`peft.py`** — LoRA 기본값: r=64, alpha=32, 대상 모듈 `["q_proj", "v_proj"]`
- **`fsdp.py`** — 혼합 정밀도, 샤딩 전략, 활성화 체크포인팅

### 데이터셋 클래스 (`src/datasets/bbh_dataset.py`)

각 학습 방법마다 전용 Dataset 클래스가 데이터를 다르게 포맷:
- `InstructionDataset` — 정답 SFT
- `LLMSTDataset` — 교사 모델 CoT 응답 SFT
- `LLMCMTDataset` — CasCoD 계단식 멀티태스크 (CoT를 하위 단계로 분해)
- `KRSLDataset` — 편집 거리 기반 가중치가 적용된 선호도 쌍 (chosen vs rejected)
- `LLMMTCoTDataset` / `LLMMTRaDataset` / `LLMMTReDataset` — 멀티태스크 변형

`src/datasets/utils.py`의 `custom_tokenize()`: 프롬프트 토큰을 마스킹(label=-100)하여 응답 토큰에만 손실 계산.

### 데이터 흐름

```
JSON 데이터 → Dataset 클래스 → custom_tokenize() → DataLoader (DistributedSampler)
    → PEFT 래핑 모델 (Llama-2-7B + LoRA) → FSDP 샤딩
    → 방법별 Tool.forward() → 손실 → 옵티마이저 스텝
    → 체크포인트 (HF 형식)
```

### 평가 흐름

`evaluation.py` → `src/utils/inference_utils.py:eval_inference()` — 체크포인트 로드, 토큰 생성(선택적 vLLM), "the answer is" 패턴으로 답 추출, 태스크별/매크로 정확도 계산.

## 핵심 규칙

- **기본 모델:** `meta-llama/Llama-2-7b-hf`
- **기본 학습 설정:** LR 2e-4, 배치 16/GPU, gradient accumulation 4, 10~15 에폭, AdamW, BF16 혼합 정밀도
- **KRSL 손실 파라미터:** krsl_alpha=1.0, krsl_beta=-0.025 (음수 = 역방향 KL)
- **답 추출:** 생성 텍스트에서 "the answer is" 패턴 탐색
- **데이터셋 네이밍 규칙:** 학습용 `bbh_llm{method}_dataset`, 평가용 `{benchmark}_eval_dataset`
- **로그:** `./log/{dataset_name}/{model_name}/`에 저장
- **Shell 스크립트**에서 `CUDA_VISIBLE_DEVICES` 설정 후 `torchrun` 직접 호출 — GPU 수에 맞게 조정 필요

## 데이터 디렉토리 구조

- `dataset/bbh/bbh_all_data/` — 주요 학습 데이터 (JSON + EDIT용 사전 계산 .pkl)
- `dataset/bbh/cot-prompts/`, `cot-ahp/`, `cot-ccp/` — CoT 프롬프트 템플릿
- `dataset/{bb,agieval,arc-c,arc-e}/merged_data/` — 평가 벤치마크
- `call_llm/` — OpenAI API(GPT-4)를 통한 교사 LLM 응답 생성 스크립트
