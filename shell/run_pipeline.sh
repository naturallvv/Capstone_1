#!/bin/bash
# 단일 스테이지 파이프라인: 한 번 실행에 1스테이지만 수행
# Stage 1: SFT + 데이터 준비 + KRSL + eval(4벤치마크) → best epoch 선택
# Stage 2+: 이전 스테이지 eval 결과에서 best epoch 자동 읽기 → 데이터 준비 + KRSL + eval
#
# Usage: ./run_pipeline.sh [MODEL_NAME] [STAGE]
# Example:
#   ./run_pipeline.sh "meta-llama/Llama-3.2-1B" 1   # SFT + Stage 1
#   ./run_pipeline.sh "meta-llama/Llama-3.2-1B" 2   # Stage 2 (이전 best epoch 자동 읽기)
#   ./run_pipeline.sh "meta-llama/Llama-3.2-1B" 3   # Stage 3
#   ./run_pipeline.sh "meta-llama/Llama-3.2-1B" 4   # Stage 4

set -e  # 에러 발생 시 즉시 종료

cd '../'  # 프로젝트 루트로 이동 (run_sft.sh, run_krsl.sh와 동일)

# ==================== 인자 처리 ====================
if [ $# -lt 2 ]; then
    echo "Usage: ./run_pipeline.sh [MODEL_NAME] [STAGE]"
    echo "Example:"
    echo "  ./run_pipeline.sh \"meta-llama/Llama-3.2-1B\" 1   # SFT + Stage 1"
    echo "  ./run_pipeline.sh \"meta-llama/Llama-3.2-1B\" 2   # Stage 2 (이전 best epoch 자동 읽기)"
    exit 1
fi

MODEL_NAME="$1"
STAGE="$2"
MODEL_SHORT="${MODEL_NAME##*/}"

# SFT는 무조건 epoch 15 사용, KRSL은 20 에폭 학습 후 eval로 best epoch 자동 선택
SFT_LOAD_EPOCH=15

# 파이프라인 중간 파일 저장 베이스 경로
PIPELINE_DIR="./dataset/bbh/bbh_all_data/pipeline/${MODEL_SHORT}"

# 교차 라벨링용 입력 데이터 (CoT 프롬프트 적용 + response 초기화 완료)
# Student A용: B모델이 A데이터를 라벨링
BEFORE_LABEL_A="./dataset/bbh/bbh_all_data/before_pseudo_labeling_A(train_B,labeling_A).json"
# Student B용: A모델이 B데이터를 라벨링
BEFORE_LABEL_B="./dataset/bbh/bbh_all_data/before_pseudo_labeling_B(train_A,labeling_B).json"

# 원본 학생별 학습 데이터 (final_check 참조용)
ORIG_DATA_A="./dataset/bbh/bbh_all_data/all_task_train_right_wronghint_answer_A.json"
ORIG_DATA_B="./dataset/bbh/bbh_all_data/all_task_train_right_wronghint_answer_B.json"

# 프롬프트 디렉토리
AHP_DIR="./dataset/bbh/cot-ahp"
CCP_DIR="./dataset/bbh/cot-ccp"

# 데이터 유틸리티 경로
DATA_UTILS="./dataset/bbh/bbh_all_data/data_utils.py"
LABELING="./dataset/bbh/bbh_all_data/labeling.py"
PAIR_SCRIPT="./dataset/bbh/bbh_all_data/pair_student_inference.py"
FINAL_CHECK="./dataset/bbh/bbh_all_data/final_check.py"
EDIT_DIS="./edit_dis_precal.py"

# 평가 GPU/포트 설정
EVAL_GPU=0
EVAL_PORT=56543

# 평가 벤치마크 목록
BENCHMARKS="bbh_eval_dataset bb_eval_dataset agieval_eval_dataset arcc_eval_dataset"

# ==================== 유틸: 4개 벤치마크 평균으로 best epoch 선택 ====================
find_best_epoch() {
    local STAGE_DIR="$1"
    python3 -c "
import json, glob, os

stage_dir = '${STAGE_DIR}'
benchmarks = ['bbh_eval_dataset', 'bb_eval_dataset', 'agieval_eval_dataset', 'arcc_eval_dataset']

# 에폭별 정확도 수집: {epoch: [acc1, acc2, acc3, acc4]}
epoch_accs = {}
for bm in benchmarks:
    result_dir = os.path.join(stage_dir, 'results', bm)
    for f in glob.glob(os.path.join(result_dir, 'epoch-*-eval_result.json')):
        epoch = int(os.path.basename(f).split('epoch-')[1].split('-eval_result')[0])
        with open(f) as fp:
            data = json.load(fp)
        acc = data[0]['total_accuracy']
        epoch_accs.setdefault(epoch, []).append(acc)

# 4개 벤치마크 전부 결과가 있는 에폭만 대상
best_epoch, best_avg = -1, -1.0
for epoch, accs in epoch_accs.items():
    if len(accs) == len(benchmarks):
        avg = sum(accs) / len(accs)
        if avg > best_avg:
            best_avg = avg
            best_epoch = epoch

print(best_epoch)
"
}

echo "============================================"
echo "Pipeline: ${MODEL_SHORT}, Stage ${STAGE}"
echo "  SFT load epoch:  ${SFT_LOAD_EPOCH} (고정)"
echo "  KRSL epochs:     20 (학습 후 eval로 best 자동 선택)"
echo "  Pipeline dir:    ${PIPELINE_DIR}"
echo "============================================"

# ==================== Stage 1이면 SFT 먼저 실행 ====================
if [ "$STAGE" -eq 1 ]; then
    echo ""
    echo "============================================"
    echo "[SFT] 학습 (Stage 1 전용)"
    echo "============================================"

    echo "  SFT Student A..."
    cd shell && ./run_sft.sh "$MODEL_NAME" A && cd ../

    echo "  SFT Student B..."
    cd shell && ./run_sft.sh "$MODEL_NAME" B && cd ../
fi

# ==================== 모델 경로 결정 ====================
if [ "$STAGE" -eq 1 ]; then
    MODEL_A="../slm/hf/${MODEL_SHORT}/A/sft/epoch-${SFT_LOAD_EPOCH}"
    MODEL_B="../slm/hf/${MODEL_SHORT}/B/sft/epoch-${SFT_LOAD_EPOCH}"
    LOAD_EPOCH_A=$SFT_LOAD_EPOCH
    LOAD_EPOCH_B=$SFT_LOAD_EPOCH
else
    PREV_STAGE=$((STAGE - 1))
    PREV_DIR_A="../slm/hf/${MODEL_SHORT}/A/stage${PREV_STAGE}"
    PREV_DIR_B="../slm/hf/${MODEL_SHORT}/B/stage${PREV_STAGE}"

    # 이전 스테이지 eval 결과에서 best epoch 자동 읽기
    LOAD_EPOCH_A=$(find_best_epoch "$PREV_DIR_A")
    LOAD_EPOCH_B=$(find_best_epoch "$PREV_DIR_B")

    if [ "$LOAD_EPOCH_A" -eq -1 ] || [ "$LOAD_EPOCH_B" -eq -1 ]; then
        echo "ERROR: 이전 스테이지(stage${PREV_STAGE}) eval 결과를 찾을 수 없습니다."
        echo "  Stage ${PREV_STAGE}를 먼저 실행해주세요."
        exit 1
    fi

    MODEL_A="${PREV_DIR_A}/epoch-${LOAD_EPOCH_A}"
    MODEL_B="${PREV_DIR_B}/epoch-${LOAD_EPOCH_B}"

    echo ""
    echo "  이전 스테이지(stage${PREV_STAGE}) best epoch: A=${LOAD_EPOCH_A}, B=${LOAD_EPOCH_B}"
fi

# ==================== 데이터 준비 (교차 라벨링) ====================
for STUDENT in A B; do
    echo ""
    echo "---- Stage ${STAGE}, Student ${STUDENT} 데이터 준비 ----"

    WORK="${PIPELINE_DIR}/stage${STAGE}/${STUDENT}"
    mkdir -p "$WORK"

    # 교차 라벨링: 각 모델이 상대 데이터(안 본 데이터)를 라벨링
    # Model A (SFT: answer_A) → 추론: before_pseudo_labeling_A(train_B,labeling_A)
    # Model B (SFT: answer_B) → 추론: before_pseudo_labeling_B(train_A,labeling_B)
    if [ "$STUDENT" = "A" ]; then
        CROSS_MODEL_PATH="$MODEL_A"
        BEFORE_LABEL="$BEFORE_LABEL_A"
        ORIG_DATA="$ORIG_DATA_B"
    else
        CROSS_MODEL_PATH="$MODEL_B"
        BEFORE_LABEL="$BEFORE_LABEL_B"
        ORIG_DATA="$ORIG_DATA_A"
    fi

    # 2-1. 교차 추론 (기존 before_pseudo_labeling 데이터 사용)
    echo "  [2-1] 교차 추론 (모델: ${CROSS_MODEL_PATH})..."
    python "$LABELING" \
        --model-name "$CROSS_MODEL_PATH" \
        --input-json "$BEFORE_LABEL" \
        --output-json "${WORK}/after_labeling.json"

    # 2-2. 정오답 분류 + CCP/AHP 프롬프트 적용
    echo "  [2-2] AHP/CCP 프롬프트 적용..."
    python "$DATA_UTILS" ahp-ccp \
        --input-json "${WORK}/after_labeling.json" \
        --output-json "${WORK}/after_ahp_ccp.json" \
        --ahp-dir "$AHP_DIR" \
        --ccp-dir "$CCP_DIR"

    # 2-3. CCP/AHP 프롬프트로 재추론
    echo "  [2-3] CCP/AHP 재추론..."
    python "$LABELING" \
        --model-name "$CROSS_MODEL_PATH" \
        --input-json "${WORK}/after_ahp_ccp.json" \
        --output-json "${WORK}/after_relabeling.json"

    # 2-4. 듀얼 데이터셋 생성
    echo "  [2-4] 듀얼 데이터셋 생성..."
    python "$PAIR_SCRIPT" \
        --first-json "${WORK}/after_labeling.json" \
        --second-json "${WORK}/after_relabeling.json" \
        --output-json "${WORK}/paired.json"

    # 2-5. 필터링
    echo "  [2-5] 필터링..."
    python "$FINAL_CHECK" \
        --input-json "${WORK}/paired.json" \
        --reference-json "$ORIG_DATA" \
        --output-json "${WORK}/filtered.json"

    # 2-6. 편집 거리 계산
    echo "  [2-6] 편집 거리 계산..."
    python "$EDIT_DIS" \
        --data-file "${WORK}/filtered.json" \
        --model-name "$MODEL_NAME"

    echo "  ---- Student ${STUDENT} 데이터 준비 완료 ----"
done

# ==================== KRSL 학습 ====================
WORK_A="${PIPELINE_DIR}/stage${STAGE}/A"
WORK_B="${PIPELINE_DIR}/stage${STAGE}/B"

echo ""
echo "[KRSL] Student A 학습 (epoch-${LOAD_EPOCH_A} 로드)..."
cd shell && ./run_krsl.sh "$MODEL_NAME" A "$STAGE" "$LOAD_EPOCH_A" \
    "${WORK_A}/filtered.json" "${WORK_A}/filtered_precal.pkl" && cd ../

echo "[KRSL] Student B 학습 (epoch-${LOAD_EPOCH_B} 로드)..."
cd shell && ./run_krsl.sh "$MODEL_NAME" B "$STAGE" "$LOAD_EPOCH_B" \
    "${WORK_B}/filtered.json" "${WORK_B}/filtered_precal.pkl" && cd ../

# ==================== eval (4개 벤치마크 × A,B 전 에폭) ====================
STAGE_DIR_A="../slm/hf/${MODEL_SHORT}/A/stage${STAGE}"
STAGE_DIR_B="../slm/hf/${MODEL_SHORT}/B/stage${STAGE}"

echo ""
echo "[EVAL] 4개 벤치마크 평가 시작 (A,B 각 20 에폭 전부)..."

#벤치마크 1에폭만 평가중 코드 수정 해야됌
for BENCHMARK in $BENCHMARKS; do
    echo "  Student A - ${BENCHMARK}..."
    cd shell && ./run_eval.sh "${STAGE_DIR_A}/epoch-1" "$STAGE_DIR_A" "$BENCHMARK" $EVAL_GPU $EVAL_PORT && cd ../

    echo "  Student B - ${BENCHMARK}..."
    cd shell && ./run_eval.sh "${STAGE_DIR_B}/epoch-1" "$STAGE_DIR_B" "$BENCHMARK" $EVAL_GPU $EVAL_PORT && cd ../
done

# ==================== best epoch 선택 ====================
BEST_EPOCH_A=$(find_best_epoch "$STAGE_DIR_A")
BEST_EPOCH_B=$(find_best_epoch "$STAGE_DIR_B")

echo ""
echo "============================================"
echo "Stage ${STAGE} 완료: ${MODEL_SHORT}"
echo "  Best epoch (4벤치마크 평균): A=epoch-${BEST_EPOCH_A}, B=epoch-${BEST_EPOCH_B}"
echo "  모델 A: ${STAGE_DIR_A}/epoch-${BEST_EPOCH_A}"
echo "  모델 B: ${STAGE_DIR_B}/epoch-${BEST_EPOCH_B}"
echo "  다음 스테이지: ./run_pipeline.sh \"${MODEL_NAME}\" $((STAGE + 1))"
echo "============================================"
