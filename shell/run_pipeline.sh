#!/bin/bash
# Master pipeline script: SFT + 4-stage KRSL for both students A and B
# Usage: ./run_pipeline.sh [MODEL_NAME] [SFT_EPOCHS] [KRSL_EPOCHS]
# Example: ./run_pipeline.sh "meta-llama/Llama-3.2-1B" 15 10

set -e  # Exit on error

MODEL_NAME=${1:-"meta-llama/Llama-3.2-1B"}
SFT_EPOCHS=${2:-15}        # Best epoch to load from SFT
KRSL_EPOCHS=${3:-10}       # Best epoch to load from each KRSL stage
MODEL_SHORT="${MODEL_NAME##*/}"

echo "============================================"
echo "Pipeline: ${MODEL_SHORT}"
echo "  SFT best epoch: ${SFT_EPOCHS}"
echo "  KRSL best epoch: ${KRSL_EPOCHS}"
echo "============================================"

# ==================== Step 1: SFT (one-time) ====================
echo "[Step 1/5] Running SFT for Student A..."
./run_sft.sh "$MODEL_NAME" A

echo "[Step 1/5] Running SFT for Student B..."
./run_sft.sh "$MODEL_NAME" B

# ==================== Step 2~5: KRSL Stages 1~4 ====================
for STAGE in 1 2 3 4; do
    echo "============================================"
    echo "[Step $((STAGE + 1))/5] Stage ${STAGE}"
    echo "============================================"

    # Determine previous epoch to load from
    if [ "$STAGE" -eq 1 ]; then
        PREV_EPOCH=$SFT_EPOCHS
    else
        PREV_EPOCH=$KRSL_EPOCHS
    fi

    # TODO: Add pseudo labeling step here
    # e.g., run inference with the previous stage model to generate pseudo labels

    # TODO: Add edit distance calculation step here
    # e.g., python edit_dis_precal_A.py / edit_dis_precal_B.py

    # KRSL training
    echo "  Running KRSL Stage ${STAGE} for Student A (loading epoch-${PREV_EPOCH})..."
    ./run_krsl.sh "$MODEL_NAME" A $STAGE $PREV_EPOCH

    echo "  Running KRSL Stage ${STAGE} for Student B (loading epoch-${PREV_EPOCH})..."
    ./run_krsl.sh "$MODEL_NAME" B $STAGE $PREV_EPOCH
done

echo "============================================"
echo "Pipeline complete for ${MODEL_SHORT}"
echo "Output: ../slm/hf/${MODEL_SHORT}/{A,B}/{sft,stage1,stage2,stage3,stage4}/"
echo "============================================"
