#!/bin/bash
# 전체 파이프라인 (Stage 1~4) 연속 실행
# Usage: ./run_all.sh [MODEL_NAME]
# Example: ./run_all.sh "meta-llama/Llama-3.2-3B"

set -e

MODEL_NAME=${1:?"Usage: ./run_all.sh MODEL_NAME"}

for STAGE in 1 2 3 4; do
    echo ""
    echo "########## Stage ${STAGE} / 4 시작 ##########"
    ./run_pipeline.sh "$MODEL_NAME" "$STAGE"
    echo "########## Stage ${STAGE} / 4 완료 ##########"
done

echo ""
echo "전체 파이프라인(Stage 1~4) 완료: ${MODEL_NAME}"
