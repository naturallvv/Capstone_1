#!/bin/bash
# 전체 파이프라인 (Stage 1~4) 연속 실행
# 완료된 스테이지는 자동 스킵 (eval 결과 기반)
# Usage: ./run_all.sh [MODEL_NAME] [START_STAGE]
# Example:
#   ./run_all.sh "meta-llama/Llama-3.2-3B"       # Stage 1부터
#   ./run_all.sh "meta-llama/Llama-3.2-3B" 3     # Stage 3부터

# ./run_all.sh "meta-llama/Llama-3.2-1B" 완
# ./run_all.sh "meta-llama/Llama-3.2-3B" 완
# ./run_all.sh "meta-llama/Llama-3.1-8B" 진행 중(11번 서버)
# ./run_all.sh "meta-llama/Llama-2-13b-hf" 진행중(13번 서버)
# ./run_all.sh "TinyLlama/TinyLlama_v1.1"
# ./run_all.sh "mistralai/Mistral-7B-v0.1"
# ./run_all.sh "Qwen/Qwen3-0.6B"
# ./run_all.sh "Qwen/Qwen3-1.7B"
# ./run_all.sh "Qwen/Qwen3-4B"
# ./run_all.sh "Qwen/Qwen3-8B"
# ./run_all.sh "Qwen/Qwen3-14B"
# ./run_all.sh "microsoft/Phi-4-mini-instruct"
# ./run_all.sh "microsoft/phi-4"

set -e

MODEL_NAME=${1:?"Usage: ./run_all.sh MODEL_NAME [START_STAGE]"}
START_STAGE=${2:-1}

for STAGE in $(seq $START_STAGE 4); do
    echo ""
    echo "########## Stage ${STAGE} / 4 시작 ##########"
    ./run_pipeline.sh "$MODEL_NAME" "$STAGE"
    echo "########## Stage ${STAGE} / 4 완료 ##########"
done

echo ""
echo "전체 파이프라인(Stage ${START_STAGE}~4) 완료: ${MODEL_NAME}"
