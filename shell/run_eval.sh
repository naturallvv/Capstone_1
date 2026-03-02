#!/bin/bash
# 파라미터화된 평가 스크립트 (기존 8개 eval 스크립트 통합)
# Usage: ./run_eval.sh [MODEL_PATH] [SAVED_MODEL_DIR] [TEST_DATASET] [GPU_ID] [PORT]
# Example: ./run_eval.sh "/path/to/model/epoch-1" "/path/to/saved_model_dir" "bbh_eval_dataset" 0 56543

cd '../'

# ==================== 인자 ====================
MODEL_PATH=${1:?"Usage: ./run_eval.sh MODEL_PATH SAVED_MODEL_DIR TEST_DATASET [GPU_ID] [PORT]"}
SAVED_MODEL_DIR=${2:?"Usage: ./run_eval.sh MODEL_PATH SAVED_MODEL_DIR TEST_DATASET [GPU_ID] [PORT]"}
TEST_DATASET=${3:?"Usage: ./run_eval.sh MODEL_PATH SAVED_MODEL_DIR TEST_DATASET [GPU_ID] [PORT]"}
GPU_ID=${4:-0}
PORT=${5:-56543}

# ==================== 평가 설정 ====================
model_name="$MODEL_PATH"
saved_model_dir="$SAVED_MODEL_DIR"
test_dataset="$TEST_DATASET"

eval_epoch_begin=-1
num_workers_dataloader=1
val_batch_size=256
sc_cot=False
vote_num=1
train_dataset='bbh_llmcmt_dataset'
load_type='hf'
quantization=False
max_words=150
max_new_tokens=1024
prompt_file=None
seed=42
do_sample=False
min_length=None
use_cache=True
top_p=1.0
temperature=1.0
top_k=50
repetition_penalty=1.0
length_penalty=1
enable_azure_content_safety=False
enable_sensitive_topics=False
enable_salesforce_content_safety=False
max_padding_length=256
use_fast_kernels=False
peft_model=None

# ==================== 실행 ====================
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$PORT"

echo "=========================================="
echo "Evaluation: ${TEST_DATASET}"
echo "  Model: ${MODEL_PATH}"
echo "  SavedDir: ${SAVED_MODEL_DIR}"
echo "  GPU: ${GPU_ID}, Port: ${PORT}"
echo "=========================================="

python evaluation.py \
    --model_name "$model_name" \
    --num_workers_dataloader $num_workers_dataloader \
    --seed $seed \
    --val_batch_size $val_batch_size \
    --max_words $max_words \
    --train_dataset "$train_dataset" \
    --test_dataset "$test_dataset" \
    --peft_model "$peft_model" \
    --max_new_tokens $max_new_tokens \
    --prompt_file "$prompt_file" \
    --do_sample $do_sample \
    --min_length $min_length \
    --use_cache $use_cache \
    --top_p $top_p \
    --temperature $temperature \
    --top_k $top_k \
    --repetition_penalty $repetition_penalty \
    --length_penalty $length_penalty \
    --enable_azure_content_safety $enable_azure_content_safety \
    --enable_sensitive_topics $enable_sensitive_topics \
    --enable_salesforce_content_safety $enable_salesforce_content_safety \
    --max_padding_length $max_padding_length \
    --use_fast_kernels $use_fast_kernels \
    --load_type $load_type \
    --sc_cot $sc_cot \
    --vote_num $vote_num \
    --eval_epoch_begin $eval_epoch_begin \
    --saved_model_dir $saved_model_dir
