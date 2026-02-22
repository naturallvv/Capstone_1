#!/bin/bash
# Parameterized KRSL script (replaces run_edit_step2_krsl.sh)
# Usage: ./run_krsl.sh [MODEL_NAME] [STUDENT] [STAGE_NUM] [PREV_EPOCH]
# Example: ./run_krsl.sh "meta-llama/Llama-3.2-1B" A 1 15

cd '../'

# ==================== Arguments ====================
if [ $# -lt 4 ]; then
    echo "Usage: ./run_krsl.sh [MODEL_NAME] [STUDENT] [STAGE_NUM] [PREV_EPOCH] [KRSL_DATA] [KRSL_WEIGHT]"
    echo "Example: ./run_krsl.sh \"TinyLlama/TinyLlama_v1.1\" A 1 15 \"./data.json\" \"./weight.pkl\""
    exit 1
fi

MODEL_NAME="$1"      # 필수 인자 (기본값 제거)
STUDENT="$2"         # 필수 인자 (기본값 제거)
STAGE_NUM="$3"       # 필수 인자 (기본값 제거)
PREV_EPOCH="$4"      # 필수 인자 (기본값 제거)
MODEL_SHORT="${MODEL_NAME##*/}"

# ==================== Previous stage model path ====================
if [ "$STAGE_NUM" -eq 1 ]; then
    model_name="../slm/hf/${MODEL_SHORT}/${STUDENT}/sft/epoch-${PREV_EPOCH}"
else
    PREV_STAGE=$((STAGE_NUM - 1))
    model_name="../slm/hf/${MODEL_SHORT}/${STUDENT}/stage${PREV_STAGE}/epoch-${PREV_EPOCH}"
fi

stage="stage${STAGE_NUM}"
student="$STUDENT"

# ==================== KRSL Hyperparameters ====================
lr=5e-6
weight_decay=0
gamma=0.95
alpha=0.5 # alpha * answer loss + (1-alpha) * rloss
gama=0.1  # preference loss coefficient, no use
krsl_alpha=1.0
krsl_beta=-0.025
krsl_gamma=0.0
krsl_nll_threshold=20
rouge2_below=1.01

dataset="bbh_krsl_dataset"
krsl_pre_dataset="bbh_llmst_dataset"
KRSL_DATA=${5:-"./dataset/bbh/bbh_all_data/all_task_train_right_preference_with_answer.json"}
KRSL_WEIGHT=${6:-"./dataset/bbh/bbh_all_data/all_task_train_right_preference_with_answer_precal.pkl"}
krsl_train_data_path="$KRSL_DATA"
krsl_weight_path="$KRSL_WEIGHT"
forward_type='concat'
beta=0.1       # for dpo, no use
dpo_loss_type='sigmoid' # for dpo, no use

# ==================== Training Config ====================
refine_it=0
save_type="hf"
output_dir="../slm/${save_type}/${MODEL_SHORT}"
ckpt_continue=None
max_words=1024
num_epochs=20

# ==================== 모델 크기별 배치 사이즈 자동 설정 ====================
PARAM_SIZE=$(echo "$MODEL_NAME" | grep -oiE '[0-9]+\.?[0-9]*b' | tail -1 | sed 's/[bB]$//')

if [ -z "$PARAM_SIZE" ]; then
    if echo "$MODEL_NAME" | grep -qi 'phi-4' && ! echo "$MODEL_NAME" | grep -qi 'mini'; then
        PARAM_SIZE="14"
    elif echo "$MODEL_NAME" | grep -qi 'phi-4.*mini\|phi-4-mini'; then
        PARAM_SIZE="4"
    elif echo "$MODEL_NAME" | grep -qi 'tinyllama'; then
        PARAM_SIZE="1"
    else
        PARAM_SIZE="1"
    fi
fi

# ≤3B: bs=8/ga=8, 4~8B: bs=4/ga=16, 13B+: bs=2/ga=32
SIZE_CAT=$(echo "$PARAM_SIZE" | awk '{if ($1 >= 13) print "large"; else if ($1 >= 4) print "medium"; else print "small"}')
if [ "$SIZE_CAT" = "large" ]; then
    batch_size_training=2
    gradient_accumulation_steps=32
elif [ "$SIZE_CAT" = "medium" ]; then
    batch_size_training=4
    gradient_accumulation_steps=16
else
    batch_size_training=8
    gradient_accumulation_steps=8
fi
echo "  [KRSL] 모델 크기: ${PARAM_SIZE}B (${SIZE_CAT}) → batch=${batch_size_training}, grad_accum=${gradient_accumulation_steps}"
num_workers_dataloader=1

use_peft=True
peft_method="lora"
seed=42
enable_fsdp=True
low_cpu_fsdp=False
run_validation=False
use_fp16=False
mixed_precision=True
val_batch_size=1
freeze_layers=False
num_freeze_layers=1
quantization=False
one_gpu=False
save_model=True
dist_checkpoint_root_folder="./output/fsdp/${MODEL_SHORT}/${STUDENT}/${stage}"
dist_checkpoint_folder="fine-tuned"
save_optimizer=False
use_fast_kernels=False

fsdp_activation_checkpointing=True
pure_bf16=False
optimizer="AdamW"

# ==================== Inference Config (no use during training) ====================
max_new_tokens=512
prompt_file=None
do_sample=True
min_length=None
top_p=1.0
temperature=1.0
top_k=50
repetition_penalty=1.0
length_penalty=1
max_padding_length=None
peft_model=None
enable_azure_content_safety=False
enable_sensitive_topics=False
enable_salesforce_content_safety=False
use_cache=True

# ==================== Log ====================
log_dir="./log/${MODEL_SHORT}/${STUDENT}/${stage}"
mkdir -p "$log_dir"
log_file="${log_dir}/log.txt"

# ==================== Run ====================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
torchrun --nnodes 1 --nproc_per_node 10 --master-port 29828 ./finetuning.py \
    --max_words $max_words \
    --enable_fsdp $enable_fsdp \
    --model_name "$model_name" \
    --low_cpu_fsdp $low_cpu_fsdp \
    --run_validation $run_validation \
    --batch_size_training $batch_size_training \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_epochs $num_epochs \
    --num_workers_dataloader $num_workers_dataloader \
    --lr $lr \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --seed $seed \
    --use_fp16 $use_fp16 \
    --mixed_precision $mixed_precision \
    --val_batch_size $val_batch_size \
    --dataset "$dataset" \
    --peft_method "$peft_method" \
    --use_peft $use_peft \
    --output_dir "$output_dir" \
    --freeze_layers $freeze_layers \
    --num_freeze_layers $num_freeze_layers \
    --quantization $quantization \
    --one_gpu $one_gpu \
    --save_model $save_model \
    --dist_checkpoint_root_folder "$dist_checkpoint_root_folder" \
    --dist_checkpoint_folder "$dist_checkpoint_folder" \
    --save_optimizer $save_optimizer \
    --use_fast_kernels $use_fast_kernels \
    --mixed_precision $mixed_precision \
    --fsdp_activation_checkpointing $fsdp_activation_checkpointing \
    --pure_bf16 $pure_bf16 \
    --optimizer "$optimizer" \
    --ckpt_continue $ckpt_continue \
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
    --save_type $save_type \
    --alpha $alpha \
    --refine_it $refine_it \
    --forward_type "$forward_type" \
    --beta $beta \
    --gama $gama \
    --krsl_alpha $krsl_alpha \
    --krsl_beta $krsl_beta \
    --krsl_gamma $krsl_gamma \
    --krsl_nll_threshold $krsl_nll_threshold \
    --rouge2_below $rouge2_below \
    --krsl_pre_dataset $krsl_pre_dataset \
    --krsl_train_data_path $krsl_train_data_path \
    --krsl_weight_path $krsl_weight_path \
    --max_padding_length $max_padding_length \
    --stage $stage \
    --student $student 2>&1 | tee "$log_file"
