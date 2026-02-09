import json

def save_step_samples(
    input_json_path: str,
    output_json_path: str,
    step: int,
    global_batch_size: int = 32,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_idx = step * global_batch_size          
    end_idx = (step + 1) * global_batch_size       

    subset = data[start_idx:end_idx]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"step {step} → [{start_idx}, {end_idx - 1}] 샘플 {len(subset)}개 저장 완료: {output_json_path}")


# 예시 사용
save_step_samples(
    input_json_path="/home/Tenemin/Project/CasCoD/dataset/bbh/bbh_all_data/step_2_all_task_train_preference_with_answer_A.json",        
    output_json_path="step_5_samples.json",
    step=5,
    global_batch_size=32,
)
