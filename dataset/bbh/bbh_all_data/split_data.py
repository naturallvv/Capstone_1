import json
import random
import os

def split_data(input_path, f_A, f_B, split_ratio=0.5, seed=42):
    with open(input_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(total * split_ratio)
    data_A = data[:split_idx]
    data_B = data[split_idx:]

    with open(f_A, "w", encoding="utf-8") as f:
        json.dump(data_A, f, ensure_ascii=False, indent=4)
    with open(f_B, "w", encoding="utf-8") as f:
        json.dump(data_B, f, ensure_ascii=False, indent=4)

    #why the fuck can't type korean ssibal geotgatnea
    print(f"✅ Split complete!")
    print(f" ├─ Total samples : {total}")
    print(f" ├─ A set : {len(data_A)} samples → {os.path.abspath(f_A)}")
    print(f" └─ B set : {len(data_A)} samples → {os.path.abspath(f_B)}")

input_path = "all_task_train_wrong_answer_hint_right.json"
f_A = "all_task_train_wrong_answer_hint_right_A.json"
f_B = "all_task_train_wrong_answer_hint_right_B.json"

split_data(
    input_path = input_path,
    f_A = f_A,
    f_B = f_B
)