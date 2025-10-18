import json

input_path = "all_task_train_right_wronghint_answer_B.json"
output_path = "before_pseudo_labeling_A(train_B,labeling_A).json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "response" in item:
        item["response"] = ""
    if "task_discription" in item:
        del item["task_discription"]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f" 'response' 및 'task_discription' 필드 제거 : {output_path}")
