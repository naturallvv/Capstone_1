import json
import random
import os

input_path = "all_task_train_preference_with_answer.json"
with open(input_path, "r", encoding='utf-8') as f:
    data = json.load(f)

print(len(data))