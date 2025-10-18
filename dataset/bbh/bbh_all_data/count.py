import json
import random
import os

input_path = "pseudo_from_B(train_A).json"
with open(input_path, "r", encoding='utf-8') as f:
    data = json.load(f)

print(len(data))