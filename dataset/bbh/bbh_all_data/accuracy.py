import json, re

input_json="after_labeling_B.json"
response_field="response"
answer_field = "output"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

i = 0
k = 1

for item in data:
    response = item[response_field] 
    match = re.findall(r"Therefore, the answer is\s*(.*?)(?=Therefore, the answer is|$)", response) 
    predicted = match[-1].strip(". \n") if match else "Cant find answer"
    answer = item[answer_field]

    if predicted == "Cant find answer":
        k += 1

    if predicted == answer:
        i += 1
        accuracy = i / len(data)
        print(f"\r[Accuracy] {i} / {len(data)} = {accuracy:.4f}", end="")
        print(f" | Skipped: {k}", end="")