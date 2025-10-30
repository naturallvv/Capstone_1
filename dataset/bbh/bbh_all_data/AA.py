import json, re, os

# -----------------------------
# 경로 설정
# -----------------------------
input_json = "after_labeling_A.json"
response_field = "response"
answer_field = "output"
task_field = "task_name"
template_dir_correct = "/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ccp"   # 정답용 템플릿 폴더
template_dir_incorrect = "/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ahp" # 오답용 템플릿 폴더
output_json = "after_labeling_A_chage_prompt.json"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# 템플릿 캐시 (중복 로드 방지)
# -----------------------------
template_cache = {}    