import json
import re
import os
from typing import List, Dict

INPUT_JSON_FILE = "new_response_Model_B_to_dataset_A.json"
OUTPUT_JSON_FILE = "re-prompted_dataset.json"
CORRECT_PROMPT_DIR = "/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ahp"
INCORRECT_PROMPT_DIR = "/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ccp"
TASK_NAME_FIELD = "task_name" 

def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = s.strip().strip("().,?! ")
    return s

def parse_response_answer(response_text: str) -> str | None:
    pattern = r"Therefore,?\s*the answer is\s*"
    matches = list(re.finditer(pattern, response_text, re.IGNORECASE))
    
    if not matches:
        return None
        
    last_match = matches[-1]
    raw_answer = response_text[last_match.end():]
    return normalize_answer(raw_answer)

def read_prompt_file(base_dir: str, task_name: str) -> str | None:
    file_path = os.path.join(base_dir, f"{task_name}.txt")
    
    if not os.path.exists(file_path):
        print(f"    [File Not Found] {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"    [File Read Error] {file_path}: {e}")
        return None

def main_processor():
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data: List[Dict] = json.load(f)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{INPUT_JSON_FILE}'을(를) 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: 입력 파일 '{INPUT_JSON_FILE}'이(가) 올바른 JSON 형식이 아닙니다.")
        return

    if not isinstance(data, list):
        data = [data]

    new_data_list = []
    correct_count = 0
    incorrect_count = 0
    skipped_count = 0

    print(f"총 {len(data)}개 항목에 대해 프롬프트 재구성을 시작합니다...")

    for i, item in enumerate(data):
        new_item = item.copy()
        
        response_text = item.get("response")
        ground_truth = item.get("output")
        task_name = item.get(TASK_NAME_FIELD)

        if not task_name:
            print(f"--- [Skip] 인덱스 {i}: '{TASK_NAME_FIELD}' 필드가 없습니다.")
            skipped_count += 1
            continue

        is_correct = False
        if response_text and ground_truth:
            extracted_answer = parse_response_answer(response_text)
            normalized_ground_truth = normalize_answer(ground_truth)
            if extracted_answer == normalized_ground_truth:
                is_correct = True
        
        if is_correct:
            prompt_dir = CORRECT_PROMPT_DIR
            correct_count += 1
        else:
            prompt_dir = INCORRECT_PROMPT_DIR
            incorrect_count += 1 

        prompt_content = read_prompt_file(prompt_dir, task_name)

        if prompt_content is None:
            print(f"--- [Skip] 인덱스 {i}: 프롬프트 파일 {prompt_dir}/{task_name}.txt 를 찾을 수 없습니다.")
            skipped_count += 1
            continue

        new_item["prompt"] = prompt_content
        new_item["response"] = ""
        
        new_data_list.append(new_item)

    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_data_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"\n오류: 새 파일 '{OUTPUT_JSON_FILE}' 저장 중 문제 발생: {e}")
        return

    print("\n" + "="*30)
    print("    프롬프트 재구성 완료    ")
    print("="*30)
    print(f"총 {len(data)}개 항목 처리 시도.")
    print(f"새 파일 '{OUTPUT_JSON_FILE}'에 {len(new_data_list)}개 항목 저장 완료.")
    print(f"  - 정답 (cot-ahp) 프롬프트 적용: {correct_count}개")
    print(f"  - 오답/파싱실패 (cot-ccp) 프롬프트 적용: {incorrect_count}개")
    print(f"  - 스킵 (task_name 누락 / .txt 파일 누락): {skipped_count}개")

if __name__ == "__main__":
    main_processor()