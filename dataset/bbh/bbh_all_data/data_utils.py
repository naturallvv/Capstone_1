"""BBH 데이터 전처리 유틸리티.

정답 체크는 metric_utils.py 로직(다중 delimiter 폴백 체인)을 사용한다.
기존 accuracy.py, Change_prompt.py, Only_change_prompt.py, delete_response.py,
pair.py, merge.py, split_data.py 기능을 하나의 모듈로 통합.
"""

import json
import os
import re
import copy
import random
import argparse


# ============================================================
# 정답 체크 (metric_utils.py 로직 기반)
# ============================================================

def extract_answers_for_model(model_output):
    """모델 출력에서 (choice, content) 튜플을 추출한다. metric_utils.py 원본."""
    if model_output is None:
        return "none", "none"
    model_output = model_output.strip()
    model_output = model_output.rstrip(".")
    model_output = model_output.lower()
    model_output = model_output.replace("\uff09", ")").replace("\uff08", "(")

    if len(model_output) == 1:
        md_choice = f"({model_output})"
        md_content = "none"
    else:
        pattern = r"\([a-z]\)"
        match = re.search(pattern, model_output)
        if match:
            md_choice = match.group(0)
            sp_list = model_output.split(")")
            if len(sp_list) == 1:
                md_content = "none"
            else:
                md_content = sp_list[1]
                md_content = md_content.lstrip(".")
                md_content = md_content.rstrip(".")
                md_content = md_content.strip()
        else:
            md_choice = "none"
            md_content = model_output.lstrip(".")
            md_content = md_content.rstrip(".")
            md_content = md_content.strip()

    return md_choice, md_content


def extract_answers_for_gt(original_output):
    """정답 텍스트에서 (choice, content) 튜플을 추출한다. metric_utils.py 원본."""
    original_output = original_output.strip()
    original_output = original_output.rstrip(".")
    original_output = original_output.lower()
    original_output = original_output.replace("\uff09", ")").replace("\uff08", "(")

    pattern = r"\([a-z]\)"
    match = re.search(pattern, original_output)
    if match:
        gt_choice = match.group(0)
        sp_list = original_output.split(")")
        if len(sp_list) == 1:
            gt_content = "none"
        else:
            gt_content = sp_list[1]
            gt_content = gt_content.lstrip(".")
            gt_content = gt_content.rstrip(".")
            gt_content = gt_content.strip()
    else:
        gt_choice = "none"
        gt_content = original_output.lstrip(".")
        gt_content = gt_content.rstrip(".")
        gt_content = gt_content.strip()

    return gt_choice, gt_content


def decide(ground_truth: str, model_answer: str) -> bool:
    """formal_fallacies 태스크 전용 판정 함수. metric_utils.py 원본."""
    if "invalid" in ground_truth:
        return "invalid" in model_answer
    else:
        return "invalid" not in model_answer


def extract_answer(text: str) -> str:
    """다중 delimiter 폴백 체인으로 모델 응답에서 답변 부분만 추출한다.

    metric_utils.py compute_metrics() 내부의 6단계 폴백 체인:
      1. "Therefore, the answer is"
      2. "[Answer Prediction]:"
      3. "Answer:" (Explanation 분리 처리)
      4. "the answer is" (Explanation 분리 처리)
      5. "\\n\\nA:"
      6. "### Response:"

    Returns:
        추출된 답변 문자열. 추출 실패 시 빈 문자열 반환.
    """
    if not text:
        return ""

    if "Therefore, the answer is" in text:
        return text.split("Therefore, the answer is")[-1].split(".")[0].strip()
    elif "Therefore the answer is" in text:
        return text.split("Therefore the answer is")[-1].split(".")[0].strip()
    elif "[Answer Prediction]:" in text:
        return text.split("[Answer Prediction]:")[-1].split(".")[0].strip()
    elif "Answer:" in text and "Explanation" not in text:
        return text.split("Answer:")[1].split(".")[0].strip()
    elif "Answer:" in text and "Explanation" in text:
        return text.split("Explanation")[0].split("Answer:")[-1].split(".")[0].strip()
    elif "The answer is" in text and "Explanation" in text:
        return text.split("Explanation")[0].split("The answer is")[-1].split(".")[0].strip()
    elif "the answer is" in text:
        return text.split("the answer is")[-1].split(".")[0].strip()
    elif "\n\nA:" in text:
        return text.split("\n\nA:")[1].split(".")[0].strip()
    elif "### Response:" in text:
        return text.split("### Response:")[1].split(".")[0].strip()
    else:
        return ""


def check_answer(response: str, ground_truth: str, task_name: str = "") -> bool:
    """모델 응답과 정답을 비교한다.

    1. extract_answer()로 모델 응답에서 답변 추출
    2. formal_fallacies 태스크면 decide() 사용
    3. 그 외: choice 비교 -> content 비교 순서

    Returns:
        bool: 정답 여부
    """
    extracted = extract_answer(response)

    if task_name == "formal_fallacies":
        gt_content = ground_truth.lower().strip()
        md_content = extracted.lower().strip() if extracted else response.lower().strip()
        return decide(ground_truth=gt_content, model_answer=md_content)

    if not extracted:
        return False

    gt_choice, gt_content = extract_answers_for_gt(copy.deepcopy(ground_truth))
    md_choice, md_content = extract_answers_for_model(copy.deepcopy(extracted))

    if md_choice and md_choice != "none" and gt_choice and gt_choice != "none":
        # 선택지가 있으면 선택지끼리 비교
        return md_choice in gt_choice
    else:
        # 선택지가 없으면 내용끼리 비교
        if md_content == "none" or md_content == "":
            return False
        return gt_content == md_content


# ============================================================
# 정확도 평가
# ============================================================

def evaluate_accuracy(input_json, response_field="response", answer_field="output",
                      task_field="task_name"):
    """JSON 파일의 정확도를 평가한다. 태스크별 정확도 및 파싱 실패 리포트 포함.

    정답 체크에 metric_utils.py 기반 check_answer()를 사용한다.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct_cnt = 0
    skip_cnt = 0
    failed_indices = []
    failed_samples = []

    # 태스크별 정확도 집계용 딕셔너리
    task_total = {}
    task_correct = {}

    for idx, item in enumerate(data):
        response = item.get(response_field, "")
        answer = item.get(answer_field, "").strip()
        task_name = item.get(task_field, "")

        # 태스크별 총 개수 카운트
        task_total[task_name] = task_total.get(task_name, 0) + 1

        # 답변 추출 시도
        extracted = extract_answer(response)
        if not extracted and task_name != "formal_fallacies":
            skip_cnt += 1
            failed_indices.append(idx)
            failed_samples.append({
                "index": idx,
                "task_name": task_name,
                "response_preview": response[:200].replace("\n", "\\n"),
                "parsed": None,
            })
            continue

        if check_answer(response, answer, task_name):
            correct_cnt += 1
            task_correct[task_name] = task_correct.get(task_name, 0) + 1

    accuracy = correct_cnt / total if total > 0 else 0.0

    # -------- 결과 보고서 출력 --------
    print("\n========== Accuracy Report ==========")
    print(f"Total: {total}")
    print(f"Correct: {correct_cnt}")
    print(f"Skipped (parse fail): {skip_cnt}")
    print(f"Accuracy: {accuracy:.4f}")
    print("=====================================")

    # 태스크별 정확도 출력
    if task_total:
        print("\n----- Per-task Accuracy -----")
        for task in sorted(task_total.keys()):
            t_total = task_total[task]
            t_correct = task_correct.get(task, 0)
            t_acc = t_correct / t_total if t_total > 0 else 0.0
            print(f"  {task}: {t_correct}/{t_total} = {t_acc:.4f}")

    # 파싱 실패 리포트 출력
    if failed_indices:
        print(f"\nParse failures: {len(failed_indices)} items")
        print("First 5 failed indices:", failed_indices[:5])
        print("\n----- Failed samples (max 5) -----")
        for sample in failed_samples[:5]:
            print(f"  [Index] {sample['index']}  [Task] {sample['task_name']}")
            print(f"  [Preview] {sample['response_preview']}")
            print("  ---")

    return {
        "total": total,
        "correct": correct_cnt,
        "skipped": skip_cnt,
        "accuracy": accuracy,
        "task_accuracy": {
            t: task_correct.get(t, 0) / task_total[t]
            for t in task_total
        },
        "failed_indices": failed_indices,
    }


# ============================================================
# 프롬프트 변환
# ============================================================

_template_cache = {}


def load_template(path_dir: str, task_name: str) -> str | None:
    """템플릿 파일을 로드한다. 캐시를 사용하여 중복 읽기를 방지."""
    key = f"{path_dir}|{task_name}"
    if key in _template_cache:
        return _template_cache[key]

    file_path = os.path.join(path_dir, f"{task_name}.txt")
    if not os.path.exists(file_path):
        print(f"[WARN] Template not found: {file_path}")
        _template_cache[key] = None
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
    except Exception as e:
        print(f"[WARN] Template read error: {file_path} ({e})")
        _template_cache[key] = None
        return None

    _template_cache[key] = template
    return template


def apply_ahp_ccp(input_json, output_json, ahp_dir, ccp_dir,
                   response_field="response", answer_field="output",
                   task_field="task_name", instruction_field="instruction"):
    """정답이면 CCP, 오답이면 AHP 템플릿을 적용한다. check_answer() 사용.

    파싱 실패 또는 템플릿이 없는 항목은 결과에서 제거된다.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    kept_items = []
    correct_cnt = 0
    incorrect_cnt = 0
    removed_cnt = 0

    print(f"Processing {len(data)} items...")

    for idx, item in enumerate(data):
        response_text = item.get(response_field, "")
        gt_answer = str(item.get(answer_field, "")).strip()
        task_name = str(item.get(task_field, "")).strip()
        instruction = str(item.get(instruction_field, "")).strip()

        if not task_name:
            print(f"[Remove] index {idx}: no task_name")
            removed_cnt += 1
            continue

        is_correct = check_answer(response_text, gt_answer, task_name)

        # 정답/오답에 따라 템플릿 로드
        if is_correct:
            template = load_template(ccp_dir, task_name)
        else:
            template = load_template(ahp_dir, task_name)

        if template is None:
            print(f"[Remove] index {idx}: no template ({task_name})")
            removed_cnt += 1
            continue

        # CCP / AHP 프롬프트 생성
        if is_correct:
            correct_cnt += 1
            new_prompt = (
                template.replace("{QUESTION}", instruction)
                        .replace("{RIGHT}", response_text.strip())
                        .replace("{WRONG}", "")
            )
        else:
            incorrect_cnt += 1
            new_prompt = (
                template.replace("{QUESTION}", instruction)
                        .replace("{CORRECT}", gt_answer)
            )

        item["prompt"] = new_prompt
        item[response_field] = ""
        kept_items.append(item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kept_items, f, ensure_ascii=False, indent=2)

    print(f"\n========== Result ==========")
    print(f"Input:   {len(data)}")
    print(f"Kept:    {len(kept_items)}")
    print(f"Removed: {removed_cnt}")
    print(f"Correct (CCP): {correct_cnt}")
    print(f"Wrong   (AHP): {incorrect_cnt}")
    print(f"Saved -> {output_json}")


def replace_prompts(input_json, output_json, prompt_dir):
    """CoT few-shot 프롬프트로 교체한다."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    total = len(data)
    replaced = 0
    skipped = 0

    for item in data:
        task_name = item.get("task_name")
        instruction = item.get("instruction", "")

        if not task_name:
            skipped += 1
            continue

        prompt_file_path = os.path.join(prompt_dir, f"{task_name}.txt")
        if not os.path.exists(prompt_file_path):
            print(f"[WARN] Prompt file not found: {prompt_file_path}")
            skipped += 1
            continue

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template_text = f.read().rstrip()

        tail = (
            "\n\n"
            f"Q: {instruction}\n"
            f"A: Let's think step by step."
        )
        item["prompt"] = template_text + tail
        replaced += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n====== DONE ======")
    print(f"Total: {total}")
    print(f"Replaced: {replaced}")
    print(f"Skipped: {skipped}")
    print(f"Saved -> {output_json}")


# ============================================================
# 데이터 조작
# ============================================================

def clear_responses(input_json, output_json):
    """response 필드를 초기화하고 task_discription 필드를 제거한다."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "response" in item:
            item["response"] = ""
        if "task_discription" in item:
            del item["task_discription"]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Cleared responses, removed task_discription -> {output_json}")


def merge_json(file1, file2, output_file):
    """두 JSON 배열을 하나로 병합한다."""
    with open(file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    merged = data1 + data2
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(data1)} + {len(data2)} = {len(merged)} -> {output_file}")


def split_data(input_json, file_a, file_b, split_ratio=0.5, seed=42):
    """데이터를 A/B 두 세트로 랜덤 분할한다."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(total * split_ratio)
    data_a = data[:split_idx]
    data_b = data[split_idx:]

    with open(file_a, "w", encoding="utf-8") as f:
        json.dump(data_a, f, ensure_ascii=False, indent=4)
    with open(file_b, "w", encoding="utf-8") as f:
        json.dump(data_b, f, ensure_ascii=False, indent=4)

    print(f"Split complete!")
    print(f"  Total:  {total}")
    print(f"  A set:  {len(data_a)} -> {os.path.abspath(file_a)}")
    print(f"  B set:  {len(data_b)} -> {os.path.abspath(file_b)}")


def create_preference_pairs(pos_pos, pos_neg, neg_pos, neg_neg, output_file):
    """EDIT용 chosen/rejected 선호도 쌍을 생성한다.

    pos_pos + pos_neg -> chosen=pos_pos.response, rejected=pos_neg.response
    neg_pos + neg_neg -> chosen=neg_pos.response, rejected=neg_neg.response
    매칭 기준: instruction 필드가 동일한 항목끼리 쌍을 이룸
    """
    PROMPT_TEMPLATE = "Task Description:\n{task_description}\nQ:{instruction}\n\nA:"

    with open(pos_pos, "r", encoding="utf-8") as f:
        data_pp = json.load(f)
    with open(pos_neg, "r", encoding="utf-8") as f:
        data_pn = json.load(f)
    with open(neg_pos, "r", encoding="utf-8") as f:
        data_np = json.load(f)
    with open(neg_neg, "r", encoding="utf-8") as f:
        data_nn = json.load(f)

    # instruction 기준 딕셔너리 매핑 (O(n) 탐색)
    pn_map = {item["instruction"]: item for item in data_pn}
    nn_map = {item["instruction"]: item for item in data_nn}

    edit_data_list = []

    for data in data_pp:
        matched = pn_map.get(data["instruction"])
        if matched is None:
            continue
        edit_data_list.append({
            "input": PROMPT_TEMPLATE.format_map(data),
            "chosen": data["response"],
            "rejected": matched["response"],
        })

    for data in data_np:
        matched = nn_map.get(data["instruction"])
        if matched is None:
            continue
        edit_data_list.append({
            "input": PROMPT_TEMPLATE.format_map(data),
            "chosen": data["response"],
            "rejected": matched["response"],
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(edit_data_list, f, ensure_ascii=False, indent=4)

    print(f"Created {len(edit_data_list)} preference pairs -> {output_file}")


# ============================================================
# CLI (커맨드라인 인터페이스)
# ============================================================

def _build_parser():
    """argparse 파서를 구성한다. 서브커맨드별로 인자를 정의."""
    parser = argparse.ArgumentParser(
        prog="data_utils",
        description="BBH data utilities (unified)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- 정확도 평가 ---
    p = sub.add_parser("accuracy", help="Evaluate accuracy")
    p.add_argument("--input-json", required=True)
    p.add_argument("--response-field", default="response")
    p.add_argument("--answer-field", default="output")
    p.add_argument("--task-field", default="task_name")

    # --- AHP/CCP 템플릿 적용 ---
    p = sub.add_parser("ahp-ccp", help="Apply AHP/CCP templates")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--ahp-dir", required=True)
    p.add_argument("--ccp-dir", required=True)

    # --- 프롬프트 교체 ---
    p = sub.add_parser("replace-prompts", help="Replace prompts with CoT templates")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--prompt-dir", required=True)

    # --- response 초기화 ---
    p = sub.add_parser("clear", help="Clear responses")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)

    # --- JSON 병합 ---
    p = sub.add_parser("merge", help="Merge two JSON files")
    p.add_argument("--file1", required=True)
    p.add_argument("--file2", required=True)
    p.add_argument("--output", required=True)

    # --- 데이터 분할 ---
    p = sub.add_parser("split", help="Split data into A/B sets")
    p.add_argument("--input-json", required=True)
    p.add_argument("--file-a", required=True)
    p.add_argument("--file-b", required=True)
    p.add_argument("--split-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # --- 선호도 쌍 생성 ---
    p = sub.add_parser("pair", help="EDIT용 선호도 쌍 생성")
    p.add_argument("--pos-pos", required=True)
    p.add_argument("--pos-neg", required=True)
    p.add_argument("--neg-pos", required=True)
    p.add_argument("--neg-neg", required=True)
    p.add_argument("--output", required=True)

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "accuracy":
        evaluate_accuracy(
            input_json=args.input_json,
            response_field=args.response_field,
            answer_field=args.answer_field,
            task_field=args.task_field,
        )

    elif args.command == "ahp-ccp":
        apply_ahp_ccp(
            input_json=args.input_json,
            output_json=args.output_json,
            ahp_dir=args.ahp_dir,
            ccp_dir=args.ccp_dir,
        )

    elif args.command == "replace-prompts":
        replace_prompts(
            input_json=args.input_json,
            output_json=args.output_json,
            prompt_dir=args.prompt_dir,
        )

    elif args.command == "clear":
        clear_responses(
            input_json=args.input_json,
            output_json=args.output_json,
        )

    elif args.command == "merge":
        merge_json(
            file1=args.file1,
            file2=args.file2,
            output_file=args.output,
        )

    elif args.command == "split":
        split_data(
            input_json=args.input_json,
            file_a=args.file_a,
            file_b=args.file_b,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )

    elif args.command == "pair":
        create_preference_pairs(
            pos_pos=args.pos_pos,
            pos_neg=args.pos_neg,
            neg_pos=args.neg_pos,
            neg_neg=args.neg_neg,
            output_file=args.output,
        )
