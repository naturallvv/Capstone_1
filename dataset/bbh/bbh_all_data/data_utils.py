"""BBH data preprocessing utilities.

Answer checking uses the metric_utils.py logic (multi-delimiter fallback chain).
Consolidates functionality from accuracy.py, Change_prompt.py, Only_change_prompt.py,
delete_response.py, pair.py, merge.py, and split_data.py into a single module.
"""

import json
import os
import re
import copy
import random
import argparse


# ============================================================
# Answer checking (based on metric_utils.py logic)
# ============================================================

def extract_answers_for_model(model_output):
    """Extract (choice, content) from model output. From metric_utils.py."""
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
    """Extract (choice, content) from ground truth text. From metric_utils.py."""
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
    """Judgment for formal_fallacies task. From metric_utils.py."""
    if "invalid" in ground_truth:
        return "invalid" in model_answer
    else:
        return "invalid" not in model_answer


def extract_answer(text: str) -> str:
    """Extract the answer portion from model response using a multi-delimiter fallback chain.

    Based on the 6-stage fallback chain from metric_utils.py compute_metrics():
      1. "Therefore, the answer is"
      2. "[Answer Prediction]:"
      3. "Answer:" (with Explanation separation)
      4. "the answer is" (with Explanation separation)
      5. "\\n\\nA:"
      6. "### Response:"

    Returns:
        Extracted answer string. Empty string on parse failure.
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
    """Compare model response against ground truth.

    1. Extract answer from model response via extract_answer()
    2. Use decide() for formal_fallacies task
    3. Otherwise: compare choice first, then content

    Returns:
        bool: Whether the answer is correct.
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
        return md_choice in gt_choice
    else:
        if md_content == "none" or md_content == "":
            return False
        return gt_content == md_content


# ============================================================
# Accuracy evaluation
# ============================================================

def evaluate_accuracy(input_json, response_field="response", answer_field="output",
                      task_field="task_name"):
    """Evaluate accuracy of a JSON file with per-task breakdown and parse failure report.

    Uses check_answer() based on metric_utils.py logic.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct_cnt = 0
    skip_cnt = 0
    failed_indices = []
    failed_samples = []

    # Per-task accuracy counters
    task_total = {}
    task_correct = {}

    for idx, item in enumerate(data):
        response = item.get(response_field, "")
        answer = item.get(answer_field, "").strip()
        task_name = item.get(task_field, "")

        task_total[task_name] = task_total.get(task_name, 0) + 1

        # Attempt answer extraction
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

    # -------- Report --------
    print("\n========== Accuracy Report ==========")
    print(f"Total: {total}")
    print(f"Correct: {correct_cnt}")
    print(f"Skipped (parse fail): {skip_cnt}")
    print(f"Accuracy: {accuracy:.4f}")
    print("=====================================")

    # Per-task accuracy
    if task_total:
        print("\n----- Per-task Accuracy -----")
        for task in sorted(task_total.keys()):
            t_total = task_total[task]
            t_correct = task_correct.get(task, 0)
            t_acc = t_correct / t_total if t_total > 0 else 0.0
            print(f"  {task}: {t_correct}/{t_total} = {t_acc:.4f}")

    # Parse failure report
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
# Prompt transformation
# ============================================================

_template_cache = {}


def load_template(path_dir: str, task_name: str) -> str | None:
    """Load a template file (with caching)."""
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
    """Apply CCP template for correct answers, AHP for incorrect. Uses check_answer().

    Items with parse failures or missing templates are removed.
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

        # Load template
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
    """CoT few-shot 프롬프트로 교체."""
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
    """response 필드 초기화 + task_discription 제거."""
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
    """두 JSON 배열 병합."""
    with open(file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    merged = data1 + data2
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(data1)} + {len(data2)} = {len(merged)} -> {output_file}")


def split_data(input_json, file_a, file_b, split_ratio=0.5, seed=42):
    """데이터를 A/B로 랜덤 분할."""
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
    """EDIT용 chosen/rejected 선호도 쌍 생성.

    pos_pos + pos_neg -> chosen=pos_pos.response, rejected=pos_neg.response
    neg_pos + neg_neg -> chosen=neg_pos.response, rejected=neg_neg.response
    매칭 기준: instruction 동일
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

    # instruction -> item 인덱스 맵 (O(n) 매칭)
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
# CLI
# ============================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="data_utils",
        description="BBH data utilities (unified)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- accuracy ---
    p = sub.add_parser("accuracy", help="Evaluate accuracy")
    p.add_argument("--input-json", required=True)
    p.add_argument("--response-field", default="response")
    p.add_argument("--answer-field", default="output")
    p.add_argument("--task-field", default="task_name")

    # --- ahp-ccp ---
    p = sub.add_parser("ahp-ccp", help="Apply AHP/CCP templates")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--ahp-dir", required=True)
    p.add_argument("--ccp-dir", required=True)

    # --- replace-prompts ---
    p = sub.add_parser("replace-prompts", help="Replace prompts with CoT templates")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--prompt-dir", required=True)

    # --- clear ---
    p = sub.add_parser("clear", help="Clear responses")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)

    # --- merge ---
    p = sub.add_parser("merge", help="Merge two JSON files")
    p.add_argument("--file1", required=True)
    p.add_argument("--file2", required=True)
    p.add_argument("--output", required=True)

    # --- split ---
    p = sub.add_parser("split", help="Split data into A/B sets")
    p.add_argument("--input-json", required=True)
    p.add_argument("--file-a", required=True)
    p.add_argument("--file-b", required=True)
    p.add_argument("--split-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # --- pair ---
    p = sub.add_parser("pair", help="Create preference pairs for EDIT")
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
