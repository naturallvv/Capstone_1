"""초기추론 + CCP/AHP 재추론 결과를 chosen/rejected 쌍으로 병합.

초기 추론이 정답이면 chosen=초기, rejected=재추론.
초기 추론이 오답이면 chosen=재추론, rejected=초기.
정답 비교는 metric_utils의 논문 저자 파서를 사용한다.
"""

import json
import copy
import argparse
import sys

sys.path.append(".")
from src.utils.metric_utils import extract_answers_for_model, extract_answers_for_gt


def extract_raw_answer(text):
    """compute_metrics()의 답 추출 로직 (metric_utils.py L127-146)."""
    if not text:
        return None
    if "Therefore, the answer is" in text:
        return text.split("Therefore, the answer is")[-1].split(".")[0].strip()
    elif "[Answer Prediction]:" in text:
        return text.split("[Answer Prediction]:")[-1].split(".")[0].strip()
    elif "Answer:" in text and "Explanation" not in text:
        return text.split("Answer:")[1].split(".")[0].strip()
    elif "Answer:" in text and "Explanation" in text:
        return text.split("Explanation")[0].split("Answer:")[-1].split(".")[0].strip()
    elif "The answer is" in text and "Explanation" in text:
        return text.split("Explanation")[0].split("The answer is")[-1].split(".")[0].strip()
    elif "\n\nA:" in text:
        return text.split("\n\nA:")[1].split(".")[0].strip()
    elif "### Response:" in text:
        return text.split("### Response:")[1].split(".")[0].strip()
    elif "the answer is" in text:
        return text.split("the answer is")[1].split(".")[0].strip()
    return None


def check_correct(response_text, gt_text):
    """compute_metrics()의 정답 비교 로직 (metric_utils.py L153-198)."""
    raw = extract_raw_answer(response_text)
    if raw is None:
        return False

    md_choice, md_content = extract_answers_for_model(copy.deepcopy(raw))
    gt_choice, gt_content = extract_answers_for_gt(copy.deepcopy(gt_text))

    if md_choice != "none" and gt_choice != "none":
        return md_choice in gt_choice
    else:
        if md_content == "none" or md_content == "":
            return False
        return gt_content == md_content


def merge(first_json, second_json, output_json):
    with open(first_json, "r", encoding="utf-8") as f:
        first_data = json.load(f)
    with open(second_json, "r", encoding="utf-8") as f:
        second_data = json.load(f)

    if len(first_data) != len(second_data):
        print("두 JSON 파일의 데이터 길이가 다릅니다.")
        return

    merged = []
    for item1, item2 in zip(first_data, second_data):
        input_text = item1.get("instruction", "")
        gt_answer = str(item1.get("output", "")).strip()

        resp_first = item1.get("response", "").strip()
        resp_second = item2.get("response", "").strip()

        is_correct = check_correct(resp_first, gt_answer)

        if is_correct:
            chosen, rejected = resp_first, resp_second
        else:
            chosen, rejected = resp_second, resp_first

        merged.append({
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected,
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"병합 완료: {output_json} ({len(merged)}건)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="초기추론 + CCP/AHP 추론 결과 병합")
    parser.add_argument("--first-json", required=True, help="초기 추론 JSON")
    parser.add_argument("--second-json", required=True, help="CCP/AHP 재추론 JSON")
    parser.add_argument("--output-json", required=True, help="병합 결과 저장 JSON")
    args = parser.parse_args()

    merge(args.first_json, args.second_json, args.output_json)
