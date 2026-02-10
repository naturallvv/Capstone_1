"""최종 chosen/rejected 쌍 필터링.

필터링 조건:
1. 빈 응답 제거
2. 답 파싱 실패 제거
3. chosen/rejected 답이 동일하면 제거
4. chosen 답이 ground truth와 다르면 제거

정답 비교는 metric_utils의 논문 저자 파서를 사용.
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


def answers_equal(text1, text2):
    """두 응답의 파싱된 답이 같은지 확인."""
    raw1 = extract_raw_answer(text1)
    raw2 = extract_raw_answer(text2)
    if raw1 is None or raw2 is None:
        return raw1 is None and raw2 is None

    md_choice1, md_content1 = extract_answers_for_model(copy.deepcopy(raw1))
    md_choice2, md_content2 = extract_answers_for_model(copy.deepcopy(raw2))

    if md_choice1 != "none" and md_choice2 != "none":
        return md_choice1 == md_choice2
    return md_content1 == md_content2


def filter_pairs(input_json, reference_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(reference_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    kept = []
    removed = 0

    for item, ref_item in zip(data, ref_data):
        chosen_text = item.get("chosen", "")
        rejected_text = item.get("rejected", "")
        gt_answer = str(ref_item.get("output", "")).strip()

        # 1) 빈 응답 제거
        if not chosen_text.strip() or not rejected_text.strip():
            removed += 1
            continue

        # 2) 답 파싱 실패 제거
        if extract_raw_answer(chosen_text) is None or extract_raw_answer(rejected_text) is None:
            removed += 1
            continue

        # 3) chosen/rejected 답이 동일하면 제거
        if answers_equal(chosen_text, rejected_text):
            removed += 1
            continue

        # 4) chosen 답이 ground truth와 다르면 제거
        if not check_correct(chosen_text, gt_answer):
            removed += 1
            continue

        kept.append(item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print("========== Final Filter Result ==========")
    print(f"총 입력 데이터: {len(data)}")
    print(f"유지된 데이터: {len(kept)}")
    print(f"삭제된 데이터: {removed}")
    print(f"저장 경로 → {output_json}")
    print("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최종 chosen/rejected 필터링")
    parser.add_argument("--input-json", required=True, help="병합된 paired JSON")
    parser.add_argument("--reference-json", required=True, help="정답 포함 원본 데이터")
    parser.add_argument("--output-json", required=True, help="필터링 결과 저장 JSON")
    args = parser.parse_args()

    filter_pairs(args.input_json, args.reference_json, args.output_json)
