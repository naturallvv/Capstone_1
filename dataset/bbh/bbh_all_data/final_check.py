import json
import re
import argparse

# -----------------------------
# 파서 함수
# -----------------------------
def parse_answer(text: str) -> str | None:
    if not text:
        return None

    pattern = r"Therefore,?\s*the answer is\s*(.*?)(?=Therefore,?\s*the answer is|$)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))

    if not matches:
        return None

    raw = matches[-1].group(1)
    raw = raw.split("\n")[0]
    cleaned = raw.strip().strip(" .,:;\"'()[]{}")

    return cleaned if cleaned else None


# -----------------------------
# 필터링 함수 (reference 정답 비교 추가)
# -----------------------------
def filter_pairs(input_json, reference_json, output_json):
    # merged chosen/rejected 데이터
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # reference 파일 (정답 보유 데이터)
    with open(reference_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    kept = []
    removed = 0

    for idx, (item, ref_item) in enumerate(zip(data, ref_data)):

        chosen_text = item.get("chosen", "")
        rejected_text = item.get("rejected", "")
        gt_answer = str(ref_item.get("output", "")).strip()

        # 1) 빈 문자열 → 삭제
        if not chosen_text.strip() or not rejected_text.strip():
            removed += 1
            continue

        # 파싱
        parsed_chosen = parse_answer(chosen_text)
        parsed_rejected = parse_answer(rejected_text)

        # 2) 파싱 실패 → 삭제
        if parsed_chosen is None or parsed_rejected is None:
            removed += 1
            continue

        # normalize
        norm_chosen = parsed_chosen.lower().strip(" .,:;\"'()[]{}")
        norm_rejected = parsed_rejected.lower().strip(" .,:;\"'()[]{}")
        norm_gt = gt_answer.lower().strip(" .,:;\"'()[]{}")

        # 3) chosen/rejected 정답이 같으면 → 삭제
        if norm_chosen == norm_rejected:
            removed += 1
            continue

        # ⭐ 4) chosen의 답 ≠ reference JSON의 output → 삭제
        if norm_chosen != norm_gt:
            removed += 1
            continue

        kept.append(item)

    # 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print("========== Final Filter Result ==========")
    print(f"총 입력 데이터: {len(data)}")
    print(f"유지된 데이터: {len(kept)}")
    print(f"삭제된 데이터: {removed}")
    print(f"저장 경로 → {output_json}")
    print("=========================================")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최종 chosen/rejected 필터링")

    parser.add_argument("--input-json", required=True, help="합쳐진 merged JSON")
    parser.add_argument("--reference-json", required=True, help="change_prompt_after_labeling_A.json")
    parser.add_argument("--output-json", required=True, help="출력 JSON")

    args = parser.parse_args()

    filter_pairs(
        input_json=args.input_json,
        reference_json=args.reference_json,
        output_json=args.output_json
    )
