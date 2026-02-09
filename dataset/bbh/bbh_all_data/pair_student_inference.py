import json
import re
import argparse


# ---------------------------------------------------------
# robust parser for "Therefore, the answer is ..."
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# merge function
# ---------------------------------------------------------
def merge(first_json, second_json, output_json):
    with open(first_json, "r", encoding="utf-8") as f:
        first_data = json.load(f)

    with open(second_json, "r", encoding="utf-8") as f:
        second_data = json.load(f)

    if len(first_data) != len(second_data):
        print("❌ 두 JSON 파일의 데이터 길이가 다릅니다.")
        return

    merged = []

    for idx, (item1, item2) in enumerate(zip(first_data, second_data)):
        instruction = item1.get("instruction", "")
        input_text = instruction if instruction else ""

        gt_answer = str(item1.get("output", "")).strip()

        # 첫 번째 추론 결과에서 정답 파싱
        pred_first = parse_answer(item1.get("response", ""))

        # 정답 여부 판단
        norm_first = pred_first.lower().strip(" .,:;\"'()[]{}") if pred_first else None
        norm_gt = gt_answer.lower().strip(" .,:;\"'()[]{}")

        is_correct = (norm_first == norm_gt)

        # -------------------------------
        # 🔥 정오답에 따라 chosen/rejected 결정
        # -------------------------------
        resp_first  = item1.get("response", "").strip()
        resp_second = item2.get("response", "").strip()

        if is_correct:
            chosen = resp_first
            rejected = resp_second
        else:
            chosen = resp_second
            rejected = resp_first

        merged.append({
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected
        })

    # 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ 병합 완료! 저장 경로 → {output_json}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="초기추론 + CCP/AHP 추론 결과 병합기")

    parser.add_argument("--first-json", required=True, help="초기 추론 JSON")
    parser.add_argument("--second-json", required=True, help="CCP/AHP 재추론 JSON")
    parser.add_argument("--output-json", required=True, help="병합 결과 저장 JSON")

    args = parser.parse_args()

    merge(
        first_json=args.first_json,
        second_json=args.second_json,
        output_json=args.output_json
    )
