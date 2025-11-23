import json
import re
import argparse


def evaluate_accuracy(input_json, response_field="response", answer_field="output"):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct_cnt = 0
    skip_cnt = 0

    pattern = r"Therefore,?\s*the answer is\s*(.*?)(?=Therefore,?\s*the answer is|$)"
    failed_indices = []       # 파싱 실패 인덱스
    failed_samples = []       # 파싱 실패 상세 데이터

    for idx, item in enumerate(data):
        response = item.get(response_field, "")
        answer = item.get(answer_field, "").strip()

        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        predicted = matches[-1].strip(" .\n") if matches else None

        if predicted is None:
            skip_cnt += 1
            failed_indices.append(idx)
            failed_samples.append({
                "index": idx,
                "response_preview": response[:200].replace("\n", "\\n"),  # 앞 200자만
                "parsed": None
            })
            continue

        if predicted == answer:
            correct_cnt += 1

    accuracy = correct_cnt / total

    # -------- 보고서 출력 --------
    print("\n========== Accuracy Report ==========")
    print(f"총 데이터 개수: {total}")
    print(f"정답 맞춘 개수: {correct_cnt}")
    print(f"스킵(파싱 실패): {skip_cnt}")
    print(f"정확도 Accuracy: {accuracy:.4f}")
    print("=====================================")

    # -------- 파싱 실패 인덱스 출력 --------
    print("\n파싱 실패한 데이터 인덱스 5개 (있을 경우):")
    if failed_indices:
        print(failed_indices[:5])
    else:
        print("파싱 실패 없음")

    print("\n----- 파싱 실패 데이터 상세 (최대 5개) -----")
    for sample in failed_samples[:10]:
        print(f"[Index] {sample['index']}")
        print(f"[Response Preview] {sample['response_preview']}")
        print(f"[Parsed Result] {sample['parsed']}")
        print("-------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학생 모델 정답 비교 + 파싱 실패 분석")

    parser.add_argument("--input-json", required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--response-field", default="response", help="모델 응답 필드 이름")
    parser.add_argument("--answer-field", default="output", help="정답 필드 이름")

    args = parser.parse_args()

    evaluate_accuracy(
        input_json=args.input_json,
        response_field=args.response_field,
        answer_field=args.answer_field
    )
