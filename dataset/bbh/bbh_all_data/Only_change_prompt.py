import os
import json
import argparse

# -----------------------------------
# 메인 처리 함수
# -----------------------------------
def replace_prompts(input_json, output_json, prompt_dir):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 데이터 구조가 dict 하나일 수도 있으니 리스트 강제
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
            print(f"⚠ 프롬프트 파일 없음: {prompt_file_path}")
            skipped += 1
            continue

        # 템플릿 내용 불러오기
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template_text = f.read().rstrip()

        # 마지막에 붙일 내용
        tail = (
            "\n\n"
            f"Q: {instruction}\n"
            f"A: Let's think step by step."
        )

        # 최종 prompt 생성
        new_prompt = template_text + tail

        # 적용
        item["prompt"] = new_prompt
        replaced += 1

    # 결과 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n====== DONE ======")
    print(f"총 데이터: {total}")
    print(f"프롬프트 교체 완료: {replaced}")
    print(f"스킵(task_name 누락 또는 파일 없음): {skipped}")
    print(f"저장 경로 → {output_json}")


# -----------------------------------
# CLI
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="task_name 기반 prompt 교체기")

    parser.add_argument("--input-json", required=True, help="입력 JSON 경로")
    parser.add_argument("--output-json", required=True, help="출력 JSON 경로")
    parser.add_argument("--prompt-dir", required=True, help="cot-prompts 디렉토리 경로")

    args = parser.parse_args()

    replace_prompts(
        input_json=args.input_json,
        output_json=args.output_json,
        prompt_dir=args.prompt_dir
    )
