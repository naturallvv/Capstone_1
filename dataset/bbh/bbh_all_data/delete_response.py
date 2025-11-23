import json
import argparse

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "response" in item:
            item["response"] = ""
        if "task_discription" in item:
            del item["task_discription"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f" 'response' 초기화 및 'task_discription' 필드 제거 완료 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="response 초기화 및 task_discription 필드 제거 스크립트")

    parser.add_argument("--input-path", required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output-path", required=True, help="출력 JSON 파일 경로")

    args = parser.parse_args()

    main(args.input_path, args.output_path)