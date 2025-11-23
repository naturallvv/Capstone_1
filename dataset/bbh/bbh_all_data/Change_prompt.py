import json
import re
import os
import argparse

# ---------------------------------------------------------
# 강력한 답 파싱 함수 (robust version)
# ---------------------------------------------------------
def robust_parse_answer(text: str) -> str | None:
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
# 템플릿 캐시 + 로더
# ---------------------------------------------------------
template_cache = {}

def load_template(path_dir: str, task_name: str, tag: str) -> str | None:
    key = f"{task_name}_{tag}"

    if key in template_cache:
        return template_cache[key]

    file_path = os.path.join(path_dir, f"{task_name}.txt")

    if not os.path.exists(file_path):
        print(f"❌ 템플릿 없음: {file_path}")
        template_cache[key] = None
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
    except Exception as e:
        print(f"❌ 템플릿 읽기 오류: {file_path} ({e})")
        template_cache[key] = None
        return None

    template_cache[key] = template
    return template


# ---------------------------------------------------------
# 메인 처리 함수 (실패 항목 삭제)
# ---------------------------------------------------------
def process(
    input_json: str,
    output_json: str,
    ahp_dir: str,
    ccp_dir: str,
    response_field: str = "response",
    answer_field: str = "output",
    task_field: str = "task_name",
    instruction_field: str = "instruction",
):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    kept_items = []
    correct_cnt = 0
    incorrect_cnt = 0
    removed_cnt = 0

    print(f"총 {len(data)}개 항목 처리 시작…")

    for idx, item in enumerate(data):
        response_text = item.get(response_field, "")
        gt_answer = str(item.get(answer_field, "")).strip()
        task_name = str(item.get(task_field, "")).strip()
        instruction = str(item.get(instruction_field, "")).strip()

        if not task_name:
            print(f"[삭제] index {idx}: task_name 없음")
            removed_cnt += 1
            continue

        # 정답 판단만 predicted 로
        predicted = robust_parse_answer(response_text)
        predicted_str = predicted if predicted else ""

        norm_pred = predicted_str.lower().strip(" .,:;\"'()[]{}")
        norm_gt = gt_answer.lower().strip(" .,:;\"'()[]{}")
        is_correct = (norm_pred == norm_gt)

        # 템플릿 로드
        if is_correct:
            template = load_template(ccp_dir, task_name, tag="CCP")
        else:
            template = load_template(ahp_dir, task_name, tag="AHP")

        if template is None:
            print(f"[삭제] index {idx}: 템플릿 없음 ({task_name})")
            removed_cnt += 1
            continue

        # -------------------------------------------------
        # 🔥 CCP / AHP 프롬프트 생성 규칙 업데이트
        # -------------------------------------------------

        if is_correct:
            # ✔ 정답 맞춘 경우
            # {RIGHT} ← response 전체
            # {WRONG} ← ""
            correct_cnt += 1
            new_prompt = (
                template.replace("{QUESTION}", instruction)
                        .replace("{RIGHT}", response_text.strip())   # 전체 reasoning 삽입
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

    # 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kept_items, f, ensure_ascii=False, indent=2)

    print("\n========== 처리 결과 ==========")
    print(f"총 입력 데이터: {len(data)}")
    print(f"  ✔ 유지(성공): {len(kept_items)}")
    print(f"  ❌ 삭제됨: {removed_cnt}")
    print(f"  정답 → CCP 적용: {correct_cnt}")
    print(f"  오답 → AHP 적용: {incorrect_cnt}")
    print(f"저장 경로 → {output_json}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="정오답 기반 AHP/CCP 프롬프트 재구성")

    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ahp-dir", default="/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ahp")
    parser.add_argument("--ccp-dir", default="/home/Tenemin/Project/CasCoD/dataset/bbh/cot-ccp")

    args = parser.parse_args()

    process(
        input_json=args.input_json,
        output_json=args.output_json,
        ahp_dir=args.ahp_dir,
        ccp_dir=args.ccp_dir,
    )
