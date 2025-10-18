import json
import re
from typing import List, Dict

# --------------------------------------------------
# ⬇️⬇️⬇️ 이 변수에 검사할 JSON 파일의 경로를 입력하세요. ⬇️⬇️⬇️
JSON_FILE_PATH = "new_response_Model_B_to_dataset_A.json"
# --------------------------------------------------


def normalize_answer(s: str) -> str:
    """
    답변을 비교 가능하도록 정규화하는 함수.
    1. 문자열로 변환
    2. 소문자화
    3. 앞뒤 공백/특수문자(.,?!() 등) 제거
    """
    s = str(s).lower()
    # 앞뒤 공백 및 흔한 문장 부호 제거
    s = s.strip().strip("().,?! ")
    return s


def parse_response_answer(response_text: str) -> str | None:
    """
    [수정] "Therefore, the answer is"의 "마지막" 발생 지점 뒤의
    텍스트를 파싱하고 정규화합니다.
    """
    
    # 1. "Therefore, the answer is" 키 구문을 찾기 위한 정규식
    #    (쉼표, 띄어쓰기 차이 무시)
    pattern = r"Therefore,?\s*the answer is\s*"
    
    # 2. re.finditer를 사용해 모든 일치 항목을 찾음 (대소문자 무시)
    matches = list(re.finditer(pattern, response_text, re.IGNORECASE))
    
    # 3. 일치 항목이 없는 경우 (파싱 실패)
    if not matches:
        return None
        
    # 4. "마지막" 일치 항목을 가져옴
    last_match = matches[-1]
    
    # 5. 마지막 일치 구문이 끝난 "이후"의 모든 텍스트를 추출
    #    (last_match.end()는 일치 구문 바로 다음의 인덱스를 반환)
    raw_answer = response_text[last_match.end():]
    
    # 6. 정규화하여 반환
    return normalize_answer(raw_answer)


def main():
    """
    JSON 파일을 로드하여 response와 output을 비교하고 정확도를 계산합니다.
    [수정] 파싱 실패(Parse Error)를 오답(Incorrect)으로 처리합니다.
    """
    if not JSON_FILE_PATH:
        print("="*50)
        print(" [오류] 'JSON_FILE_PATH' 변수가 비어있습니다. ")
        print(" 스크립트 상단의 JSON_FILE_PATH = \"\" 부분에 ")
        print(" 검사할 JSON 파일의 경로를 입력해주세요. ")
        print("="*50)
        return

    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data: List[Dict] = json.load(f)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. '{JSON_FILE_PATH}'")
        return
    except json.JSONDecodeError:
        print(f"오류: JSON 형식이 올바르지 않습니다. '{JSON_FILE_PATH}'")
        return

    if not isinstance(data, list):
        data = [data]

    total_count = len(data)
    correct_count = 0
    # parse_error_count = 0  # 👈 [제거] 파싱 실패 카운트 제거
    skipped_count = 0      # 👈 [추가] 필수 필드가 없는 항목 카운트
    
    print(f"총 {total_count}개 항목에 대한 채점을 시작합니다...")
    print("-" * 30)

    for i, item in enumerate(data):
        response_text = item.get("response")
        ground_truth = item.get("output")

        if response_text is None or ground_truth is None:
            print(f"--- [Skip] 인덱스 {i}: 'response' 또는 'output' 필드가 없습니다.")
            skipped_count += 1
            continue

        # 🔻 [수정] 로직 변경: 성공(Correct)이 아니면 모두 오답(Incorrect)으로 처리
        
        extracted_answer = parse_response_answer(response_text)
        normalized_ground_truth = normalize_answer(ground_truth)

        if extracted_answer == normalized_ground_truth:
            # 1. 정답인 경우
            correct_count += 1
        else:
            # 2. 오답이거나 파싱 실패인 경우 (모두 오답으로 처리)
            print(f"--- [Incorrect] 인덱스 {i} ---")
            print(f"  정답 (output): '{normalized_ground_truth}'")
            
            if extracted_answer is None:
                # 2-1. 파싱 실패인 경우
                print(f"  모델 답 (parsed): 'None (파싱 실패)'")
                print(f"  (오류: 'Therefore, the answer is' 구문을 찾을 수 없습니다.)")
                print(f"  Response (마지막 100자): ...{response_text[-100:]}")
            else:
                # 2-2. 답이 틀린 경우
                print(f"  모델 답 (parsed): '{extracted_answer}'")
            
            # parse_error_count += 1  # 👈 [제거]
            # continue                # 👈 [제거]

    print("\n" + "="*30)
    print("      채점 완료      ")
    print("="*30)
    
    # 🔻 [수정] 최종 리포트 로직
    analyzed_count = total_count - skipped_count # 스킵된 항목 제외
    
    if analyzed_count == 0:
        print("분석할 데이터가 없습니다.")
        return

    incorrect_count = analyzed_count - correct_count

    print(f"총 분석 항목 (Skip 제외): {analyzed_count}")
    print(f"정답 (Correct): {correct_count}")
    print(f"오답 (Incorrect, 파싱실패 포함): {incorrect_count}")
    # print(f"파싱 실패 (Parse Errors): {parse_error_count}") # 👈 [제거]
    
    if analyzed_count > 0:
        accuracy = (correct_count / analyzed_count) * 100
        print(f"\n정확도 (정답 / 분석 항목): {accuracy:.2f}%")
    else:
        print("\n유효한 채점 항목이 없습니다.")

if __name__ == "__main__":
    main() 