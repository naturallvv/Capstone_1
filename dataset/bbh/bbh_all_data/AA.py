import json
from typing import List, Dict

# --------------------------------------------------
# ⬇️⬇️⬇️ 이 변수에 검사할 JSON 파일의 경로를 입력하세요. ⬇️⬇️⬇️
input_PATH = "response_Model_B_to_dataset_A.json"
# --------------------------------------------------

# 검사할 필드 이름 (기본값: "response")
RESPONSE_FIELD_NAME = "response"


def find_empty_responses(filepath: str, field_name: str):
    """
    JSON 파일을 읽어 'response' 필드가 비어있는 항목의
    개수와 인덱스 번호를 찾습니다.
    """
    
    if not filepath:
        print("="*50)
        print(" [오류] 'input_PATH' 변수가 비어있습니다. ")
        print(" 스크립트 상단의 input_PATH = \"\" 부분에 ")
        print(" 검사할 JSON 파일의 경로를 입력해주세요. ")
        print("="*50)
        return

    empty_indices = [] # 👈 비어있는 항목의 인덱스를 저장할 리스트
    total_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data: List[Dict] = json.load(f)

        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                print(f"오류: '{filepath}' 파일의 형식이 딕셔너리 리스트가 아닙니다.")
                return

        total_count = len(data)

        # enumerate를 사용해 0번부터 시작하는 인덱스(idx)와 항목(item)을 함께 가져옴
        for idx, item in enumerate(data):
            if not item.get(field_name, "").strip():
                empty_indices.append(idx) # 👈 비어있으면 인덱스 번호 추가

        empty_count = len(empty_indices)

        # --- 결과 출력 ---
        print(f"파일 분석 완료: '{filepath}'")
        print(f"---------------------------------")
        print(f"  총 항목 개수: {total_count}")
        print(f"  '{field_name}' 필드가 비어있는 항목 개수: {empty_count}")
        print(f"  '{field_name}' 필드가 채워진 항목 개수: {total_count - empty_count}")
        print(f"---------------------------------")

        # 👈 [추가] 비어있는 항목의 인덱스 번호(0-based) 출력
        if empty_indices:
            print(f"'{field_name}' 필드가 비어있는 항목의 인덱스 (0-based):")
            # 리스트를 보기 좋게 문자열로 변환하여 출력
            print_indices = ', '.join(map(str, empty_indices))
            print(f" [ {print_indices} ]")
        else:
            print(f"'{field_name}' 필드가 비어있는 항목이 없습니다.")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: '{filepath}'")
    except json.JSONDecodeError:
        print(f"오류: JSON 형식이 올바르지 않습니다. 파일을 확인하세요: '{filepath}'")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    find_empty_responses(input_PATH, RESPONSE_FIELD_NAME)