import json
import os
from collections import Counter

# ==========================================
# [설정] 분석할 데이터 파일의 경로를 여기에 입력하세요
# ==========================================
FILE_PATH = "after_pseudo_labeling_B(train_A,labeling_B).json"  # 예: "C:/Users/User/Desktop/data.json"
# ==========================================

def count_and_print_stats(file_path):
    # 1. 파일이 있는지 확인
    if not os.path.exists(file_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return

    try:
        # 2. JSON 파일 읽기
        print(f"📂 '{file_path}' 파일을 읽는 중...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 데이터 확인
        if not isinstance(data, list):
            print("❌ 오류: 데이터가 리스트([]) 형식이 아닙니다.")
            return

        # 4. task_name 개수 세기
        # task_name이 없는 데이터는 'Unknown'으로 분류하거나 제외할 수 있습니다.
        task_list = [item.get('task_name', 'Unknown') for item in data]
        counter = Counter(task_list)

        # 5. 결과 출력
        print("\n" + "="*30)
        print(f"📊 분석 결과 (총 데이터: {len(data)}개)")
        print("="*30)
        
        # 개수가 많은 순서대로 출력
        for task, count in counter.most_common():
            print(f"• {task}: {count}개")
            
        print("="*30 + "\n")

    except json.JSONDecodeError:
        print(f"❌ 오류: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"❌ 알 수 없는 오류 발생: {e}")

# 실행
if __name__ == "__main__":
    count_and_print_stats(FILE_PATH)