"""편집 거리 가중치 사전 계산 스크립트 (A/B 통합, CLI 인자화).

기존 edit_dis_precal_A.py / edit_dis_precal_B.py 로직을 그대로 사용하되,
--data-file, --model-name, --max-words 를 CLI 인자로 받는다.
출력: 입력 파일의 .json → _precal.pkl 자동 변환.
"""

from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from src.datasets.utils import custom_tokenize
import pickle
import argparse
import os


def levenshtein_operations_tokens(s1_tokens, s2_tokens):
    # 공통 접두사 최적화
    prefix_len = 0
    while prefix_len < len(s1_tokens) and prefix_len < len(s2_tokens) and s1_tokens[prefix_len] == s2_tokens[prefix_len]:
        prefix_len += 1
    s1_tokens = s1_tokens[prefix_len:]
    s2_tokens = s2_tokens[prefix_len:]

    rows = len(s1_tokens) + 1
    cols = len(s2_tokens) + 1
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
    for i in range(1, cols):
        dp[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s1_tokens[row - 1] == s2_tokens[col - 1] else 1
            dp[row][col] = min(dp[row - 1][col] + 1,
                            dp[row][col - 1] + 1,
                            dp[row - 1][col - 1] + cost)

    operations = []
    row, col = rows - 1, cols - 1
    while row > 0 or col > 0:
        if row > 0 and dp[row][col] == dp[row - 1][col] + 1:
            operations.append(f'delete {s1_tokens[row - 1]}')
            row -= 1
        elif col > 0 and dp[row][col] == dp[row][col - 1] + 1:
            operations.append(f'insert {s2_tokens[col - 1]}')
            col -= 1
        else:
            operation = 'none' if s1_tokens[row - 1] == s2_tokens[col - 1] else f'replace {s1_tokens[row - 1]} with {s2_tokens[col - 1]}'
            operations.append(operation)
            row -= 1
            col -= 1

    operations.extend(['none'] * prefix_len)
    return operations[::-1], dp


def assign_weights(s1_tokens, s2_tokens, s1_type):
    """각 샘플에 대해 s1과 s2의 최소 편집 연산 가중치를 계산한다."""
    seq_len = s1_tokens.shape[0]
    operations, _ = levenshtein_operations_tokens(s1_tokens, s2_tokens)

    s1_weights = []
    for op in operations:
        if s1_type == 'chosen':
            weight = -1.0 if 'delete' in op or 'replace' in op else 0
        else:  # rejected
            weight = 0.5 if 'delete' in op or 'replace' in op else 0
        s1_weights.append(weight)

    if len(s1_weights) < seq_len:
        s1_weights += [0] * (seq_len - len(s1_weights))
    else:
        s1_weights = s1_weights[:seq_len]
    s1_weights = torch.tensor(s1_weights)

    return s1_weights, operations


def main():
    parser = argparse.ArgumentParser(description="편집 거리 가중치 사전 계산")
    parser.add_argument("--data-file", required=True, help="입력 preference JSON 파일 경로")
    parser.add_argument("--model-name", required=True, help="토크나이저 로드용 모델 경로 (base 모델)")
    parser.add_argument("--max-words", type=int, default=1024, help="최대 토큰 수")
    args = parser.parse_args()

    data_file = args.data_file
    model_name = args.model_name
    max_words = args.max_words

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    with open(data_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 편집 거리 가중치 계산
    new_data = []
    for i, ann in enumerate(tqdm(data, colour='white', desc='EDIT Distance Calculation', dynamic_ncols=True)):
        chosen_input_ids, _, _ = custom_tokenize(ann['input'], ann['input'] + ann['chosen'], tokenizer, max_words)
        rejected_input_ids, _, _ = custom_tokenize(ann['input'], ann['input'] + ann['rejected'], tokenizer, max_words)

        chosen_weights, chosen_operations = assign_weights(chosen_input_ids, rejected_input_ids, 'chosen')
        rejected_weights, rejected_operations = assign_weights(rejected_input_ids, chosen_input_ids, 'rejected')

        ann['chosen_weights'] = chosen_weights.tolist()
        ann['rejected_weights'] = rejected_weights.tolist()
        ann['chosen_operations'] = chosen_operations
        ann['rejected_operations'] = rejected_operations
        new_data.append(ann)

    # .json → _precal.pkl 자동 변환
    output_file = data_file.replace('.json', '_precal.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(new_data, file)

    print(f"저장 완료 -> {output_file}")


if __name__ == '__main__':
    main()
