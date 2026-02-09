import copy
import json

import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from src.datasets.utils import custom_tokenize

PROMPT_DICT = {
    "prompt": (
        "{instruction}\n\nA:"
    ),
}

class EvalDataset(Dataset):
    # direct intruction fine-tuning with dataset ground truth
    def __init__(self, dataset_config, tokenizer, partition="test", max_words=30):
        self.ann = json.load(open(dataset_config.test_data_path))
        self.dataset = dataset_config.dataset
        # self.ann = self.ann[:8]
        self.max_words = dataset_config.max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # 원본 데이터를 가져옴 (여기엔 input, target만 있음)
        ann = self.ann[index]

        # [수정 1] 데이터 매핑 (ARC 데이터 포맷 -> 코드 요구 포맷)
        # 'input'을 'instruction'으로 복사
        if 'instruction' not in ann and 'input' in ann:
            ann['instruction'] = ann['input']
        
        # 'target'을 'output'으로 복사
        if 'output' not in ann and 'target' in ann:
            ann['output'] = ann['target']

        # [수정 2] 없는 키(Key)들에 대한 기본값 설정 (에러 방지)
        if 'task_name' not in ann:
            ann['task_name'] = 'ARC'  # 임의의 이름 지정
        
        if 'task_description' not in ann:
            ann['task_description'] = '' # 설명이 없으므로 빈칸 처리

        if 'response' not in ann:
            ann['response'] = '' # response가 없으면 빈칸 (Evaluation에서는 주로 output만 씀)

        # [수정 3] 이제 안전하게 데이터 사용 가능
        user_prompt = PROMPT_DICT['prompt'].format_map(ann)
        original_input = ann['instruction']
        original_output = ann['output']
        task_name = ann['task_name']
        teacher_response = ann['response']
        task_desc = ann['task_description']

        return {
            "user_prompt": user_prompt,
            "original_input": original_input,
            "original_output": original_output,
            'task_name': task_name,
            'task_description': task_desc,
            'teacher_response': teacher_response
        }