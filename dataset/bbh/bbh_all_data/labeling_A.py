""" CUDA_VISIBLE_DEVICES=0,1,2 python labeling_A.py """

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 병렬 토크나이징 활성화
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# -----------------------------
# Dataset 정의
# -----------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


# -----------------------------
# 모델 로드 함수
# -----------------------------
def load_model(model_name, quantization=False):
    print(f"[1] 모델 로드 중: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=quantization,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    print("[2] 모델 로드 완료")
    return tokenizer, model


# -----------------------------
# 병렬 배치 추론 함수
# -----------------------------
def generate_responses(
    model,
    tokenizer,
    prompts,
    max_new_tokens=1024,
    batch_size=8,
    num_workers=24
):
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    all_outputs = []
    print(f"[3] 총 {len(prompts)}개 프롬프트 추론 시작 "
          f"(batch_size={batch_size}, num_workers={num_workers})")

    for batch in tqdm(dataloader, desc="추론 진행", dynamic_ncols=True):
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_padded_length = enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

        for j, seq in enumerate(outputs):
            gen_tokens = seq[input_padded_length:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_outputs.append(text)

    print("[4] 추론 완료")
    return all_outputs

# -----------------------------
# 메인 실행 함수
# -----------------------------
def main(
    model_name,
    input_json="before_pseudo_labeling_A(train_B,labeling_A).json",
    output_json="after_labeling_A.json",
    prompt_field="prompt",
    response_field="response",
    quantization=False,
    max_new_tokens=2048,
    batch_size=8,
    num_workers=24
):
    tokenizer, model = load_model(model_name, quantization=quantization)

    # 입력 데이터 로드
    if input_json:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item[prompt_field] for item in data]
    else:
        prompts = []
        print("입력 JSON이 없어 수동 입력 모드입니다.")
        while True:
            p = input("프롬프트 입력 (종료: exit): ")
            if p.strip().lower() == "exit":
                break
            prompts.append(p)

    if not prompts:
        print("입력된 프롬프트가 없습니다.")
        return

    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 결과 저장 또는 출력
    if input_json and output_json:
        for item, resp in zip(data, responses):
            item[response_field] = resp
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[5] 결과 저장 완료 → {output_json}")
    else:
        for p, r in zip(prompts, responses):
            print(f"\n[입력] {p}\n[출력] {r}\n")


# -----------------------------
# CLI 인터페이스
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="NeaHyuk/Llama-3.2-1B_A")
    parser.add_argument("--input-json", default="before_pseudo_labeling_A(train_B,labeling_A).json", help="입력 JSON 파일 경로")
    parser.add_argument("--output-json", default="after_laveling_A.json", help="출력 JSON 파일 경로")
    parser.add_argument("--prompt-field", default="prompt", help="프롬프트 필드 이름")
    parser.add_argument("--response-field", default="response", help="응답 필드 이름")
    parser.add_argument("--quantization", action="store_true", help="8-bit 양자화 사용")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()

    main(**vars(args))