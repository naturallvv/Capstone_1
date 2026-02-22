""" CUDA_VISIBLE_DEVICES=0,1,2 python labeling.py """

import os
import json
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

 # ✅ 추가: 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.utils.tokenizer_utils import setup_tokenizer, resize_model_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


def load_model(model_name, quantization=False):
    print(f"[1] 모델 로드 중: {model_name}")

    # ✅ 새 코드
    print("\n" + "="*60)
    print("🔧 Setting up tokenizer...")
    print("="*60)
    tokenizer, model_family = setup_tokenizer(model_name, padding_side='left')
    print("="*60 + "\n")

    # truncation_side는 별도 설정 (setup_tokenizer에 없으므로)
    tokenizer.truncation_side = "left"

    # Quantization 설정
    quantization_config = None
    if quantization:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    # ✅ 추가: embedding 크기 조정
    print("\n🔧 Checking token embeddings...")
    model = resize_model_embeddings(model, tokenizer, model_family)
    print("="*60 + "\n")
    model.eval()

    print("[2] 모델 로드 완료")
    return tokenizer, model


def generate_responses(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    batch_size,
    num_workers,
    do_sample,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    length_penalty,
    use_cache
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
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

        for seq in outputs:
            gen_tokens = seq[input_padded_length:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_outputs.append(text)

    print("[4] 추론 완료")
    return all_outputs


def main(
    model_name,
    input_json,
    output_json,
    prompt_field,
    response_field,
    quantization,
    max_new_tokens,
    batch_size,
    num_workers,
    do_sample,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    length_penalty,
    use_cache,
):
    tokenizer, model = load_model(model_name, quantization=quantization)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item[prompt_field] for item in data]

    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens,
        batch_size,
        num_workers,
        do_sample,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        length_penalty,
        use_cache
    )

    for item, resp in zip(data, responses):
        item[response_field] = resp

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[5] 결과 저장 완료 → {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)

    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")

    parser.add_argument("--quantization", action="store_true")

    # inference config
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=64)

    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--use-cache", type=bool, default=True)

    args = parser.parse_args()

    main(**vars(args))
