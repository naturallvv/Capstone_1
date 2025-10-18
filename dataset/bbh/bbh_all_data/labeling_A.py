
#python create_response.py `
# --model-id NeaHyuk/Llama-3.2-1B_A `
# --infile dataset/all_task_train_right_wronghint_answer_B_clean.json `
# --outfile dataset/response_Model_A_to_dataset_B.json


import json
import argparse
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id: str):
    print("[1] 모델 로드 시작:", model_id)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    print("[2] 토크나이저 로드 완료")
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    print("[3] 모델 로드 완료")
    return tok, model

def batched_generate_only_new_tokens(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens=524,
    temperature=0.6,
    top_p=0.9,
    batch_size=64,
):
    import torch
    all_outs = []
    print(f"[4] 총 {len(prompts)}개 프롬프트에 대해 배치 생성 시작 (batch_size={batch_size})")
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            #padding_side='left',
            truncation=True,
        ).to(model.device)

        input_lengths = enc["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, seq in enumerate(gen):
            # 입력 길이 이후(=새로 생성된 토큰)만 취함
            gen_tokens = seq[input_lengths[j]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_outs.append(text)
    print("[5] 배치 생성 완료")
    return all_outs

def main():
    print("[0] 스크립트 시작")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="/home/Tenemin/Project/slm/hf/Llama-3.2-1B_A/std-lr=0.0002-wd=0.05-alpha=1train_data_name=all_task_train_right_wronghint_answer_A-bbh_llmst_dataset/epoch-15")
    ap.add_argument("--infile", default="before_pseudo_labeling_B(train_A,labeling_B).json")
    ap.add_argument("--outfile", default="response_Model_A_to_dataset_B.json")
    ap.add_argument("--prompt-field", default="prompt", help="프롬프트 키 이름 (기본: prompt)")
    ap.add_argument("--response-field", default="response", help="결과를 쓸 키 이름 (기본: response)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--skip-has-response",
        action="store_true",
        help="이미 response가 비어있지 않은 항목은 건너뜀",
    )
    args = ap.parse_args()

    # 데이터 로드
    with open(args.infile, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    # 프롬프트 수집 (skip 옵션 적용)
    prompts, indices = [], []
    for idx, item in enumerate(data):
        if args.prompt_field not in item or not str(item[args.prompt_field]).strip():
            raise ValueError(f"[{idx}] '{args.prompt_field}'가 비어있습니다.")
        if args.skip_has_response and item.get(args.response_field, "").strip():
            continue
        prompts.append(item[args.prompt_field])
        indices.append(idx)

    if not prompts:
        print("생성할 샘플이 없습니다. (모든 항목에 response가 존재하거나 skip 옵션에 의해 건너뜀)")
        return

    # 모델 로드 & 생성
    tok, model = load_model(args.model_id)
    generations = batched_generate_only_new_tokens(
        tokenizer=tok,
        model=model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    assert len(generations) == len(indices)

    # 결과 주입
    for idx, resp in zip(indices, generations):
        data[idx][args.response_field] = resp

    # 저장
    outpath = args.outfile or args.infile
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {outpath} (updated {len(indices)} samples))")

if __name__ == "__main__":
    main()