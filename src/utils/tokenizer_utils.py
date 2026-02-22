# src/utils/tokenizer_utils.py

from transformers import AutoTokenizer


def setup_tokenizer(model_name, padding_side='left'):
    """
    모델별로 적절한 tokenizer 설정
    
    Args:
        model_name (str): HuggingFace 모델 경로 또는 로컬 체크포인트 경로
        padding_side (str): padding 방향 ('left' or 'right')
    
    Returns:
        tuple: (tokenizer, model_family)
            - tokenizer: 설정이 완료된 AutoTokenizer 객체
            - model_family: 모델 패밀리 이름 (str)
    """
    import os
    import json
    
    # 원본 모델 이름 추출 (로컬 경로 지원)
    original_model_name = model_name
    
    # 로컬 경로인 경우 config.json에서 원본 모델 이름 찾기
    if os.path.exists(model_name):
        config_path = os.path.join(model_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                # _name_or_path에 원본 모델 이름 저장됨
                original_model_name = config.get("_name_or_path", model_name)
                print(f"📁 Local checkpoint detected: {model_name}")
                print(f"📌 Original model: {original_model_name}")
    
    # Tokenizer 로드 (로컬 경로든 HF 이름이든 모두 가능)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    
    # 모델 패밀리 감지 (원본 이름으로)
    model_name_lower = original_model_name.lower()
    
    # ⚠️ 순서 중요: 더 구체적인 것부터 체크
    if "tinyllama" in model_name_lower:
        model_family = "tinyllama"
    elif "llama" in model_name_lower:
        model_family = "llama"
    elif "mistral" in model_name_lower or "ministral" in model_name_lower:
        model_family = "mistral"
    elif "qwen" in model_name_lower:
        model_family = "qwen"
    elif "gemma" in model_name_lower:
        model_family = "gemma"
    elif "phi" in model_name_lower:
        model_family = "phi"
    else:
        model_family = "unknown"
        print(f"⚠️  Unknown model family: {original_model_name}")
    
    # PAD token 설정
    if tokenizer.pad_token is None:
        if model_family == "llama":
            # LLaMA 3.x, 2.x: 전용 PAD 토큰 추가
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            print(f"✅ [{model_family}] Added <PAD> token")
        
        elif model_family in ["tinyllama", "mistral", "qwen", "phi"]:
            # TinyLlama, Mistral, Qwen, Phi: EOS를 PAD로 사용
            tokenizer.pad_token = tokenizer.eos_token
            print(f"✅ [{model_family}] Using EOS as PAD")
        
        elif model_family == "gemma":
            # Gemma: 보통 자동으로 설정되지만, 없으면 EOS 사용
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
                print(f"✅ [{model_family}] PAD token already exists: {tokenizer.pad_token}")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"✅ [{model_family}] Using EOS as PAD")
        
        else:
            # 기본값: EOS를 PAD로 (안전)
            tokenizer.pad_token = tokenizer.eos_token
            print(f"⚠️  [unknown] Using EOS as PAD")
    
    else:
        print(f"✅ [{model_family}] PAD token already exists: {tokenizer.pad_token}")
    
    # 디버깅 정보 출력
    print(f"📊 Tokenizer info:")
    print(f"   Model: {original_model_name}")
    if model_name != original_model_name:
        print(f"   Path: {model_name}")
    print(f"   Family: {model_family}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    
    return tokenizer, model_family

def resize_model_embeddings(model, tokenizer, model_family):
    """
    tokenizer에 토큰이 추가된 경우 모델의 embedding layer 크기 조정
    
    Args:
        model: AutoModelForCausalLM 객체
        tokenizer: AutoTokenizer 객체
        model_family: 모델 패밀리 이름
    
    Returns:
        model: 크기 조정된 모델
    """
    # LLaMA만 토큰을 추가하므로 크기 조정 필요
    if model_family == "llama" and len(tokenizer) != model.config.vocab_size:
        print(f"🔧 Resizing token embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    return model