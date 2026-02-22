# download_ministral.py
from transformers import AutoTokenizer, AutoConfig
from transformers.models.ministral3 import Ministral3ForCausalLM

model_name = "mistralai/Ministral-3-3B-Base-2512"

print(f"Downloading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Ministral3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

print(f"✅ {model_name} downloaded!")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Model params: {model.num_parameters() / 1e9:.2f}B")