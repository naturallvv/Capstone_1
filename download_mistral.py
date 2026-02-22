# download_mistral.py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"

print(f"Downloading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

print(f"✅ {model_name} downloaded!")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Model params: {model.num_parameters() / 1e9:.2f}B")