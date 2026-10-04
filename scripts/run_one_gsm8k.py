import torch
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/mnt/workspace/models/Qwen3-1.7B-Base"

ds = MsDataset.load(
    "modelscope/gsm8k",
    subset_name="main",
    split="train",
    cache_dir="/mnt/workspace/datasets_cache",
)

question = ds[1]["question"]
reference = ds[1]["answer"]

prompt = (
    f"Question: {question}\n"
    "Answer: Let's solve step by step.\n"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nQUESTION:")
print(question)

print("\nMODEL ANSWER:")
print(answer)

print("\nREFERENCE:")
print(reference)
