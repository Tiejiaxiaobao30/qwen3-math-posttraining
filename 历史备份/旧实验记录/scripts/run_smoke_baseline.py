import json
import time
import torch

from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
SMOKE_IDS_PATH = "smoke_ids.json"
OUTPUT_PATH = "results/baseline_smoke_384_final.jsonl"


# 1. 读取固定的 20 个题目编号
with open(SMOKE_IDS_PATH, "r") as f:
    smoke_ids = json.load(f)


# 2. 加载 GSM8K
dataset = MsDataset.load(
    "modelscope/gsm8k",
    subset_name="main",
    split="train",
    cache_dir="/mnt/workspace/datasets_cache",
)


# 3. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
).to("cuda")

model.eval()


# 4. 逐题生成
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for i, sample_id in enumerate(smoke_ids):

        question = dataset[sample_id]["question"]
        reference = dataset[sample_id]["answer"]

        prompt = (
            f"Question: {question}\n"
            "Answer: Solve step by step and end with exactly: Final answer: <number>\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        input_tokens = inputs["input_ids"].shape[1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_time = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()

        elapsed = time.time() - start_time

        new_tokens = outputs[0][input_tokens:]
        output_tokens = len(new_tokens)

        model_answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        record = {
            "sample_id": sample_id,
            "question": question,
            "reference": reference,
            "model_answer": model_answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_sec": round(elapsed, 3),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "status": "success",
        }

        fout.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

        print(
            f"[{i+1:02d}/20] "
            f"id={sample_id} "
            f"time={elapsed:.2f}s "
            f"tokens={output_tokens}"
        )


print("\nSaved to:", OUTPUT_PATH)
