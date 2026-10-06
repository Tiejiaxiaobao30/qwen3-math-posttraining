import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.reward import correctness_reward, format_reward
from scripts.evaluator import extract_answer


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
ADAPTER_PATH = "outputs/sft_full/final_adapter"
RL_PATH = "data/processed/rl_manifest.jsonl"
OUTPUT_PATH = "results/rollouts_precheck_5x4.jsonl"

NUM_PROMPTS = 5
G = 4

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 256
SEED = 42


def main():
    # 固定随机种子，方便以后复现
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 1. 读取1000条RL数据
    with open(RL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    assert len(rows) == 1000

    # 固定随机抽5道，而不是人为挑题
    samples = random.sample(rows, NUM_PROMPTS)

    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # 3. Base + SFT LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    Path("results").mkdir(exist_ok=True)

    all_results = []

    # 4. 一道题一道题采样
    for group_index, sample in enumerate(samples, start=1):
        prompt = sample["prompt"]
        gold = sample["gold"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=G,
                pad_token_id=tokenizer.eos_token_id,
            )

        group_rewards = []

        for rollout_index in range(G):
            new_tokens = outputs[rollout_index, input_length:]

            response = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            parsed = extract_answer(response)

            c_reward = correctness_reward(
                response,
                gold,
            )

            f_reward = format_reward(response)

            group_rewards.append(c_reward)

            result = {
                "sample_id": sample["id"],
                "group_index": group_index,
                "rollout_index": rollout_index + 1,
                "gold": gold,
                "response": response,
                "parsed_answer": (
                    str(parsed) if parsed is not None else None
                ),
                "correctness_reward": c_reward,
                "format_reward": f_reward,
                "generated_tokens": int(new_tokens.shape[0]),
            }

            all_results.append(result)

        print(
            f"group {group_index}/5 "
            f"id={sample['id']} "
            f"gold={gold} "
            f"rewards={group_rewards}"
        )

    # 5. 保存20条原始rollout
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in all_results:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print("\nsaved:", OUTPUT_PATH)
    print("total rollouts:", len(all_results))


if __name__ == "__main__":
    main()
