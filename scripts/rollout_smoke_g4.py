import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.reward import correctness_reward, format_reward


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
ADAPTER_PATH = "outputs/sft_full/final_adapter"
RL_MANIFEST_PATH = "data/processed/rl_manifest.jsonl"

G = 4
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 256
SEED = 42


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 1. 读取第一条真实 RL 数据
    with open(RL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    prompt = sample["prompt"]
    gold = sample["gold"]

    print("sample id:", sample["id"])
    print("gold:", gold)

    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # 3. 加载 Base
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    # 4. 挂上训练好的 SFT LoRA
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    # 5. prompt -> token
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    input_length = inputs["input_ids"].shape[1]

    # 6. 同一道题随机采样 G=4 个回答
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

    # 7. 分别 decode + reward
    rewards = []

    for i in range(G):
        new_tokens = outputs[i, input_length:]

        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        c_reward = correctness_reward(response, gold)
        f_reward = format_reward(response)

        rewards.append(c_reward)

        print("\n" + "=" * 80)
        print(f"ROLLOUT {i + 1}")
        print(response)
        print("\ncorrectness_reward:", c_reward)
        print("format_reward:", f_reward)

    print("\n" + "=" * 80)
    print("correctness reward group:", rewards)


if __name__ == "__main__":
    main()
