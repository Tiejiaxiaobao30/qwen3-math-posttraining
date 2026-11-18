import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.reward import correctness_reward


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"

SFT_ADAPTER_PATH = (
    "outputs/sft_full/final_adapter"
)

GRPO_ADAPTER_PATH = (
    "outputs/grpo_smoke/final_adapter"
)

RL_FILE = (
    "data/processed/rl_manifest.jsonl"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
        ]


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    # ==================================================
    # 1. tokenizer
    # ==================================================

    print("\n=== LOAD TOKENIZER ===")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ==================================================
    # 2. 加载最原始 Base
    #
    # 当前：
    # W_base
    # ==================================================

    print("\n=== LOAD BASE ===")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )


    # ==================================================
    # 3. 加载 Day07 SFT adapter
    #
    # 当前：
    # W_base + ΔW_sft
    # ==================================================

    print("\n=== LOAD SFT ADAPTER ===")

    sft_model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
        is_trainable=False,
    )


    # ==================================================
    # 4. merge SFT
    #
    # W_sft =
    # W_base + ΔW_sft
    # ==================================================

    print("\n=== MERGE SFT ===")

    sft_merged_model = (
        sft_model.merge_and_unload()
    )

    old_lora_count = sum(
        1
        for name, _ in
        sft_merged_model.named_parameters()
        if "lora_" in name
    )

    print(
        "LoRA tensors after SFT merge:",
        old_lora_count,
    )


    # ==================================================
    # 5. 加载刚刚 GRPO smoke 训练出的 adapter
    #
    # 当前：
    # W_sft + ΔW_grpo
    # ==================================================

    print("\n=== LOAD GRPO ADAPTER ===")

    model = PeftModel.from_pretrained(
        sft_merged_model,
        GRPO_ADAPTER_PATH,
        is_trainable=False,
    )

    model = model.to(device)
    model.eval()

    print(
        "model type:",
        type(model).__name__,
    )

    print(
        "loaded adapters:",
        list(model.peft_config.keys()),
    )

    grpo_lora_count = sum(
        1
        for name, _ in model.named_parameters()
        if "lora_" in name
    )

    print(
        "GRPO LoRA tensor count:",
        grpo_lora_count,
    )


    # ==================================================
    # 6. 取真实 RL prompt
    # ==================================================

    rows = load_jsonl(
        RL_FILE
    )

    test_rows = rows[:3]

    print(
        "\n=== RELOAD INFERENCE ==="
    )


    # ==================================================
    # 7. 重新生成
    # ==================================================

    for i, row in enumerate(
        test_rows,
        start=1,
    ):

        prompt = row["prompt"]
        gold = str(row["gold"])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.no_grad():

            outputs = model.generate(
                **inputs,

                max_new_tokens=256,

                # reload smoke 先用确定性生成
                do_sample=False,

                pad_token_id=(
                    tokenizer.pad_token_id
                ),

                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        new_tokens = outputs[
            :,
            input_length:
        ]

        response = tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        )

        reward = correctness_reward(
            response,
            gold,
        )

        print(
            f"\n----- SAMPLE {i} -----"
        )

        print(
            "gold:",
            gold,
        )

        print(
            "\nresponse:"
        )

        print(response)

        print(
            "\nstrict correctness reward:",
            reward,
        )


    # ==================================================
    # 8. 验收
    # ==================================================

    print(
        "\n=============================="
    )

    print(
        "GRPO ADAPTER RELOAD PASS"
    )

    print(
        "=============================="
    )

    print(
        "Base -> SFT merge -> "
        "GRPO adapter -> inference"
    )

    print(
        "Reloaded policy is usable."
    )


if __name__ == "__main__":
    main()
