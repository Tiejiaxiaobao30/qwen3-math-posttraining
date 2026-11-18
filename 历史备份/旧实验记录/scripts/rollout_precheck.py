import json
import random
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluator import extract_answer
from scripts.reward import correctness_reward, format_reward


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
ADAPTER_PATH = "outputs/sft_full/final_adapter"
RL_PATH = "data/processed/rl_manifest.jsonl"

ROLLOUT_OUTPUT = "results/rollouts_precheck.jsonl"
STATS_OUTPUT = "results/group_stats.json"

NUM_PROMPTS = 50
G = 4

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 256
SEED = 42


def get_actual_length_and_truncation(new_tokens, eos_token_id):
    eos_positions = (
        new_tokens == eos_token_id
    ).nonzero(as_tuple=True)[0]

    if len(eos_positions) > 0:
        actual_length = int(eos_positions[0].item()) + 1
        truncated = False
    else:
        actual_length = int(new_tokens.shape[0])
        truncated = actual_length >= MAX_NEW_TOKENS

    return actual_length, truncated


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 1. 读取1000条RL manifest
    with open(RL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    assert len(rows) == 1000

    # 2. 固定seed抽50道
    samples = random.sample(rows, NUM_PROMPTS)

    # 3. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # 4. 加载 Base
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    # 5. 挂上 SFT LoRA
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    Path("results").mkdir(exist_ok=True)

    all_rollouts = []
    group_stats = []

    # 6. 50个prompt逐组采样
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

        rewards = []
        format_rewards = []
        lengths = []
        truncations = []

        for rollout_index in range(G):
            new_tokens = outputs[
                rollout_index,
                input_length:
            ]

            actual_length, truncated = (
                get_actual_length_and_truncation(
                    new_tokens,
                    tokenizer.eos_token_id,
                )
            )

            response = tokenizer.decode(
                new_tokens[:actual_length],
                skip_special_tokens=True,
            )

            parsed = extract_answer(response)

            c_reward = correctness_reward(
                response,
                gold,
            )

            f_reward = format_reward(response)

            rewards.append(c_reward)
            format_rewards.append(f_reward)
            lengths.append(actual_length)
            truncations.append(truncated)

            all_rollouts.append(
                {
                    "sample_id": sample["id"],
                    "group_index": group_index,
                    "rollout_index": rollout_index + 1,
                    "gold": gold,
                    "response": response,
                    "parsed_answer": (
                        str(parsed)
                        if parsed is not None
                        else None
                    ),
                    "correctness_reward": c_reward,
                    "format_reward": f_reward,
                    "generated_tokens": actual_length,
                    "truncated": truncated,
                }
            )

        # 7. 给这一组分类
        if all(r == 1.0 for r in rewards):
            group_type = "all_correct"
        elif all(r == 0.0 for r in rewards):
            group_type = "all_wrong"
        else:
            group_type = "mixed"

        reward_variance = statistics.pvariance(rewards)

        group_stats.append(
            {
                "sample_id": sample["id"],
                "gold": gold,
                "rewards": rewards,
                "format_rewards": format_rewards,
                "group_type": group_type,
                "reward_mean": statistics.mean(rewards),
                "reward_variance": reward_variance,
                "lengths": lengths,
                "truncated_count": sum(truncations),
            }
        )

        print(
            f"[{group_index:02d}/{NUM_PROMPTS}] "
            f"id={sample['id']} "
            f"rewards={rewards} "
            f"type={group_type}"
        )

    # 8. 保存200条原始rollout
    with open(
        ROLLOUT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        for row in all_rollouts:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    # 9. 汇总50组统计
    mixed = sum(
        g["group_type"] == "mixed"
        for g in group_stats
    )

    all_correct = sum(
        g["group_type"] == "all_correct"
        for g in group_stats
    )

    all_wrong = sum(
        g["group_type"] == "all_wrong"
        for g in group_stats
    )

    all_lengths = [
        row["generated_tokens"]
        for row in all_rollouts
    ]

    total_truncated = sum(
        row["truncated"]
        for row in all_rollouts
    )

    summary = {
        "config": {
            "num_prompts": NUM_PROMPTS,
            "num_generations": G,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        },
        "groups": {
            "total": NUM_PROMPTS,
            "mixed": mixed,
            "all_correct": all_correct,
            "all_wrong": all_wrong,
            "mixed_ratio": mixed / NUM_PROMPTS,
            "all_same_ratio": (
                all_correct + all_wrong
            ) / NUM_PROMPTS,
        },
        "rollouts": {
            "total": len(all_rollouts),
            "mean_generated_tokens": statistics.mean(
                all_lengths
            ),
            "min_generated_tokens": min(all_lengths),
            "max_generated_tokens": max(all_lengths),
            "truncated": total_truncated,
            "truncation_rate": (
                total_truncated / len(all_rollouts)
            ),
        },
        "group_details": group_stats,
    }

    with open(
        STATS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("groups:", NUM_PROMPTS)
    print("mixed:", mixed)
    print("all_correct:", all_correct)
    print("all_wrong:", all_wrong)
    print("mixed_ratio:", mixed / NUM_PROMPTS)
    print(
        "all_same_ratio:",
        (all_correct + all_wrong) / NUM_PROMPTS,
    )
    print(
        "mean_generated_tokens:",
        statistics.mean(all_lengths),
    )
    print("truncated:", total_truncated)
    print(
        "truncation_rate:",
        total_truncated / len(all_rollouts),
    )

    print("\nsaved:", ROLLOUT_OUTPUT)
    print("saved:", STATS_OUTPUT)


if __name__ == "__main__":
    main()
