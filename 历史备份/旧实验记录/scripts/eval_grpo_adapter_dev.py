import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluator import build_result


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
SFT_ADAPTER_PATH = "outputs/sft_full/final_adapter"
DEV_PATH = Path("data/processed/dev.jsonl")
PROMPT_PATH = Path("configs/sft_prompt.txt")

MAX_NEW_TOKENS = 512


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grpo-adapter",
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    device = "cuda"

    print(
        "run_name =",
        args.run_name,
        flush=True,
    )

    print(
        "GRPO adapter =",
        args.grpo_adapter,
        flush=True,
    )

    # ==================================================
    # 1. dev500
    # ==================================================

    dev = load_jsonl(
        DEV_PATH
    )[:args.limit]

    assert len(dev) == args.limit

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        prompt_template = f.read()

    # ==================================================
    # 2. tokenizer
    # ==================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # ==================================================
    # 3. Base
    # ==================================================

    print(
        "\n=== LOAD BASE ===",
        flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # ==================================================
    # 4. SFT adapter
    # ==================================================

    print(
        "=== LOAD SFT ADAPTER ===",
        flush=True,
    )

    sft_model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
        is_trainable=False,
    )

    # ==================================================
    # 5. merge SFT
    # ==================================================

    print(
        "=== MERGE SFT ===",
        flush=True,
    )

    merged_model = (
        sft_model
        .merge_and_unload()
    )

    if hasattr(
        merged_model,
        "peft_config",
    ):
        delattr(
            merged_model,
            "peft_config",
        )

    if hasattr(
        merged_model,
        "_hf_peft_config_loaded",
    ):
        merged_model._hf_peft_config_loaded = False

    # ==================================================
    # 6. GRPO adapter
    # ==================================================

    print(
        "=== LOAD GRPO ADAPTER ===",
        flush=True,
    )

    model = PeftModel.from_pretrained(
        merged_model,
        args.grpo_adapter,
        is_trainable=False,
    )

    model = model.to(device)
    model.eval()

    lora_count = sum(
        1
        for name, _
        in model.named_parameters()
        if "lora_" in name
    )

    print(
        "GRPO LoRA tensors =",
        lora_count,
        flush=True,
    )

    assert lora_count > 0

    # ==================================================
    # 7. evaluation
    # ==================================================

    output_path = Path(
        f"results/{args.run_name}_dev500.jsonl"
    )

    summary_path = Path(
        f"results/{args.run_name}_dev500_summary.json"
    )

    counts = {
        "correct": 0,
        "wrong": 0,
        "format_fail": 0,
        "truncated": 0,
    }

    print(
        "\n=== START DEV500 ===",
        flush=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for index, sample in enumerate(
            dev,
            start=1,
        ):

            prompt = prompt_template.format(
                question=sample["question"]
            )

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
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            new_tokens = outputs[
                0,
                input_length:
            ]

            prediction = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            generated_tokens = len(
                new_tokens
            )

            truncated = (
                generated_tokens
                >= MAX_NEW_TOKENS
                and
                new_tokens[-1].item()
                != tokenizer.eos_token_id
            )

            result = build_result(
                sample_id=sample["id"],
                prediction=prediction,
                reference=sample["final_answer"],
                generated_tokens=generated_tokens,
                truncated=truncated,
            )

            counts[
                result["status"]
            ] += 1

            if truncated:
                counts["truncated"] += 1

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (
                index == 1
                or index % 25 == 0
            ):
                acc = (
                    counts["correct"]
                    / index
                    * 100
                )

                print(
                    f"[{index}/500] "
                    f"correct={counts['correct']} "
                    f"wrong={counts['wrong']} "
                    f"format_fail={counts['format_fail']} "
                    f"acc={acc:.2f}%",
                    flush=True,
                )

    # ==================================================
    # 8. summary
    # ==================================================

    total = len(dev)

    accuracy = (
        counts["correct"]
        / total
        * 100
    )

    format_success = (
        (
            counts["correct"]
            +
            counts["wrong"]
        )
        / total
        * 100
    )

    truncation_rate = (
        counts["truncated"]
        / total
        * 100
    )

    summary = {
        "run_name": args.run_name,
        "grpo_adapter": args.grpo_adapter,
        "total": total,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "format_fail": counts["format_fail"],
        "truncated": counts["truncated"],
        "strict_accuracy": accuracy,
        "format_success_rate": format_success,
        "truncation_rate": truncation_rate,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n==============================",
        flush=True,
    )

    print(
        f"=== {args.run_name} RESULT ===",
        flush=True,
    )

    print(
        "==============================",
        flush=True,
    )

    print(
        "correct =",
        counts["correct"],
        flush=True,
    )

    print(
        "wrong =",
        counts["wrong"],
        flush=True,
    )

    print(
        "format_fail =",
        counts["format_fail"],
        flush=True,
    )

    print(
        "truncated =",
        counts["truncated"],
        flush=True,
    )

    print(
        "strict_accuracy =",
        f"{accuracy:.2f}%",
        flush=True,
    )

    print(
        "format_success_rate =",
        f"{format_success:.2f}%",
        flush=True,
    )

    print(
        "saved:",
        output_path,
        flush=True,
    )

    print(
        "saved:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
