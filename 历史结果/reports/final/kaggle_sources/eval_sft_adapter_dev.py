import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from scripts.evaluator import build_result


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"
DEV_PATH = Path("data/processed/dev.jsonl")
PROMPT_PATH = Path("configs/sft_prompt.txt")
RESULTS_DIR = Path("results")

MAX_NEW_TOKENS = 512


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
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

    print(
        "=== SFT ADAPTER DEV EVAL ==="
    )

    print(
        "adapter =",
        args.adapter,
    )

    print(
        "run_name =",
        args.run_name,
    )

    print(
        "samples =",
        len(dev),
    )

    print(
        "max_new_tokens =",
        MAX_NEW_TOKENS,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    model = (
        PeftModel
        .from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
        )
    )

    model = model.to(
        "cuda"
    )

    model.eval()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{args.run_name}_dev500.jsonl"
    )

    summary_path = (
        RESULTS_DIR
        / f"{args.run_name}_dev500_summary.json"
    )

    counts = {
        "correct": 0,
        "wrong": 0,
        "format_fail": 0,
        "truncated": 0,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for index, sample in enumerate(
            dev,
            start=1,
        ):

            prompt = (
                prompt_template.format(
                    question=sample[
                        "question"
                    ]
                )
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(
                "cuda"
            )

            input_length = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            with torch.no_grad():

                outputs = (
                    model.generate(
                        **inputs,
                        max_new_tokens=(
                            MAX_NEW_TOKENS
                        ),
                        do_sample=False,
                    )
                )

            new_tokens = (
                outputs[
                    0,
                    input_length:
                ]
            )

            prediction = (
                tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
            )

            generated_tokens = len(
                new_tokens
            )

            truncated = (
                generated_tokens
                >= MAX_NEW_TOKENS
                and
                new_tokens[-1].item()
                !=
                tokenizer.eos_token_id
            )

            result = build_result(
                sample_id=sample["id"],
                prediction=prediction,
                reference=sample[
                    "final_answer"
                ],
                generated_tokens=(
                    generated_tokens
                ),
                truncated=truncated,
            )

            counts[
                result["status"]
            ] += 1

            if truncated:
                counts[
                    "truncated"
                ] += 1

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(
                f"[{index}/{len(dev)}] "
                f"id={sample['id']} "
                f"status={result['status']} "
                f"tokens={generated_tokens} "
                f"truncated={truncated}"
            )

    total = len(dev)

    strict_accuracy = (
        counts["correct"]
        / total
    )

    format_success_rate = (
        counts["correct"]
        + counts["wrong"]
    ) / total

    truncation_rate = (
        counts["truncated"]
        / total
    )

    summary = {
        "run_name":
            args.run_name,

        "adapter":
            args.adapter,

        "total":
            total,

        "correct":
            counts["correct"],

        "wrong":
            counts["wrong"],

        "format_fail":
            counts["format_fail"],

        "truncated":
            counts["truncated"],

        "strict_accuracy":
            strict_accuracy,

        "strict_accuracy_percent":
            strict_accuracy * 100,

        "format_success_rate":
            format_success_rate,

        "format_success_percent":
            format_success_rate * 100,

        "truncation_rate":
            truncation_rate,

        "truncation_percent":
            truncation_rate * 100,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "do_sample":
            False,
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

    print()
    print(
        "=============================="
    )

    print(
        "SFT DEV500 COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        "correct =",
        counts["correct"],
    )

    print(
        "wrong =",
        counts["wrong"],
    )

    print(
        "format_fail =",
        counts["format_fail"],
    )

    print(
        "truncated =",
        counts["truncated"],
    )

    print(
        "strict_accuracy =",
        f"{strict_accuracy * 100:.2f}%"
    )

    print(
        "format_success_rate =",
        f"{format_success_rate * 100:.2f}%"
    )

    print(
        "truncation_rate =",
        f"{truncation_rate * 100:.2f}%"
    )

    print(
        "saved:",
        output_path,
    )

    print(
        "saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
