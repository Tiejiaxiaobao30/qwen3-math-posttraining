
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from scripts.evaluator import build_result


MAX_NEW_TOKENS = 512


def sha256_file(path):
    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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
        "--mode",
        choices=[
            "base",
            "sft",
            "grpo",
        ],
        required=True,
    )

    parser.add_argument(
        "--base-model",
        required=True,
    )

    parser.add_argument(
        "--sft-adapter",
        default=None,
    )

    parser.add_argument(
        "--grpo-adapter",
        default=None,
    )

    parser.add_argument(
        "--test-file",
        required=True,
    )

    parser.add_argument(
        "--prompt-file",
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1319,
    )

    parser.add_argument(
        "--results-dir",
        default="results/official_test",
    )

    args = parser.parse_args()

    # ========================================================
    # 1. 参数审计
    # ========================================================

    print("=" * 70)
    print("OFFICIAL TEST EVALUATOR")
    print("=" * 70)

    print("mode =", args.mode)
    print("run_name =", args.run_name)
    print("device =", args.device)
    print("base_model =", args.base_model)
    print("sft_adapter =", args.sft_adapter)
    print("grpo_adapter =", args.grpo_adapter)

    if args.mode == "sft":
        assert args.sft_adapter is not None

    if args.mode == "grpo":
        assert args.sft_adapter is not None
        assert args.grpo_adapter is not None

    # ========================================================
    # 2. 固定 official test
    # ========================================================

    test_path = Path(
        args.test_file
    )

    prompt_path = Path(
        args.prompt_file
    )

    assert test_path.exists(), test_path
    assert prompt_path.exists(), prompt_path

    test_rows = load_jsonl(
        test_path
    )

    assert len(test_rows) == 1319, (
        f"Expected 1319 test examples, "
        f"got {len(test_rows)}"
    )

    test_rows = test_rows[
        :args.limit
    ]

    with open(
        prompt_path,
        "r",
        encoding="utf-8",
    ) as f:
        prompt_template = f.read()

    print()
    print("test examples =", len(test_rows))
    print(
        "test sha256 =",
        sha256_file(test_path),
    )

    print(
        "prompt sha256 =",
        sha256_file(prompt_path),
    )

    print(
        "do_sample = False"
    )

    print(
        "max_new_tokens =",
        MAX_NEW_TOKENS,
    )

    # ========================================================
    # 3. Device
    # ========================================================

    if args.device.startswith(
        "cuda:"
    ):
        gpu_id = int(
            args.device.split(":")[1]
        )

        torch.cuda.set_device(
            gpu_id
        )

    # ========================================================
    # 4. Tokenizer
    # ========================================================

    print(
        "\n=== LOAD TOKENIZER ==="
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.base_model,
            local_files_only=True,
        )
    )

    # ========================================================
    # 5. Base
    # ========================================================

    print(
        "\n=== LOAD BASE ==="
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    # ========================================================
    # 6. SFT
    # ========================================================

    if args.mode in {
        "sft",
        "grpo",
    }:

        print(
            "\n=== LOAD SFT ADAPTER ==="
        )

        print(
            args.sft_adapter
        )

        model = (
            PeftModel
            .from_pretrained(
                model,
                args.sft_adapter,
                is_trainable=False,
            )
        )

    # ========================================================
    # 7. GRPO
    # ========================================================

    if args.mode == "grpo":

        print(
            "\n=== MERGE SFT ==="
        )

        model = (
            model
            .merge_and_unload()
        )

        if hasattr(
            model,
            "peft_config",
        ):
            delattr(
                model,
                "peft_config",
            )

        if hasattr(
            model,
            "_hf_peft_config_loaded",
        ):
            model._hf_peft_config_loaded = (
                False
            )

        print(
            "\n=== LOAD GRPO ADAPTER ==="
        )

        print(
            args.grpo_adapter
        )

        model = (
            PeftModel
            .from_pretrained(
                model,
                args.grpo_adapter,
                is_trainable=False,
            )
        )

        lora_count = sum(
            1
            for name, _
            in model.named_parameters()
            if "lora_" in name
        )

        print(
            "GRPO LoRA tensors =",
            lora_count,
        )

        assert lora_count > 0

    # ========================================================
    # 8. GPU
    # ========================================================

    model = model.to(
        args.device
    )

    model.eval()

    # ========================================================
    # 9. Output
    # ========================================================

    results_dir = Path(
        args.results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / f"{args.run_name}_test1319.jsonl"
    )

    summary_path = (
        results_dir
        / f"{args.run_name}_test1319_summary.json"
    )

    counts = {
        "correct": 0,
        "wrong": 0,
        "format_fail": 0,
        "truncated": 0,
    }

    # ========================================================
    # 10. Official Test
    # ========================================================

    print()
    print("=" * 70)
    print("START OFFICIAL TEST")
    print("=" * 70)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for index, sample in enumerate(
            test_rows,
            start=1,
        ):

            prompt = (
                prompt_template.format(
                    question=
                    sample["question"]
                )
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(
                args.device
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
                != tokenizer.eos_token_id
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

            if (
                index == 1
                or index % 25 == 0
                or index == len(
                    test_rows
                )
            ):

                current_acc = (
                    counts["correct"]
                    / index
                    * 100
                )

                print(
                    f"[{index}/"
                    f"{len(test_rows)}] "
                    f"correct="
                    f"{counts['correct']} "
                    f"wrong="
                    f"{counts['wrong']} "
                    f"format_fail="
                    f"{counts['format_fail']} "
                    f"acc="
                    f"{current_acc:.2f}%",
                    flush=True,
                )

    # ========================================================
    # 11. Summary
    # ========================================================

    total = len(
        test_rows
    )

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

        "mode":
            args.mode,

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

        "do_sample":
            False,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "test_sha256":
            sha256_file(
                test_path
            ),

        "prompt_sha256":
            sha256_file(
                prompt_path
            ),

        "base_model":
            args.base_model,

        "sft_adapter":
            args.sft_adapter,

        "grpo_adapter":
            args.grpo_adapter,
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

    # ========================================================
    # 12. Final
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"{args.run_name} "
        "OFFICIAL TEST RESULT"
    )
    print("=" * 70)

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
