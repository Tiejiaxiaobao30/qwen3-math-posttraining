import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluator import build_result


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"

SFT_ADAPTER_PATH = Path(
    "outputs/sft_full/final_adapter"
)

GRPO_DIR = Path(
    "outputs/grpo_full"
)

DEV_PATH = Path(
    "data/processed/dev.jsonl"
)

PROMPT_PATH = Path(
    "configs/sft_prompt.txt"
)

RESULTS_DIR = Path(
    "results"
)

REPORT_DIR = Path(
    "reports/day14"
)

CHECKPOINT_STEPS = [
    250,
    500,
    750,
    1000,
]

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


def load_policy(checkpoint_path, tokenizer):
    """
    Base
      +
    SFT LoRA
      ↓ merge
    W_sft
      +
    checkpoint GRPO LoRA
    """

    print(
        "\n=== LOAD BASE ===",
        flush=True,
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    print(
        "=== LOAD SFT ADAPTER ===",
        flush=True,
    )

    sft_model = (
        PeftModel
        .from_pretrained(
            base_model,
            str(SFT_ADAPTER_PATH),
            is_trainable=False,
        )
    )

    print(
        "=== MERGE SFT ===",
        flush=True,
    )

    merged_model = (
        sft_model
        .merge_and_unload()
    )

    # 清理 merge 后残留 PEFT metadata
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

    remaining_lora = [
        name
        for name, _
        in merged_model.named_parameters()
        if "lora_" in name
    ]

    assert len(remaining_lora) == 0

    print(
        "=== LOAD GRPO CHECKPOINT ===",
        checkpoint_path,
        flush=True,
    )

    model = (
        PeftModel
        .from_pretrained(
            merged_model,
            str(checkpoint_path),
            is_trainable=False,
        )
    )

    model = model.to(
        "cuda"
    )

    model.eval()

    grpo_lora_count = sum(
        1
        for name, _
        in model.named_parameters()
        if "lora_" in name
    )

    print(
        "GRPO LoRA tensors:",
        grpo_lora_count,
        flush=True,
    )

    assert grpo_lora_count > 0

    return model


def evaluate_checkpoint(
    model,
    tokenizer,
    dev_rows,
    prompt_template,
    step,
):

    output_path = (
        RESULTS_DIR
        / f"grpo_ckpt_{step}_dev500.jsonl"
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
            dev_rows,
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
                inputs["input_ids"]
                .shape[1]
            )

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),
                    do_sample=False,
                )

            new_tokens = outputs[
                0,
                input_length:
            ]

            prediction = (
                tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
            )

            generated_tokens = len(
                new_tokens
            )

            truncated = False

            if generated_tokens > 0:

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
            ):

                accuracy = (
                    counts["correct"]
                    / index
                    * 100
                )

                print(
                    f"[ckpt={step}] "
                    f"[{index}/500] "
                    f"correct={counts['correct']} "
                    f"wrong={counts['wrong']} "
                    f"format_fail={counts['format_fail']} "
                    f"acc={accuracy:.2f}%",
                    flush=True,
                )

    total = len(
        dev_rows
    )

    strict_accuracy = (
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
        "step": step,
        "correct": (
            counts["correct"]
        ),
        "wrong": (
            counts["wrong"]
        ),
        "format_fail": (
            counts["format_fail"]
        ),
        "truncated": (
            counts["truncated"]
        ),
        "strict_accuracy": (
            strict_accuracy
        ),
        "format_success_rate": (
            format_success
        ),
        "truncation_rate": (
            truncation_rate
        ),
    }

    print(
        "\n--------------------------------",
        flush=True,
    )

    print(
        f"CHECKPOINT {step} RESULT",
        flush=True,
    )

    print(
        "correct:",
        counts["correct"],
        flush=True,
    )

    print(
        "wrong:",
        counts["wrong"],
        flush=True,
    )

    print(
        "format_fail:",
        counts["format_fail"],
        flush=True,
    )

    print(
        "truncated:",
        counts["truncated"],
        flush=True,
    )

    print(
        f"strict accuracy: "
        f"{strict_accuracy:.2f}%",
        flush=True,
    )

    print(
        "--------------------------------",
        flush=True,
    )

    return summary


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # 1. 检查 checkpoints
    # ==================================================

    print(
        "=== CHECK CHECKPOINTS ==="
    )

    checkpoint_paths = {}

    for step in CHECKPOINT_STEPS:

        path = (
            GRPO_DIR
            / f"checkpoint-{step}"
        )

        adapter_file = (
            path
            / "adapter_model.safetensors"
        )

        config_file = (
            path
            / "adapter_config.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

        if not adapter_file.exists():
            raise FileNotFoundError(
                f"Missing adapter: {adapter_file}"
            )

        if not config_file.exists():
            raise FileNotFoundError(
                f"Missing adapter config: "
                f"{config_file}"
            )

        checkpoint_paths[
            step
        ] = path

        print(
            f"checkpoint-{step}: PASS"
        )

    # ==================================================
    # 2. dev500
    # ==================================================

    dev_rows = load_jsonl(
        DEV_PATH
    )

    assert len(dev_rows) == 500

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        prompt_template = f.read()

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    # ==================================================
    # 3. 依次评测
    # ==================================================

    summaries = []

    for step in CHECKPOINT_STEPS:

        print(
            "\n\n"
            "================================"
        )

        print(
            f"=== EVALUATE CHECKPOINT {step} ==="
        )

        print(
            "================================"
        )

        model = load_policy(
            checkpoint_paths[step],
            tokenizer,
        )

        summary = evaluate_checkpoint(
            model=model,
            tokenizer=tokenizer,
            dev_rows=dev_rows,
            prompt_template=(
                prompt_template
            ),
            step=step,
        )

        summaries.append(
            summary
        )

        # ------------------------------------------------
        # 清理显存，再评下一个 checkpoint
        # ------------------------------------------------

        del model

        gc.collect()

        torch.cuda.empty_cache()

    # ==================================================
    # 4. 汇总
    # ==================================================

    combined_path = (
        RESULTS_DIR
        / "grpo_checkpoint_ablation.json"
    )

    with open(
        combined_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    best = max(
        summaries,
        key=lambda x: (
            x["strict_accuracy"]
        ),
    )

    print(
        "\n\n"
        "================================"
    )

    print(
        "=== CHECKPOINT ABLATION RESULT ==="
    )

    print(
        "================================"
    )

    print(
        "\nSFT baseline: 78.00%"
    )

    for item in summaries:

        print(
            f"GRPO-{item['step']:4d}: "
            f"{item['strict_accuracy']:.2f}% "
            f"({item['correct']}/500)"
        )

    print()

    print(
        "BEST CHECKPOINT:",
        best["step"],
    )

    print(
        "BEST ACCURACY:",
        f"{best['strict_accuracy']:.2f}%",
    )

    print(
        "GAIN VS SFT:",
        f"{best['strict_accuracy'] - 78.0:+.2f} pp",
    )

    # ==================================================
    # 5. 自动写 Day14 markdown
    # ==================================================

    report_path = (
        REPORT_DIR
        / "grpo_checkpoint_ablation.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Day14｜GRPO Checkpoint Ablation\n\n"
        )

        f.write(
            "Only training horizon is varied; "
            "all other GRPO hyperparameters "
            "remain fixed.\n\n"
        )

        f.write(
            "| Model | Correct | "
            "Strict Accuracy | "
            "Gain vs SFT |\n"
        )

        f.write(
            "| --- | ---: | ---: | ---: |\n"
        )

        f.write(
            "| SFT | 390/500 | "
            "78.00% | 0.00 pp |\n"
        )

        for item in summaries:

            gain = (
                item[
                    "strict_accuracy"
                ]
                - 78.0
            )

            f.write(
                f"| GRPO-{item['step']} "
                f"| {item['correct']}/500 "
                f"| {item['strict_accuracy']:.2f}% "
                f"| {gain:+.2f} pp |\n"
            )

        f.write(
            "\n## Best checkpoint\n\n"
        )

        f.write(
            f"- Step: {best['step']}\n"
        )

        f.write(
            f"- Strict accuracy: "
            f"{best['strict_accuracy']:.2f}%\n"
        )

        f.write(
            f"- Gain vs SFT: "
            f"{best['strict_accuracy'] - 78.0:+.2f} pp\n"
        )

    print(
        "\nsaved:",
        combined_path,
    )

    print(
        "saved:",
        report_path,
    )


if __name__ == "__main__":
    main()
