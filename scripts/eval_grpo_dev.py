import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from scripts.evaluator import build_result


MODEL_PATH = "/mnt/workspace/models/Qwen3-1.7B-Base"

SFT_ADAPTER_PATH = Path(
    "outputs/sft_full/final_adapter"
)

GRPO_ADAPTER_PATH = Path(
    "outputs/grpo_full/final_adapter"
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

# 必须与之前 Base / SFT 正式评测完全一致
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
        "--limit",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    device = "cuda"

    print(
        "=== GRPO DEV EVALUATION ===",
        flush=True,
    )

    print(
        "device =",
        device,
        flush=True,
    )

    print(
        "max_new_tokens =",
        MAX_NEW_TOKENS,
        flush=True,
    )


    # ==================================================
    # 1. 固定 dev500
    # ==================================================

    dev = load_jsonl(
        DEV_PATH
    )[:args.limit]

    assert len(dev) == args.limit

    print(
        "dev samples =",
        len(dev),
        flush=True,
    )


    # ==================================================
    # 2. 与 Base / SFT 完全相同的 prompt
    # ==================================================

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        prompt_template = f.read()

    print(
        "prompt =",
        PROMPT_PATH,
        flush=True,
    )


    # ==================================================
    # 3. tokenizer
    # ==================================================

    print(
        "\n=== LOAD TOKENIZER ===",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )


    # ==================================================
    # 4. 原始 Base
    #
    # 当前：
    # W_base
    # ==================================================

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


    # ==================================================
    # 5. 加载 SFT LoRA
    #
    # 当前：
    # W_base + ΔW_sft
    # ==================================================

    print(
        "\n=== LOAD SFT ADAPTER ===",
        flush=True,
    )

    print(
        "sft adapter =",
        SFT_ADAPTER_PATH,
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


    # ==================================================
    # 6. merge SFT
    #
    # 得到：
    # W_sft = W_base + ΔW_sft
    # ==================================================

    print(
        "\n=== MERGE SFT INTO BASE ===",
        flush=True,
    )

    merged_model = (
        sft_model
        .merge_and_unload()
    )


    # ==================================================
    # 7. 清理 merge 后残留的 PEFT 身份信息
    #
    # SFT 权重已经真正写进主体权重。
    # 这里清掉残留 metadata，
    # 避免加载 GRPO LoRA 时被当成
    # “已有 adapter 的多 adapter 模型”。
    # ==================================================

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


    remaining_sft_lora = [
        name
        for name, _
        in merged_model.named_parameters()
        if "lora_" in name
    ]

    print(
        "LoRA tensors after SFT merge =",
        len(remaining_sft_lora),
        flush=True,
    )

    assert (
        len(remaining_sft_lora)
        == 0
    )


    # ==================================================
    # 8. 加载正式 GRPO LoRA
    #
    # 当前：
    #
    # W_sft + ΔW_grpo
    #
    # 这才是最终 GRPO policy
    # ==================================================

    print(
        "\n=== LOAD FULL GRPO ADAPTER ===",
        flush=True,
    )

    print(
        "grpo adapter =",
        GRPO_ADAPTER_PATH,
        flush=True,
    )

    model = (
        PeftModel
        .from_pretrained(
            merged_model,
            str(GRPO_ADAPTER_PATH),
            is_trainable=False,
        )
    )

    grpo_lora_names = [
        name
        for name, _
        in model.named_parameters()
        if "lora_" in name
    ]

    print(
        "GRPO LoRA tensors =",
        len(grpo_lora_names),
        flush=True,
    )

    assert (
        len(grpo_lora_names)
        > 0
    )


    # ==================================================
    # 9. 移到 GPU
    # ==================================================

    print(
        "\n=== MOVE FINAL POLICY TO CUDA ===",
        flush=True,
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "model ready.",
        flush=True,
    )


    # ==================================================
    # 10. 输出文件
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"grpo_dev_{args.limit}.jsonl"
    )


    counts = {
        "correct": 0,
        "wrong": 0,
        "format_fail": 0,
        "truncated": 0,
    }


    # ==================================================
    # 11. 正式评估
    #
    # 与之前 Base / SFT：
    #
    # do_sample=False
    # max_new_tokens=512
    # build_result()
    #
    # 完全保持一致
    # ==================================================

    print(
        "\n=== START DEV EVALUATION ===",
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
                device
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


            truncated = (
                generated_tokens
                >= MAX_NEW_TOKENS

                and

                new_tokens[-1].item()
                != tokenizer.eos_token_id
            )


            # ------------------------------------------
            # 和之前 Base / SFT 共用同一个 evaluator
            # ------------------------------------------

            result = build_result(

                sample_id=sample[
                    "id"
                ],

                prediction=(
                    prediction
                ),

                reference=sample[
                    "final_answer"
                ],

                generated_tokens=(
                    generated_tokens
                ),

                truncated=(
                    truncated
                ),
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


            # ------------------------------------------
            # 实时显示进度
            # ------------------------------------------

            print(
                f"[{index}/{len(dev)}] "
                f"id={sample['id']} "
                f"status={result['status']} "
                f"tokens={generated_tokens} "
                f"truncated={truncated}",
                flush=True,
            )


    # ==================================================
    # 12. 最终结果
    # ==================================================

    total = len(
        dev
    )


    strict_accuracy = (
        counts["correct"]
        / total
        * 100
    )


    format_success_rate = (
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


    print(
        "\n================================",
        flush=True,
    )

    print(
        "=== GRPO DEV500 RESULT ===",
        flush=True,
    )

    print(
        "================================",
        flush=True,
    )

    print(
        "saved:",
        output_path,
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
        f"{strict_accuracy:.2f}%",
        flush=True,
    )

    print(
        "format_success_rate =",
        f"{format_success_rate:.2f}%",
        flush=True,
    )

    print(
        "truncation_rate =",
        f"{truncation_rate:.2f}%",
        flush=True,
    )


    # ==================================================
    # 13. 保存 summary
    # ==================================================

    summary = {

        "model": (
            "Qwen3-1.7B-Base "
            "+ merged SFT "
            "+ GRPO LoRA"
        ),

        "total": total,

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
            format_success_rate
        ),

        "truncation_rate": (
            truncation_rate
        ),

        "max_new_tokens": (
            MAX_NEW_TOKENS
        ),

        "do_sample": False,
    }


    summary_path = (
        RESULTS_DIR
        / "grpo_dev500_summary.json"
    )


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
        "summary:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
