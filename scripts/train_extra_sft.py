import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from datasets import Dataset

from peft import (
    LoraConfig,
    PeftModel,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import (
    SFTConfig,
    SFTTrainer,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "extra_sft_rl1000.json"
        ),
    )

    parser.add_argument(
        "--inspect-only",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def load_dataset_from_config(
    config,
):

    with open(
        config["prompt_file"],
        "r",
        encoding="utf-8",
    ) as f:

        prompt_template = (
            f.read()
        )

    rows = []

    with open(
        config["train_file"],
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            sample = json.loads(
                line
            )

            rows.append(
                {
                    "prompt":
                        prompt_template
                        .format(
                            question=
                            sample[
                                "question"
                            ]
                        ),

                    "completion":
                        sample[
                            "answer"
                        ],
                }
            )

    return Dataset.from_list(
        rows
    )


def build_lora_config(
    config,
):

    lora = config[
        "lora"
    ]

    return LoraConfig(

        r=lora["r"],

        lora_alpha=
            lora["alpha"],

        lora_dropout=
            lora["dropout"],

        bias=lora["bias"],

        task_type="CAUSAL_LM",

        target_modules=
            lora[
                "target_modules"
            ],
    )


def main():

    args = parse_args()

    config = load_json(
        args.config
    )

    train_cfg = config[
        "training"
    ]

    print(
        "=== EXTRA-SFT CONTROL ==="
    )

    print(
        "config:",
        args.config,
    )

    print(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ==================================================
    # 1. Dataset
    # ==================================================

    dataset = (
        load_dataset_from_config(
            config
        )
    )

    print(
        "\n=== DATASET ==="
    )

    print(
        "samples =",
        len(dataset),
    )

    print(
        "columns =",
        dataset.column_names,
    )

    assert len(
        dataset
    ) == 1000

    assert (
        dataset.column_names
        ==
        [
            "prompt",
            "completion",
        ]
    )

    print(
        "\nfirst completion:"
    )

    print(
        dataset[0][
            "completion"
        ]
    )

    # ==================================================
    # 2. Tokenizer
    # ==================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            config[
                "model_path"
            ],
            local_files_only=True,
        )
    )

    # ==================================================
    # 3. Load original Base
    # ==================================================

    print(
        "\n=== LOAD BASE ==="
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            config[
                "model_path"
            ],
            torch_dtype=
                torch.bfloat16,
            local_files_only=True,
        )
    )

    # ==================================================
    # 4. Load original SFT adapter
    #
    # W_base + ΔW_sft
    # ==================================================

    print(
        "\n=== LOAD ORIGINAL SFT ADAPTER ==="
    )

    sft_adapter_path = Path(
        config[
            "sft_adapter_path"
        ]
    )

    adapter_weight = (
        sft_adapter_path
        / "adapter_model.safetensors"
    )

    assert (
        adapter_weight.exists()
    )

    sft_sha256 = (
        sha256_file(
            adapter_weight
        )
    )

    print(
        "SFT adapter SHA256 =",
        sft_sha256,
    )

    sft_model = (
        PeftModel
        .from_pretrained(
            base_model,
            str(
                sft_adapter_path
            ),
            is_trainable=False,
        )
    )

    # ==================================================
    # 5. Merge SFT
    #
    # 得到和 GRPO 相同的 W_sft 起点
    # ==================================================

    print(
        "\n=== MERGE SFT ==="
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

        merged_model._hf_peft_config_loaded = (
            False
        )

    old_lora_count = sum(
        1
        for name, _
        in merged_model
        .named_parameters()
        if "lora_" in name
    )

    print(
        "LoRA tensors after merge =",
        old_lora_count,
    )

    assert (
        old_lora_count == 0
    )

    # ==================================================
    # 6. Fresh Extra-SFT LoRA
    # ==================================================

    peft_config = (
        build_lora_config(
            config
        )
    )

    print(
        "\n=== FRESH EXTRA-SFT LORA ==="
    )

    print(
        "r =",
        peft_config.r,
    )

    print(
        "alpha =",
        peft_config.lora_alpha,
    )

    print(
        "dropout =",
        peft_config.lora_dropout,
    )

    print(
        "target_modules =",
        peft_config.target_modules,
    )

    # ==================================================
    # 7. Training config
    # ==================================================

    training_args = (
        SFTConfig(

            output_dir=
                train_cfg[
                    "output_dir"
                ],

            per_device_train_batch_size=
                train_cfg[
                    "per_device_train_batch_size"
                ],

            gradient_accumulation_steps=
                train_cfg[
                    "gradient_accumulation_steps"
                ],

            learning_rate=
                train_cfg[
                    "learning_rate"
                ],

            max_steps=
                train_cfg[
                    "max_steps"
                ],

            optim=
                "adamw_torch",

            logging_steps=
                train_cfg[
                    "logging_steps"
                ],

            logging_first_step=True,

            report_to="none",

            save_strategy="steps",

            save_steps=
                train_cfg[
                    "save_steps"
                ],

            save_total_limit=2,

            save_only_model=False,

            max_length=
                config[
                    "max_length"
                ],

            completion_only_loss=
                config[
                    "completion_only_loss"
                ],

            packing=False,

            eos_token=
                tokenizer.eos_token,

            pad_token=
                tokenizer.pad_token,

            bf16=True,

            fp16=False,

            seed=
                train_cfg[
                    "seed"
                ],

            data_seed=
                train_cfg[
                    "seed"
                ],
        )
    )

    trainer = (
        SFTTrainer(

            model=merged_model,

            args=training_args,

            train_dataset=
                dataset,

            processing_class=
                tokenizer,

            peft_config=
                peft_config,
        )
    )

    print(
        "\n=== TRAINABLE PARAMETERS ==="
    )

    trainer.model.print_trainable_parameters()

    # ==================================================
    # 8. First batch supervision audit
    # ==================================================

    first_batch = next(
        iter(
            trainer
            .get_train_dataloader()
        )
    )

    labels = first_batch[
        "labels"
    ]

    masked_tokens = (
        labels == -100
    ).sum().item()

    supervised_tokens = (
        labels != -100
    ).sum().item()

    print(
        "\n=== FIRST BATCH AUDIT ==="
    )

    print(
        "input shape =",
        tuple(
            first_batch[
                "input_ids"
            ].shape
        ),
    )

    print(
        "masked tokens =",
        masked_tokens,
    )

    print(
        "supervised tokens =",
        supervised_tokens,
    )

    assert (
        masked_tokens > 0
    )

    assert (
        supervised_tokens > 0
    )

    # ==================================================
    # 9. Inspect only
    # ==================================================

    if args.inspect_only:

        print(
            "\n=============================="
        )

        print(
            "EXTRA-SFT INSPECT PASS"
        )

        print(
            "No optimizer step executed."
        )

        print(
            "=============================="
        )

        return

    # ==================================================
    # 10. Snapshot trainable params
    # ==================================================

    before = {}

    for name, param in (
        trainer.model
        .named_parameters()
    ):

        if param.requires_grad:

            before[name] = (
                param
                .detach()
                .float()
                .cpu()
                .clone()
            )

    # ==================================================
    # 11. Train
    # ==================================================

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    print(
        "\n=== EXTRA-SFT TRAIN START ==="
    )

    start_time = time.time()

    result = trainer.train()

    wall_time = (
        time.time()
        - start_time
    )

    # ==================================================
    # 12. Parameter delta
    # ==================================================

    changed_tensors = 0
    changed_values = 0
    max_abs_change = 0.0

    for name, param in (
        trainer.model
        .named_parameters()
    ):

        if name not in before:
            continue

        after = (
            param
            .detach()
            .float()
            .cpu()
        )

        delta = (
            after
            - before[name]
        ).abs()

        nonzero = (
            delta > 0
        )

        if nonzero.any():

            changed_tensors += 1

            changed_values += (
                nonzero
                .sum()
                .item()
            )

            max_abs_change = max(
                max_abs_change,
                delta.max().item(),
            )

    # ==================================================
    # 13. Save
    # ==================================================

    output_dir = Path(
        train_cfg[
            "output_dir"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_adapter_dir = (
        output_dir
        / "final_adapter"
    )

    trainer.save_model(
        str(
            final_adapter_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            final_adapter_dir
        )
    )

    metrics = dict(
        result.metrics
    )

    metrics.update(
        {
            "global_step":
                trainer.state.global_step,

            "wall_time_seconds":
                wall_time,

            "source_sft_adapter":
                str(
                    sft_adapter_path
                ),

            "source_sft_sha256":
                sft_sha256,

            "train_samples":
                len(dataset),

            "changed_tensors":
                changed_tensors,

            "changed_values":
                changed_values,

            "max_abs_change":
                max_abs_change,

            "budget":
                config[
                    "budget"
                ],
        }
    )

    if torch.cuda.is_available():

        metrics[
            "peak_gpu_allocated_gb"
        ] = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        metrics[
            "peak_gpu_reserved_gb"
        ] = (
            torch.cuda
            .max_memory_reserved()
            / 1024**3
        )

    metrics_path = (
        output_dir
        / "train_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ==================================================
    # 14. Final summary
    # ==================================================

    print(
        "\n=============================="
    )

    print(
        "EXTRA-SFT TRAIN COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        "global_step =",
        trainer.state.global_step,
    )

    print(
        "training_loss =",
        result.training_loss,
    )

    print(
        "wall_time_seconds =",
        f"{wall_time:.2f}",
    )

    print(
        "changed_tensors =",
        changed_tensors,
    )

    print(
        "changed_values =",
        changed_values,
    )

    print(
        "max_abs_change =",
        max_abs_change,
    )

    if torch.cuda.is_available():

        print(
            "peak_gpu_allocated_gb =",
            f"{metrics['peak_gpu_allocated_gb']:.2f}",
        )

        print(
            "peak_gpu_reserved_gb =",
            f"{metrics['peak_gpu_reserved_gb']:.2f}",
        )

    print(
        "metrics =",
        metrics_path,
    )

    print(
        "final_adapter =",
        final_adapter_dir,
    )


if __name__ == "__main__":
    main()
