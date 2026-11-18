import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_smoke.json",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def load_train_dataset(config):
    with open(config["prompt_file"], "r") as f:
        prompt_template = f.read()

    rows = []

    with open(config["train_file"], "r") as f:
        for line in f:
            sample = json.loads(line)

            rows.append({
                "prompt": prompt_template.format(
                    question=sample["question"]
                ),
                "completion": sample["answer"],
            })

    return Dataset.from_list(rows)


def build_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        local_files_only=True,
    )

    return tokenizer


def build_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    return model


def build_lora_config(config):
    lora = config["lora"]

    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora["target_modules"],
    )


def build_training_args(config, tokenizer):
    train = config["training"]

    return SFTConfig(
        # ----- output -----
        output_dir=train["output_dir"],

        # ----- batch -----
        per_device_train_batch_size=
            train["per_device_train_batch_size"],

        gradient_accumulation_steps=
            train["gradient_accumulation_steps"],

        # ----- optimization -----
        learning_rate=train["learning_rate"],
        max_steps=train["max_steps"],
        optim="adamw_torch",

        # ----- logging -----
        logging_steps=train["logging_steps"],
        logging_first_step=True,
        report_to="none",

        # ----- checkpoint -----
        save_strategy="steps",
        save_steps=train["save_steps"],
        save_total_limit=2,
        save_only_model=False,

        # ----- SFT data -----
        max_length=config["max_length"],
        completion_only_loss=
            config["completion_only_loss"],
        packing=False,

        # ----- tokens -----
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,

        # ----- precision -----
        bf16=True,
        fp16=False,

        # ----- reproducibility -----
        seed=train["seed"],
        data_seed=train["seed"],
    )


def main():
    cli_args = parse_args()

    # =========================================================
    # 1. CONFIG
    # =========================================================

    config = load_config(cli_args.config)
    train_cfg = config["training"]

    print("===== CONFIG =====")
    print("config =", cli_args.config)
    print(
        json.dumps(
            train_cfg,
            indent=2,
        )
    )

    # =========================================================
    # 2. DATASET
    # =========================================================

    dataset = load_train_dataset(config)

    print("\n===== DATASET =====")
    print("samples =", len(dataset))
    print("columns =", dataset.column_names)

    assert len(dataset) == 2000
    assert dataset.column_names == [
        "prompt",
        "completion",
    ]

    # =========================================================
    # 3. TOKENIZER
    # =========================================================

    tokenizer = build_tokenizer(config)

    print("\n===== TOKENIZER =====")
    print("eos_token =", repr(tokenizer.eos_token))
    print("eos_token_id =", tokenizer.eos_token_id)
    print("pad_token =", repr(tokenizer.pad_token))
    print("pad_token_id =", tokenizer.pad_token_id)

    # =========================================================
    # 4. BASE MODEL
    # =========================================================

    print("\n===== LOAD BASE MODEL =====")

    model = build_model(config)

    base_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("base_params =", base_params)

    # =========================================================
    # 5. LORA
    # =========================================================

    peft_config = build_lora_config(config)

    print("\n===== LORA =====")
    print("r =", peft_config.r)
    print("alpha =", peft_config.lora_alpha)
    print(
        "target_modules =",
        peft_config.target_modules,
    )

    # =========================================================
    # 6. SFT CONFIG
    # =========================================================

    training_args = build_training_args(
        config,
        tokenizer,
    )

    # =========================================================
    # 7. TRAINER
    # =========================================================

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n===== TRAINABLE PARAMETERS =====")
    trainer.model.print_trainable_parameters()

    # =========================================================
    # 8. PRE-TRAIN CHECK
    # =========================================================

    first_batch = next(
        iter(trainer.get_train_dataloader())
    )

    labels = first_batch["labels"]

    masked_tokens = (
        labels == -100
    ).sum().item()

    supervised_tokens = (
        labels != -100
    ).sum().item()

    print("\n===== FIRST BATCH CHECK =====")
    print(
        "input_ids shape =",
        tuple(first_batch["input_ids"].shape),
    )
    print(
        "labels shape =",
        tuple(labels.shape),
    )
    print(
        "masked_tokens =",
        masked_tokens,
    )
    print(
        "supervised_tokens =",
        supervised_tokens,
    )

    assert supervised_tokens > 0
    assert masked_tokens > 0

    # =========================================================
    # 9. TRAIN
    # =========================================================

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n===== TRAIN START =====")

    print(
        "resume_from_checkpoint =",
        cli_args.resume_from_checkpoint,
    )

    result = trainer.train(
        resume_from_checkpoint=
            cli_args.resume_from_checkpoint
    )

    # =========================================================
    # 10. METRICS
    # =========================================================

    metrics = dict(result.metrics)

    metrics["global_step"] = (
        trainer.state.global_step
    )

    metrics["config"] = cli_args.config

    metrics["resume_from_checkpoint"] = (
        cli_args.resume_from_checkpoint
    )

    if torch.cuda.is_available():
        metrics["peak_gpu_allocated_gb"] = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        metrics["peak_gpu_reserved_gb"] = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

    output_dir = Path(
        train_cfg["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "train_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # =========================================================
    # 11. SAVE FINAL ADAPTER
    # =========================================================

    final_adapter_dir = (
        output_dir
        / "final_adapter"
    )

    trainer.save_model(
        str(final_adapter_dir)
    )

    tokenizer.save_pretrained(
        str(final_adapter_dir)
    )

    # =========================================================
    # 12. SUMMARY
    # =========================================================

    print("\n===== TRAIN COMPLETE =====")
    print(
        "global_step =",
        trainer.state.global_step,
    )

    print(
        "training_loss =",
        result.training_loss,
    )

    print(
        "metrics_file =",
        metrics_path,
    )

    print(
        "final_adapter =",
        final_adapter_dir,
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


if __name__ == "__main__":
    main()
