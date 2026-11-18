import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from scripts.reward import correctness_reward


# ============================================================
# 1. 读取 jsonl
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
        ]


# ============================================================
# 2. GRPO correctness reward
#
# completions:
#   当前 policy 在线生成的回答
#
# gold:
#   dataset 中保留下来的标准答案
#
# 返回：
#   [1.0, 0.0, ...]
# ============================================================

def strict_correctness_reward(
    completions,
    gold,
    **kwargs,
):
    rewards = []

    for completion, reference in zip(
        completions,
        gold,
    ):
        reward = correctness_reward(
            completion,
            reference,
        )

        rewards.append(reward)

    return rewards


# ============================================================
# 3. 梯度审计
#
# optimizer.step() 之前检查：
# 当前可训练 GRPO LoRA 是否真的出现非零梯度
# ============================================================

class GradAuditCallback(TrainerCallback):

    def __init__(self):
        self.records = []

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        grad_tensors = 0
        nonzero_grad_tensors = 0
        max_abs_grad = 0.0

        if model is not None:

            for name, param in model.named_parameters():

                if not param.requires_grad:
                    continue

                if param.grad is None:
                    continue

                grad_tensors += 1

                local_max = (
                    param.grad
                    .detach()
                    .abs()
                    .max()
                    .item()
                )

                if local_max > 0:
                    nonzero_grad_tensors += 1

                max_abs_grad = max(
                    max_abs_grad,
                    local_max,
                )

        record = {
            "optimizer_step": int(
                state.global_step + 1
            ),
            "grad_tensors": grad_tensors,
            "nonzero_grad_tensors": (
                nonzero_grad_tensors
            ),
            "max_abs_grad": max_abs_grad,
        }

        self.records.append(record)

        print(
            "\n[GRAD AUDIT]",
            record,
        )


# ============================================================
# 4. 保存训练前所有可训练参数
#
# 当前只有新的 GRPO LoRA requires_grad=True
# ============================================================

def snapshot_trainable(model):

    snapshot = {}

    for name, param in model.named_parameters():

        if param.requires_grad:

            snapshot[name] = (
                param
                .detach()
                .cpu()
                .clone()
            )

    return snapshot


# ============================================================
# 5. 比较训练前后参数
# ============================================================

def compare_trainable(
    model,
    before,
):
    changed_tensors = 0
    changed_values = 0
    max_abs_change = 0.0

    for name, param in model.named_parameters():

        if name not in before:
            continue

        after = (
            param
            .detach()
            .cpu()
        )

        diff = (
            after
            - before[name]
        ).abs()

        local_max = diff.max().item()

        if local_max > 0:
            changed_tensors += 1

        changed_values += int(
            torch.count_nonzero(
                diff
            ).item()
        )

        max_abs_change = max(
            max_abs_change,
            local_max,
        )

    return {
        "changed_tensors": changed_tensors,
        "changed_values": changed_values,
        "max_abs_change": max_abs_change,
    }


# ============================================================
# 6. main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/grpo_smoke.json",
    )

    parser.add_argument(
        "--inspect-only",
        action="store_true",
    )

    cli_args = parser.parse_args()


    # ========================================================
    # A. 读取配置
    # ========================================================

    with open(
        cli_args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    model_path = config[
        "model_path"
    ]

    sft_adapter_path = config[
        "sft_adapter_path"
    ]

    output_dir = config[
        "output_dir"
    ]


    # ========================================================
    # B. RL Dataset
    #
    # 每条：
    #
    # prompt -> 给模型
    # gold   -> 给 reward
    # ========================================================

    rows = load_jsonl(
        config["train_file"]
    )

    assert len(rows) == 1000

    train_dataset = Dataset.from_list(
        [
            {
                "prompt": row[
                    "prompt"
                ],
                "gold": str(
                    row["gold"]
                ),
            }
            for row in rows
        ]
    )

    print("\n=== DATASET ===")

    print(train_dataset)

    print(
        "columns:",
        train_dataset.column_names,
    )

    print(
        "first gold:",
        train_dataset[0]["gold"],
    )


    # ========================================================
    # C. Tokenizer
    # ========================================================

    print("\n=== TOKENIZER ===")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
        )
    )

    # GRPO 多 prompt batch 生成时使用 left padding
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "pad token:",
        tokenizer.pad_token,
    )

    print(
        "eos token:",
        tokenizer.eos_token,
    )


    # ========================================================
    # D. 加载 Qwen Base
    #
    # 当前：
    # W_base
    # ========================================================

    print("\n=== LOAD BASE ===")

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    print(
        "base type:",
        type(base_model).__name__,
    )


    # ========================================================
    # E. 挂 Day07 SFT adapter
    #
    # 当前：
    # W_base + ΔW_sft
    # ========================================================

    print(
        "\n=== LOAD SFT ADAPTER ==="
    )

    print(
        "adapter:",
        sft_adapter_path,
    )

    sft_model = (
        PeftModel
        .from_pretrained(
            base_model,
            sft_adapter_path,
            is_trainable=False,
        )
    )

    print(
        "SFT model type:",
        type(sft_model).__name__,
    )


    # ========================================================
    # F. merge SFT LoRA
    #
    # W_sft =
    # W_base + ΔW_sft
    #
    # merge 后：
    # 旧 SFT LoRA 外壳消失
    # 但 SFT 能力留在模型主体里
    # ========================================================

    print(
        "\n=== MERGE SFT INTO BASE ==="
    )

    model = (
        sft_model
        .merge_and_unload()
    )

    old_lora_names = [
        name
        for name, _
        in model.named_parameters()
        if "lora_" in name
    ]

    print(
        "LoRA tensors after SFT merge:",
        len(old_lora_names),
    )

    assert len(old_lora_names) == 0


    # ========================================================
    # G. 创建全新的 GRPO LoRA 配置
    #
    # 接下来要学的是：
    # ΔW_grpo
    # ========================================================

    print(
        "\n=== BUILD NEW GRPO LORA ==="
    )

    lora_cfg = config[
        "grpo_lora"
    ]

    grpo_lora_config = LoraConfig(

        r=lora_cfg["r"],

        lora_alpha=lora_cfg[
            "alpha"
        ],

        lora_dropout=lora_cfg[
            "dropout"
        ],

        bias=lora_cfg[
            "bias"
        ],

        target_modules=lora_cfg[
            "target_modules"
        ],

        task_type=(
            TaskType.CAUSAL_LM
        ),
    )

    print(
        "GRPO LoRA r:",
        grpo_lora_config.r,
    )

    print(
        "GRPO LoRA alpha:",
        grpo_lora_config.lora_alpha,
    )


    # ========================================================
    # H. 读取 GRPO 参数
    # ========================================================

    t = config[
        "training"
    ]

    g = config[
        "generation"
    ]

    r = config[
        "grpo"
    ]


    # ========================================================
    # I. GRPOConfig
    # ========================================================

    training_args = GRPOConfig(

        output_dir=output_dir,

        # ----------------------
        # 只跑 10 个 update
        # ----------------------

        max_steps=t[
            "max_steps"
        ],

        per_device_train_batch_size=t[
            "per_device_train_batch_size"
        ],

        gradient_accumulation_steps=t[
            "gradient_accumulation_steps"
        ],

        learning_rate=t[
            "learning_rate"
        ],

        optim="adamw_torch",

        # ----------------------
        # log / checkpoint
        # ----------------------

        logging_steps=t[
            "logging_steps"
        ],

        logging_first_step=True,

        save_strategy="steps",

        save_steps=t[
            "save_steps"
        ],

        save_total_limit=t[
            "save_total_limit"
        ],

        report_to="none",

        # ----------------------
        # precision
        # ----------------------

        bf16=True,

        fp16=False,

        # ----------------------
        # reproducibility
        # ----------------------

        seed=t["seed"],

        data_seed=t[
            "seed"
        ],

        # gold 必须保留下来
        # 给 reward function 使用
        remove_unused_columns=False,

        # ----------------------
        # rollout
        # ----------------------

        num_generations=g[
            "num_generations"
        ],

        temperature=g[
            "temperature"
        ],

        top_p=g[
            "top_p"
        ],

        max_completion_length=g[
            "max_completion_length"
        ],

        # ----------------------
        # GRPO
        # ----------------------

        beta=r[
            "beta"
        ],

        num_iterations=r[
            "num_iterations"
        ],

        loss_type=r[
            "loss_type"
        ],

        mask_truncated_completions=r[
            "mask_truncated_completions"
        ],

        scale_rewards=r[
            "scale_rewards"
        ],
    )


    # ========================================================
    # J. 创建梯度审计 callback
    # ========================================================

    grad_audit = (
        GradAuditCallback()
    )


    # ========================================================
    # K. GRPOTrainer
    #
    # 在 W_sft 上真正挂新的 ΔW_grpo
    # ========================================================

    print(
        "\n=== BUILD GRPO TRAINER ==="
    )

    trainer = GRPOTrainer(

        model=model,

        reward_funcs=(
            strict_correctness_reward
        ),

        args=training_args,

        train_dataset=(
            train_dataset
        ),

        processing_class=(
            tokenizer
        ),

        peft_config=(
            grpo_lora_config
        ),

        callbacks=[
            grad_audit
        ],
    )


    # ========================================================
    # L. 检查可训练参数
    # ========================================================

    print(
        "\n=== TRAINABLE PARAMETERS ==="
    )

    trainer.model.print_trainable_parameters()

    trainable_names = [
        name
        for name, param
        in trainer.model.named_parameters()
        if param.requires_grad
    ]

    print(
        "trainable tensor count:",
        len(trainable_names),
    )

    print(
        "\nfirst 12 trainable tensors:"
    )

    for name in trainable_names[:12]:
        print(name)


    # 所有可训练参数必须来自 LoRA
    non_lora_trainable = [
        name
        for name in trainable_names
        if "lora_" not in name
    ]

    print(
        "\nnon-LoRA trainable tensors:",
        len(non_lora_trainable),
    )

    assert (
        len(non_lora_trainable)
        == 0
    )


    # ========================================================
    # M. 打印最终实际配置
    # ========================================================

    print(
        "\n=== RESOLVED GRPO CONFIG ==="
    )

    config_names = [

        "max_steps",

        "num_generations",

        "temperature",

        "top_p",

        "max_completion_length",

        "beta",

        "num_iterations",

        "loss_type",

        "scale_rewards",

        "mask_truncated_completions",

        "remove_unused_columns",
    ]

    for name in config_names:

        print(
            name,
            "=",
            getattr(
                trainer.args,
                name,
            ),
        )


    # ========================================================
    # N. inspect-only
    #
    # 到这里为止：
    # 完全没有 optimizer.step
    # ========================================================

    if cli_args.inspect_only:

        print(
            "\nINSPECT ONLY PASS"
        )

        print(
            "Trainer built successfully."
        )

        print(
            "No optimizer step executed."
        )

        return


    # ========================================================
    # O. 训练前参数快照
    #
    # 拍下 ΔW_grpo 当前状态
    # ========================================================

    print(
        "\n=== SNAPSHOT BEFORE TRAINING ==="
    )

    before = snapshot_trainable(
        trainer.model
    )

    print(
        "snapshotted trainable tensors:",
        len(before),
    )


    # ========================================================
    # P. 清空 GPU 峰值统计
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()


    # ========================================================
    # Q. 真正开始 GRPO
    #
    # prompt
    # ↓
    # online rollout
    # ↓
    # reward
    # ↓
    # advantage
    # ↓
    # BNPO / KL loss
    # ↓
    # backward
    # ↓
    # optimizer.step
    # ========================================================

    print(
        "\n=== START 10-STEP GRPO ==="
    )

    train_result = (
        trainer.train()
    )


    # ========================================================
    # R. 权重变化检查
    # ========================================================

    print(
        "\n=== WEIGHT CHANGE AUDIT ==="
    )

    weight_change = (
        compare_trainable(
            trainer.model,
            before,
        )
    )

    print(
        json.dumps(
            weight_change,
            indent=2,
        )
    )


    # ========================================================
    # S. 梯度审计
    # ========================================================

    print(
        "\n=== GRADIENT AUDIT SUMMARY ==="
    )

    nonzero_gradient_steps = sum(

        record[
            "nonzero_grad_tensors"
        ] > 0

        for record
        in grad_audit.records
    )

    print(
        "recorded optimizer steps:",
        len(
            grad_audit.records
        ),
    )

    print(
        "steps with nonzero gradients:",
        nonzero_gradient_steps,
    )


    # ========================================================
    # T. 检查 NaN
    # ========================================================

    has_nan = False

    for record in (
        trainer.state.log_history
    ):

        for key, value in (
            record.items()
        ):

            if isinstance(
                value,
                float,
            ):

                if (
                    torch.isnan(
                        torch.tensor(
                            value
                        )
                    )
                    .item()
                ):

                    has_nan = True

    print(
        "NaN detected:",
        has_nan,
    )


    # ========================================================
    # U. 保存最终 GRPO adapter
    # ========================================================

    final_adapter_dir = (
        Path(output_dir)
        / "final_adapter"
    )

    final_adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.model.save_pretrained(
        str(
            final_adapter_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            final_adapter_dir
        )
    )


    # ========================================================
    # V. GPU memory
    # ========================================================

    peak_allocated_gb = None
    peak_reserved_gb = None

    if torch.cuda.is_available():

        peak_allocated_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        peak_reserved_gb = (
            torch.cuda
            .max_memory_reserved()
            / 1024**3
        )


    # ========================================================
    # W. 保存实验指标
    # ========================================================

    metrics = {

        "global_step": (
            trainer.state.global_step
        ),

        "train_metrics": (
            train_result.metrics
        ),

        "weight_change": (
            weight_change
        ),

        "gradient_audit": (
            grad_audit.records
        ),

        "nonzero_gradient_steps": (
            nonzero_gradient_steps
        ),

        "nan_detected": (
            has_nan
        ),

        "peak_gpu_allocated_gb": (
            peak_allocated_gb
        ),

        "peak_gpu_reserved_gb": (
            peak_reserved_gb
        ),

        "resolved_config": {

            "max_steps": (
                trainer.args.max_steps
            ),

            "num_generations": (
                trainer.args.num_generations
            ),

            "temperature": (
                trainer.args.temperature
            ),

            "top_p": (
                trainer.args.top_p
            ),

            "max_completion_length": (
                trainer.args
                .max_completion_length
            ),

            "beta": (
                trainer.args.beta
            ),

            "num_iterations": (
                trainer.args
                .num_iterations
            ),

            "loss_type": (
                trainer.args.loss_type
            ),
        },

        "log_history": (
            trainer.state.log_history
        ),
    }


    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        Path(output_dir)
        / "grpo_smoke_metrics.json"
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


    # ========================================================
    # X. 最终验收输出
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "=== GRPO SMOKE DONE ==="
    )

    print(
        "================================"
    )

    print(
        "global_step:",
        trainer.state.global_step,
    )

    print(
        "nonzero_gradient_steps:",
        nonzero_gradient_steps,
    )

    print(
        "changed_tensors:",
        weight_change[
            "changed_tensors"
        ],
    )

    print(
        "changed_values:",
        weight_change[
            "changed_values"
        ],
    )

    print(
        "max_abs_change:",
        weight_change[
            "max_abs_change"
        ],
    )

    print(
        "NaN detected:",
        has_nan,
    )

    print(
        "peak allocated GB:",
        peak_allocated_gb,
    )

    print(
        "peak reserved GB:",
        peak_reserved_gb,
    )

    print(
        "saved adapter:",
        final_adapter_dir,
    )

    print(
        "saved metrics:",
        metrics_path,
    )


if __name__ == "__main__":
    main()
