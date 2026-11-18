import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG_PATH = "configs/sft_smoke.json"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        local_files_only=True,
    )

    # =========================================================
    # 1. REAL SFT SAMPLE
    # =========================================================

    with open(config["train_file"], "r") as f:
        sample = json.loads(f.readline())

    with open(config["prompt_file"], "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        question=sample["question"]
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        sample["answer"],
        add_special_tokens=False,
    )["input_ids"]

    input_ids_list = (
        prompt_ids
        + answer_ids
        + [tokenizer.eos_token_id]
    )

    labels_list = (
        [-100] * len(prompt_ids)
        + answer_ids
        + [tokenizer.eos_token_id]
    )

    # =========================================================
    # 2. MASK AUDIT
    # =========================================================

    assert len(input_ids_list) == len(labels_list)
    assert len(input_ids_list) <= config["max_length"]

    assert all(
        x == -100
        for x in labels_list[:len(prompt_ids)]
    )

    assert any(
        x != -100
        for x in labels_list[len(prompt_ids):]
    )

    print("===== MASK AUDIT =====")
    print("sample_id =", sample["id"])
    print("max_length =", config["max_length"])
    print("prompt_tokens =", len(prompt_ids))
    print("answer_tokens =", len(answer_ids))
    print("eos_tokens = 1")
    print("total_tokens =", len(input_ids_list))

    print(
        "masked_prompt_tokens =",
        sum(
            x == -100
            for x in labels_list[:len(prompt_ids)]
        ),
    )

    print(
        "supervised_tokens =",
        sum(
            x != -100
            for x in labels_list
        ),
    )

    print("eos_token_id =", tokenizer.eos_token_id)
    print("pad_token_id =", tokenizer.pad_token_id)

    print(
        "eos_is_supervised =",
        labels_list[-1] == tokenizer.eos_token_id,
    )

    print()
    print("boundary:")

    for pos in range(
        len(prompt_ids) - 3,
        len(prompt_ids) + 5,
    ):
        token = tokenizer.decode(
            [input_ids_list[pos]]
        )

        print(
            f"pos={pos}",
            f"token={repr(token)}",
            f"label={labels_list[pos]}",
        )

    print()
    print("MASK AUDIT PASSED")

    # =========================================================
    # 3. MODEL + LORA
    # =========================================================

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    lora_cfg = config["lora"]

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    trainable_percent = (
        trainable_params / total_params * 100
    )

    q_proj = (
        model
        .base_model
        .model
        .model
        .layers[0]
        .self_attn
        .q_proj
    )

    W = q_proj.base_layer.weight
    A = q_proj.lora_A["default"].weight
    B = q_proj.lora_B["default"].weight

    assert W.requires_grad is False
    assert A.requires_grad is True
    assert B.requires_grad is True

    assert all(
        "lora_" in name
        for name in trainable_names
    )

    print()
    print("===== TRAINABLE PARAMETER AUDIT =====")

    print("total_params =", total_params)
    print("trainable_params =", trainable_params)

    print(
        "trainable_percent =",
        f"{trainable_percent:.4f}%",
    )

    print(
        "all_trainable_are_lora =",
        True,
    )

    print()
    print("target_modules =")

    for name in lora_cfg["target_modules"]:
        print(" ", name)

    print()
    print(
        "Base W shape =",
        tuple(W.shape),
        "requires_grad =",
        W.requires_grad,
    )

    print(
        "LoRA A shape =",
        tuple(A.shape),
        "requires_grad =",
        A.requires_grad,
    )

    print(
        "LoRA B shape =",
        tuple(B.shape),
        "requires_grad =",
        B.requires_grad,
    )

    print()
    print("TRAINABLE PARAMETER AUDIT PASSED")

    # =========================================================
    # 4. REAL FORWARD + BACKWARD
    # =========================================================

    input_ids = torch.tensor(
        [input_ids_list],
        device="cuda",
    )

    labels = torch.tensor(
        [labels_list],
        device="cuda",
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    model.train()
    model.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    loss.backward()

    assert W.grad is None
    assert A.grad is not None
    assert B.grad is not None

    print()
    print("===== FORWARD / BACKWARD AUDIT =====")

    print("loss =", loss.item())

    print(
        "Base W grad is None =",
        W.grad is None,
    )

    print(
        "LoRA A grad is None =",
        A.grad is None,
    )

    print(
        "LoRA B grad is None =",
        B.grad is None,
    )

    print(
        "LoRA A grad norm =",
        A.grad.float().norm().item(),
    )

    print(
        "LoRA B grad norm =",
        B.grad.float().norm().item(),
    )

    print()
    print("FORWARD / BACKWARD AUDIT PASSED")

    print()
    print("==============================")
    print("ALL SFT SETUP AUDITS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()
