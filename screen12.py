import json

import torch
from transformers import GenerationConfig, set_seed

import train_grpo_experiment as exp

BATCH_SIZE = 12
original_record = exp.record


def record(path, **fields):
    if path.name == "run.json" and "sampling" in fields:
        fields["sampling"].update(
            prompt_batch_size=BATCH_SIZE,
            per_question_seed=None,
            seed_rule="(seed + first question ID in each batch) % 2**32",
        )
    return original_record(path, **fields)


@torch.inference_mode()
def screening_groups(model, tokenizer, rows, seed, stream):
    cfg = exp.SAMPLING
    eos = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    decoding = GenerationConfig(
        do_sample=True,
        num_return_sequences=cfg["num_generations"],
        max_new_tokens=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
    )

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        set_seed((seed + batch[0]["id"]) % 2**32)
        inputs = tokenizer(
            [row["prompt"] for row in batch], padding=True, return_tensors="pt"
        ).to(model.device)
        generated = model.generate(**inputs, generation_config=decoding)
        tokens = generated[:, inputs["input_ids"].shape[1]:].tolist()
        count = cfg["num_generations"]

        for i, row in enumerate(batch):
            outputs = []
            for ids in tokens[i * count:(i + 1) * count]:
                ended = eos in ids
                if ended:
                    ids = ids[:ids.index(eos) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                outputs.append((text, len(ids), not ended and len(ids) >= cfg["max_completion_length"]))

            texts = [output[0] for output in outputs]
            group = dict(
                id=row["id"],
                outputs=outputs,
                rewards=exp.strict_reward(texts, [row["gold"]] * count),
                format_fail=sum(1 - exp.format_ok(text) for text in texts),
                truncated=sum(output[2] for output in outputs),
            )
            stream.write(json.dumps(group, ensure_ascii=False) + "\n")
            yield group

        stream.flush()
        print(f"已筛 {start + len(batch)}/{len(rows)} 道题", flush=True)


exp.record = record
exp.screening_groups = screening_groups
exp.main()
