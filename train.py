"""SFT 或 GRPO：数据 → 模型 → LoRA → Trainer → 保存。"""
import argparse
import os
from pathlib import Path
from common import correctness_reward
from common import read, record, digest, provenance, assets, load_model

def reward(completions, gold, **kwargs):
    return [correctness_reward(text, answer) for text, answer in zip(completions, gold)]

def layout(world):
    # 12 completions/update = 3 questions × G4, consistent for 1/2/3 GPUs.
    if world not in (1, 2, 3):
        raise ValueError('Use 1, 2 or 3 GPUs on ONE machine')
    return dict(per_device_train_batch_size=1, gradient_accumulation_steps=12//world)

def build_trainer(stage, model, tokenizer, rows, options, world):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj',
                                              'gate_proj','up_proj','down_proj'])
    model.config.use_cache = False
    if stage == 'sft':
        dataset = Dataset.from_list([dict(prompt=r['prompt'], completion=r['completion']) for r in rows])
        args = SFTConfig(**options, learning_rate=2e-4, max_length=512,
                         completion_only_loss=True, packing=False, eos_token=tokenizer.eos_token,
                         pad_token=tokenizer.pad_token)
        return SFTTrainer(model=model, args=args, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=lora)
    if max(len(tokenizer(r['prompt'])['input_ids']) for r in rows) > 512:
        raise ValueError('Prompt exceeds 512 tokens')
    args = GRPOConfig(**options, learning_rate=1e-6, num_generations=4,
        steps_per_generation=12//world, max_prompt_length=512, max_completion_length=256,
        temperature=0.8, top_p=0.95, beta=0.04, loss_type='bnpo', scale_rewards='group',
        mask_truncated_completions=True, remove_unused_columns=False, use_vllm=False)
    return GRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer, peft_config=lora, reward_funcs=reward)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sft', 'grpo'])
    p.add_argument('--model', required=True, help='Full Qwen3-1.7B-Base directory')
    p.add_argument('--sft', help='SFT adapter to merge BEFORE adding a fresh GRPO LoRA')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', help='Full checkpoint-N inside output, never final_adapter')
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args()
    if a.steps < 1 or (a.stage == 'grpo' and not a.sft) or (a.stage == 'sft' and a.sft):
        p.error('Positive steps required; --sft is required ONLY for GRPO')
    out = Path(a.output)
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('Output is nonempty; choose a new run directory')
    if a.resume and (Path(a.resume).resolve().parent != out.resolve()
                     or not (Path(a.resume)/'trainer_state.json').is_file()):
        p.error('--resume must be a full checkpoint in --output')
    import torch
    import importlib.metadata as metadata
    from transformers import set_seed
    if metadata.version('trl') != '0.24.0':
        raise RuntimeError('Install teaching/requirements.txt (TRL 0.24.0)')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Cloud entry requires a BF16-capable CUDA GPU')
    set_seed(a.seed)
    rows = read(a.data)
    import json
    from common import ROOT
    split = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    allowed = set(split['splits']['sft' if a.stage == 'sft' else 'rl']['ids'])
    ids = [r['id'] for r in rows]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError('Empty, duplicate, or cross-split training IDs')
    model, tokenizer = load_model(a.model, a.sft)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    options = dict(output_dir=str(out), max_steps=a.steps, seed=a.seed, data_seed=a.seed,
        **layout(world), bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}, ddp_find_unused_parameters=False,
        logging_steps=1, save_steps=50, save_total_limit=None, report_to='none', optim='adamw_torch')
    trainer = build_trainer(a.stage, model, tokenizer, rows, options, world)
    run = dict(stage=a.stage, seed=a.seed, steps=a.steps, world=world,
        model=str(Path(a.model).resolve()), sft=a.sft, data_sha256=digest(a.data),
        assets=assets(a.model, a.sft), code=provenance(), packages={n:metadata.version(n) for n in
            ['torch','trl','transformers','peft','datasets','accelerate']})
    if a.resume:
        import json
        previous = json.loads((out/'run.json').read_text(encoding='utf-8'))
        if previous != run:
            raise ValueError('Resume configuration/source differs; start a NEW run')
    if a.inspect_only:
        print(run)
        print('Trainer constructed; no training performed')
        return
    if trainer.is_world_process_zero() and not a.resume:
        out.mkdir(parents=True, exist_ok=True)
        record(out/'run.json', **run)
    trainer.accelerator.wait_for_everyone()
    result = trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(out/'final_adapter'))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(out/'final_adapter')
        # Resume may update metrics, but never erase a previous metrics file.
        record(out/f'metrics-step{trainer.state.global_step}.json', **result.metrics)

if __name__ == '__main__':
    main()
