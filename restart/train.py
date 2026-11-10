"""Base + frozen/merged SFT + fresh GRPO LoRA; one arm per run."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from restart.common import read_jsonl, runtime_layout, sha256, verify_assets, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--sft-adapter', required=True)
    p.add_argument('--arm', choices=['strict', 'format_aux'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', help='Full Trainer checkpoint-N, not final_adapter')
    p.add_argument('--inspect-only', action='store_true', help='Build model/trainer; no sampling or optimizer step')
    p.add_argument('--swanlab-mode', choices=['cloud', 'offline', 'disabled'], default='cloud')
    a = p.parse_args()
    if a.steps is not None and a.steps <= 0:
        p.error('--steps must be positive')
    if importlib.metadata.version('trl') != '0.24.0':
        raise RuntimeError('Install restart/requirements.txt; private telemetry hook targets TRL 0.24.0')
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from restart.rewards import strict_correctness, strict_format

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required; inspect GPU/driver first')
    set_seed(a.seed)  # before initializing the new LoRA
    cfg = json.loads(Path('restart/configs/experiment.json').read_text(encoding='utf-8'))
    asset_manifest = verify_assets(a.model_dir, a.sft_adapter)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    layout = runtime_layout(world, cfg['global_completion_batch'], cfg['num_generations'])
    output = Path(a.output)
    if not a.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Nonempty output; choose a new run directory or --resume checkpoint-N')
    data_path = Path('data/restart/rl.jsonl')
    data_manifest = json.loads(Path('data/restart/manifest.json').read_text(encoding='utf-8'))
    if sha256(data_path) != data_manifest['files']['rl']['sha256']:
        raise ValueError('Training data hash changed')
    rows = read_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if max(len(tokenizer(r['prompt'])['input_ids']) for r in rows) > cfg['max_prompt_length']:
        raise ValueError('Prompt would be truncated; audit rather than silently truncate')
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                local_files_only=True, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, a.sft_adapter, is_trainable=False).merge_and_unload()
    model.config.use_cache = False
    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], target_modules=cfg['target_modules'],
        bias='none', task_type='CAUSAL_LM')
    args = GRPOConfig(output_dir=str(output), max_steps=a.steps or cfg['max_steps'],
        **layout, learning_rate=cfg['learning_rate'], seed=a.seed, data_seed=a.seed,
        num_generations=cfg['num_generations'], max_prompt_length=cfg['max_prompt_length'],
        max_completion_length=cfg['max_completion_length'], temperature=cfg['temperature'],
        top_p=cfg['top_p'], beta=cfg['beta'], loss_type=cfg['loss_type'],
        scale_rewards=cfg['scale_rewards'], mask_truncated_completions=cfg['mask_truncated_completions'],
        reward_weights=cfg['arms'][a.arm], num_iterations=1, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        logging_steps=1, logging_first_step=True, save_steps=50, save_total_limit=3,
        report_to='none', optim='adamw_torch', use_vllm=False,
        run_name=f'{a.arm}-seed{a.seed}-{output.name}', save_safetensors=True)

    class AuditTrainer(GRPOTrainer):
        # Version-pinned hook; super gathers across ALL ranks before grouping.
        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            correct = rewards[:, 0].reshape(-1, self.num_generations)
            fmt = rewards[:, 1]
            mode = 'train' if self.model.training else 'eval'
            metrics = self._metrics[mode]
            metrics['audit/correctness'].append(correct.mean().item())
            metrics['audit/format_fail'].append((1 - fmt).mean().item())
            metrics['audit/all_wrong_groups'].append((correct.sum(1) == 0).float().mean().item())
            metrics['audit/all_correct_groups'].append((correct.sum(1) == self.num_generations).float().mean().item())
            return rewards

    class ResourceLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                # Explicitly rank-0 only; no claim that this is max across GPUs.
                logs['resources/rank0_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30

    callbacks = [ResourceLog()]
    if not a.inspect_only and a.swanlab_mode != 'disabled':
        os.environ['SWANLAB_MODE'] = a.swanlab_mode
        from swanlab.integration.transformers import SwanLabCallback
        callbacks.append(SwanLabCallback(project='qwen3-reward-restart',
            experiment_name=args.run_name, description=f'Reward ablation: {a.arm}; seed={a.seed}'))
    trainer = AuditTrainer(model=model, processing_class=tokenizer,
        peft_config=lora, reward_funcs=[strict_correctness, strict_format],
        args=args, train_dataset=Dataset.from_list(rows), callbacks=callbacks)
    if not any(p.requires_grad for p in trainer.model.parameters()):
        raise RuntimeError('No trainable parameters')
    if any('lora_' not in n for n, p in trainer.model.named_parameters() if p.requires_grad):
        raise RuntimeError('Unexpected trainable base weights')
    provenance = dict(arm=a.arm, seed=a.seed, world_size=world, layout=layout,
        experiment=cfg, max_steps=args.max_steps, assets=asset_manifest,
        model_dir=str(Path(a.model_dir).resolve()), data=data_manifest,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={n: importlib.metadata.version(n) for n in
                  ['torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'swanlab']},
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        prompt_sha256=sha256('configs/sft_prompt.txt'),
        parser_sha256=sha256('scripts/evaluator.py'),
        train_data_sha256=sha256(data_path), resume=a.resume)
    if a.resume:
        previous = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        for key in ('arm', 'seed', 'world_size', 'layout', 'experiment', 'assets', 'train_data_sha256', 'git_commit'):
            if previous[key] != provenance[key]:
                raise ValueError(f'Resume incompatible: {key}')
        checkpoint = Path(a.resume).resolve()
        if checkpoint.parent != output.resolve() or not (checkpoint / 'trainer_state.json').is_file():
            raise ValueError('Resume must be a full checkpoint-N in this output directory')
    if a.inspect_only:
        if rank == 0:
            print(json.dumps(provenance, ensure_ascii=False, indent=2))
            print('INSPECT PASS: no rollout / train / optimizer step executed')
        return
    if rank == 0:
        write_json(output / ('resume.json' if a.resume else 'run.json'), provenance)
    trainer.accelerator.wait_for_everyone()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(output / 'final_adapter'))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / 'final_adapter')
        write_json(output / 'metrics.json', dict(result.metrics, elapsed_seconds=time.monotonic()-start,
                   evaluation_status='NOT_EVALUATED'))
        write_json(output / 'log_history.json', trainer.state.log_history)
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
