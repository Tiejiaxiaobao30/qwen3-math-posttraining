"""新增的GRPO定向对照入口；原train.py/common.py/sample.py/evaluate.py保持不变。"""
import argparse
from collections import Counter
from decimal import InvalidOperation
import importlib.metadata
import json
import os
from pathlib import Path
import random

from common import ROOT, assets, build_result, correctness_reward, digest, extract_answer, generate, load_model, provenance, read, record, write
from sample import select
from train import layout

TRAIN_STEPS = 200
CHECKPOINT_STEPS = 25
SAMPLING = dict(num_generations=4, max_completion_length=256, temperature=0.8, top_p=0.95, top_k=0)

SOURCES = {
    'dr_grpo': 'https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/train_zero_math.py',
    'dapo': 'https://github.com/BytedTsinghua-SIA/DAPO/tree/33fe3176f0bb212588e84fc8ccf50dd554975144',
    'trl': 'https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py',
    'open_r1': 'https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py',
}
# Each arm changes only the named factors. No full DAPO/Oat reproduction is claimed.
ARMS = {
    'baseline': {},
    'no_std': {'scale_rewards': 'none'},
    'token_norm': {'loss_type': 'dapo'},
    'token_no_std': {'loss_type': 'dapo', 'scale_rewards': 'none'},
    'dr_grpo': {'loss_type': 'dr_grpo', 'scale_rewards': 'none'},
    'low_lr': {'learning_rate': 3e-7},
}


def arm_config(arm):
    """先读baseline，再看单因素差异；不改G、prompt、reward、KL或长度。"""
    cfg = dict(learning_rate=1e-6, max_prompt_length=512, beta=0.04, **SAMPLING,
               loss_type='bnpo', scale_rewards='group', num_iterations=1,
               mask_truncated_completions=True, remove_unused_columns=False, use_vllm=False)
    cfg.update(ARMS[arm])
    return cfg


def batch_config(world, arm):
    if world == 1:
        # One G4 group per microbatch. Three groups make one optimizer update.
        # DAPO uses the generation batch for its denominator, so generate all three groups together there.
        generation_steps = 3 if arm_config(arm)['loss_type'] == 'dapo' else 1
        return dict(per_device_train_batch_size=4, gradient_accumulation_steps=3,
                    steps_per_generation=generation_steps)
    return dict(**layout(world), steps_per_generation=12//world)


def load_training_rows(path):
    """只接受原RL1000或sample.py生成的完整、未改写子集。"""
    source_path = ROOT/'data/rl.jsonl'
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    source_hash = digest(source_path)
    if source_hash != manifest['files']['rl']['sha256']:
        raise ValueError('Original RL pool changed')
    source = read(source_path)
    split = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    if [r['id'] for r in source] != split['splits']['rl']['ids']:
        raise ValueError('Original RL IDs/order differ from the frozen split')
    path = Path(path).resolve()
    rows = read(path)
    by_id = {r['id']:r for r in source}
    if not rows or len({r['id'] for r in rows}) != len(rows) or any(by_id.get(r['id']) != r for r in rows):
        raise ValueError('Empty, duplicate, foreign or rewritten training rows')
    if digest(path) != source_hash:
        mp = path.parent/'manifest.json'
        selected = json.loads(mp.read_text(encoding='utf-8'))
        if (selected.get('complete') is not True or selected['source_sha256'] != source_hash
                or selected['files'].get(path.name) != digest(path) or selected['count'] != len(rows)):
            raise ValueError('Selection is incomplete or its source/content changed')
        for filename, key in [('run.json', 'run_sha256'), ('rollouts.jsonl', 'rollouts_sha256')]:
            if digest(path.parent/filename) != selected[key]:
                raise ValueError('Selection evidence changed: ' + filename)
    return rows


def strict_reward(completions, gold, **kwargs):
    values = []
    for text, reference in zip(completions, gold):
        try:
            values.append(correctness_reward(text, reference))
        except InvalidOperation:
            # The frozen regex can match a comma-only numeral. Reject it as malformed.
            values.append(0.0)
    return values


def format_ok(text):
    try:
        return float(extract_answer(text) is not None)
    except InvalidOperation:
        return 0.0


def screening_groups(model, tokenizer, rows, seed, stream):
    from transformers import GenerationConfig, set_seed
    cfg = arm_config('baseline')
    # Supply an explicit config so different checkpoints cannot set sampling defaults.
    model.generation_config = GenerationConfig(do_sample=True, top_k=cfg['top_k'], eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id)
    for row in rows:
        set_seed((seed + row['id']) % 2**32)
        outputs = generate(model, tokenizer, row['prompt'], count=cfg['num_generations'],
                           sample=True, max_tokens=cfg['max_completion_length'])
        texts = [output[0] for output in outputs]
        group = dict(id=row['id'], outputs=outputs, rewards=strict_reward(texts, [row['gold']]*len(texts)),
                     format_fail=sum(1-format_ok(text) for text in texts), truncated=sum(o[2] for o in outputs))
        stream.write(json.dumps(group, ensure_ascii=False)+'\n')
        stream.flush()
        yield group


def paired_groups(instruct, sft):
    if [r['id'] for r in instruct] != [r['id'] for r in sft]:
        raise ValueError('Paired screening IDs/order differ')
    def label(row):
        correct = sum(row['rewards'])
        return 'all_wrong' if correct == 0 else 'all_correct' if correct == 4 else 'mixed'
    transitions = Counter(f'{label(left)}->{label(right)}' for left,right in zip(instruct,sft))
    return dict(n=len(sft), instruct=dict(Counter(map(label,instruct))), sft=dict(Counter(map(label,sft))),
                transitions=dict(transitions), instruct_format_fail=sum(r['format_fail'] for r in instruct),
                sft_format_fail=sum(r['format_fail'] for r in sft),
                instruct_truncated=sum(r['truncated'] for r in instruct), sft_truncated=sum(r['truncated'] for r in sft))


def screen(a, rows, output):
    output.mkdir(parents=True, exist_ok=False)
    identity = assets(a.model,a.sft)
    probe_ids = set(random.Random(20261118).sample([r['id'] for r in rows], min(100,len(rows))))
    probe = [r for r in rows if r['id'] in probe_ids]
    record(output/'run.json', assets=identity, instruct_assets=assets(a.instruct) if a.instruct else None,
           source_sha256=digest(a.data), code=provenance(), probe_ids=[r['id'] for r in probe],
           sampling=dict(seed=a.seed, per_question_seed='(seed + id) % 2**32',
                         **SAMPLING, prompt='raw project prompt'),
           control_seed=20261119)
    model, tokenizer = load_model(a.model,a.sft)
    model.cuda().eval()
    with (output/'rollouts.jsonl').open('x',encoding='utf-8') as stream:
        groups = list(screening_groups(model,tokenizer,rows,a.seed,stream))
    del model, tokenizer
    informative, control = select(rows,groups,20261119)
    selected_ids = {r['id'] for r in informative}
    control_ids = {r['id'] for r in control}
    record(output/'screen_summary.json', n=len(rows), histogram=dict(Counter(int(sum(r['rewards'])) for r in groups)),
           selected_n=len(informative), overlap_n=len(selected_ids & control_ids),
           format_fail=sum(r['format_fail'] for r in groups), truncated=sum(r['truncated'] for r in groups))
    if a.instruct:
        import torch
        torch.cuda.empty_cache()
        model, tokenizer = load_model(a.instruct)
        model.cuda().eval()
        with (output/'instruct_probe.jsonl').open('x',encoding='utf-8') as stream:
            instruct = list(screening_groups(model,tokenizer,probe,a.seed,stream))
        record(output/'paired_probe.json', **paired_groups(instruct,[r for r in groups if r['id'] in probe_ids]))
    if len(informative) < 3:
        raise ValueError('Fewer than three mixed questions; no training subsets produced')
    write(output/'informative.jsonl',informative)
    write(output/'random.jsonl',[r for r in rows if r['id'] in control_ids])
    record(output/'manifest.json', complete=True, source_sha256=digest(a.data), count=len(informative),
           run_sha256=digest(output/'run.json'), rollouts_sha256=digest(output/'rollouts.jsonl'),
           files={name:digest(output/name) for name in ('informative.jsonl','random.jsonl')},
           screen_assets=identity)
    print(f'Screened {len(rows)} questions; {len(informative)} per training arm; overlap {len(selected_ids & control_ids)}')


def evaluation_result(row, text, length, truncated):
    try:
        return build_result(row['id'],text,row['gold'],length,truncated)
    except InvalidOperation:
        return dict(id=row['id'],reference=row['gold'],prediction=text,parsed_answer=None,
                    status='format_fail',generated_tokens=length,truncated=truncated)


def evaluate_dev(model, tokenizer, output, run):
    """独立实验的安全评测；仍用原dev/greedy512/原严格评分，不改旧evaluate.py。"""
    from collections import Counter
    output = Path(output)
    for path in [output,output.with_suffix('.run.json'),output.with_suffix('.summary.json')]:
        if path.exists():
            raise FileExistsError('Use new evaluation files')
    path = ROOT/'data/dev.jsonl'
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    split = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    rows = read(path)
    if digest(path) != manifest['files']['dev']['sha256'] or [r['id'] for r in rows] != split['splits']['dev']['ids']:
        raise ValueError('Original dev changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    record(output.with_suffix('.run.json'),**run,data_sha256=digest(path),code=provenance(),
           do_sample=False,max_new_tokens=512,scope='Exploratory historical dev only')
    counts, truncated, tokens = Counter(), 0, 0
    with output.open('x',encoding='utf-8') as f:
        for row in rows:
            text,length,cut = generate(model,tokenizer,row['prompt'],max_tokens=512)[0]
            result = evaluation_result(row,text,length,cut)
            f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
            counts[result['status']] += 1; truncated += cut; tokens += length
    record(output.with_suffix('.summary.json'),n=len(rows),counts=dict(counts),truncated=truncated,
           generated_tokens=tokens,prediction_sha256=digest(output))
    print(dict(counts), 'n=',len(rows))


def group_summary(rewards, formats, lengths, truncated, ids, generations=4):
    """输入是跨rank gather之后的完整组；仅诊断，绝不改reward/advantage。"""
    import torch
    groups = rewards.reshape(-1, generations)
    grouped_ids = ids.reshape(-1, generations)
    if not torch.all(grouped_ids == grouped_ids[:, :1]):
        raise ValueError('A generation group contains different question IDs')
    counts = groups.sum(1)
    correct = rewards > 0
    def mean_or_zero(values):
        return values.float().mean().item() if values.numel() else 0.0
    return dict(correctness=mean_or_zero(rewards), format_fail=1-mean_or_zero(formats),
        all_wrong_groups=mean_or_zero((counts == 0).float()),
        all_correct_groups=mean_or_zero((counts == generations).float()),
        mixed_groups=mean_or_zero(((counts > 0) & (counts < generations)).float()),
        truncated=mean_or_zero(truncated), mean_tokens=mean_or_zero(lengths),
        correct_tokens=mean_or_zero(lengths[correct]), wrong_tokens=mean_or_zero(lengths[~correct]))


def require_active_completions(truncated):
    if bool((truncated > 0).all()):
        raise RuntimeError('All completions were truncated: no active loss tokens; preserve this run and inspect it')


def make_trainer(model, tokenizer, rows, arm, output, steps=TRAIN_STEPS, seed=42, world=1,
                 save_steps=CHECKPOINT_STEPS, runtime_options=None, batch_options=None):
    """真实训练与CPU微型测试共用此构建函数；优化器与loss仍由TRL实现。"""
    if importlib.metadata.version('trl') != '0.24.0':
        raise RuntimeError('Use the existing requirements.txt: TRL must be 0.24.0')
    import torch
    from accelerate.utils import gather
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    cfg = dict(output_dir=str(output), max_steps=steps, seed=seed, data_seed=seed,
        **(batch_options or batch_config(world,arm)), **arm_config(arm), bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False, logging_steps=1, save_steps=save_steps,
        save_total_limit=None, report_to='none', optim='adamw_torch')
    if runtime_options:
        cfg.update(runtime_options)  # Micro-model tests override CPU/dtype/length, never a cloud CLI flag.
    args = GRPOConfig(**cfg)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model.config.use_cache = False

    class DiagnosticTrainer(GRPOTrainer):
        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            eos = self.processing_class.eos_token_id
            pad = self.processing_class.pad_token_id
            local = torch.tensor([[format_ok(text), len(tokens), float(tokens[-1] not in (eos,pad)), row['id']]
                for text,tokens,row in zip(completions,completion_ids_list,inputs)],
                device=self.accelerator.device, dtype=torch.float32)
            stats = gather(local)
            report = group_summary(rewards[:,0], stats[:,0], stats[:,1], stats[:,2], stats[:,3], self.num_generations)
            mode = 'train' if self.model.training else 'eval'
            for key,value in report.items():
                self._metrics[mode]['audit/'+key].append(value)
            if self.is_world_process_zero():
                with (Path(self.args.output_dir)/'online_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(before_update=self.state.global_step+1, **report))+'\n')
            if self.args.mask_truncated_completions:
                require_active_completions(stats[:,2])
            return rewards

    trainer = DiagnosticTrainer(model=model, processing_class=tokenizer, args=args,
        train_dataset=Dataset.from_list(rows), peft_config=lora, reward_funcs=strict_reward)
    names = [n for n,p in trainer.model.named_parameters() if p.requires_grad]
    if not names or any('lora_' not in n for n in names):
        raise ValueError('Only the fresh GRPO LoRA may be trainable')
    return trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, default='baseline')
    p.add_argument('--data', default=str(ROOT/'data/rl.jsonl'))
    p.add_argument('--model')
    p.add_argument('--sft')
    p.add_argument('--screen', action='store_true', help='Screen all RL questions and write matched training subsets')
    p.add_argument('--instruct', help='Optional local post-trained Qwen checkpoint for 100-question paired screening')
    p.add_argument('--eval-dev', action='store_true', help='Greedy512 on original dev; --output is a JSONL file')
    p.add_argument('--grpo', help='Optional GRPO adapter for --eval-dev only; always merge the same SFT first')
    p.add_argument('--output')
    p.add_argument('--steps', type=int, default=TRAIN_STEPS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--world-size', type=int, choices=[1,2,3])
    p.add_argument('--save-steps', type=int, default=CHECKPOINT_STEPS)
    p.add_argument('--plan', action='store_true', help='Validate data and print configuration; no model imports')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--resume', help='Full checkpoint-N in output; same arm/configuration required')
    a = p.parse_args()
    if a.steps <= 0 or a.save_steps <= 0:
        p.error('steps and save-steps must be positive')
    if a.grpo and not a.eval_dev:
        p.error('--grpo is for evaluation; training always starts from SFT')
    if a.instruct and not a.screen:
        p.error('--instruct is only used with --screen')
    if a.screen and (a.eval_dev or a.plan or a.inspect_only or a.resume):
        p.error('--screen is separate from plan/inspect/eval/resume')
    if a.eval_dev and (a.plan or a.inspect_only or a.resume):
        p.error('--eval-dev is separate from plan/inspect/resume')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    planned_world = a.world_size or world
    rows = load_training_rows(a.data)
    source_hash = digest(ROOT/'data/rl.jsonl')
    if a.screen and (world != 1 or digest(a.data) != source_hash):
        p.error('--screen requires a single process and the complete original RL pool')
    if not a.eval_dev and len(rows) < 3:
        p.error('The generation batch needs at least three unique prompts')
    batch = batch_config(planned_world,a.arm)
    completion_microbatch = batch['per_device_train_batch_size'] * planned_world
    plan = dict(arm=a.arm, config=arm_config(a.arm), layout=batch,
        world_size=planned_world, generation_batch=completion_microbatch * batch['steps_per_generation'],
        completions_per_update=completion_microbatch * batch['gradient_accumulation_steps'], count=len(rows), steps=a.steps,
        data_sha256=digest(a.data), seed=a.seed, save_steps=a.save_steps, sources=SOURCES)
    if a.plan:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return
    if not a.model or not a.sft or not a.output:
        p.error('--model, --sft and --output required for inspect/train')
    if planned_world != world:
        p.error('--world-size differs from the actual torchrun WORLD_SIZE')
    output = Path(a.output).resolve()
    if not a.eval_dev and not a.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    if a.resume and (Path(a.resume).resolve().parent != output or not (Path(a.resume)/'trainer_state.json').is_file()):
        p.error('--resume must be a complete checkpoint-N in output')
    if a.resume and (output/f'metrics-step{a.steps}.json').exists():
        raise FileExistsError('This run already completed; preserve its final adapter and metrics')
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Real Qwen training requires a BF16 CUDA GPU; CPU tests are separate')
    # Same exact SFT snapshot as the original seed42 experiment; no accidental GRPO starting point.
    frozen = json.loads((ROOT/'历史结果/results/frozen_eval_manifest.json').read_text(encoding='utf-8'))
    if digest(Path(a.sft)/'adapter_model.safetensors') != frozen['artifacts']['sft_seed42']['sha256']:
        raise ValueError('This protocol requires the frozen SFT seed42 adapter')
    model_config = json.loads((Path(a.model)/'config.json').read_text(encoding='utf-8'))
    if model_config.get('model_type') != 'qwen3' or model_config.get('hidden_size') != 2048:
        raise ValueError('Expected full Qwen3-1.7B-Base, not another size or an adapter')
    if a.screen:
        screen(a,rows,output)
        return
    identity = assets(a.model,a.sft,a.grpo)
    if not a.eval_dev and digest(a.data) != source_hash:
        selected = json.loads(Path(a.data).with_name('manifest.json').read_text(encoding='utf-8'))
        if selected.get('screen_assets', identity) != identity:
            raise ValueError('Training must start from the same Base and SFT used to screen the questions')
    set_seed(a.seed)
    model, tokenizer = load_model(a.model, a.sft, a.grpo)
    if a.eval_dev:
        model.cuda().eval()
        evaluate_dev(model,tokenizer,output,dict(model=a.model,sft=a.sft,grpo=a.grpo,
            assets=identity,sources=SOURCES))
        return
    if max(len(tokenizer(row['prompt'])['input_ids']) for row in rows) > 512:
        raise ValueError('Prompt exceeds max_prompt_length')
    trainer = make_trainer(model,tokenizer,rows,a.arm,output,a.steps,a.seed,world,a.save_steps)
    run = dict(**plan, assets=identity, code=provenance(),
        model_dir=str(Path(a.model).resolve()), sft_dir=str(Path(a.sft).resolve()),
        packages={k:importlib.metadata.version(k) for k in ['torch','trl','transformers','peft','accelerate','datasets']})
    if a.resume:
        if json.loads((output/'run.json').read_text(encoding='utf-8')) != run:
            raise ValueError('Resume configuration, data, code or assets differ')
    if a.inspect_only:
        if trainer.is_world_process_zero():
            print(json.dumps(run,ensure_ascii=False,indent=2))
        return
    if trainer.is_world_process_zero() and not a.resume:
        record(output/'run.json',**run)
    trainer.accelerator.wait_for_everyone()
    result = trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(output/'final_adapter'))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output/'final_adapter')
        metrics = dict(result.metrics, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        record(output/f'metrics-step{trainer.state.global_step}.json',**metrics)
        record(output/f'logs-step{trainer.state.global_step}.json',history=trainer.state.log_history)


if __name__ == '__main__':
    main()
