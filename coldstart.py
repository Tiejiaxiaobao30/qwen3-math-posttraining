"""Early SFT → RL1000 screening → matched GRPO → official test1319."""

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random

from common import ROOT, PROMPT, assets, digest, load_model, provenance, read, record, write
from eval_test_b8 import EXPECTED, MAX_TOKENS, generate_batch
from train_grpo_experiment import (
    SAMPLING, arm_config, evaluation_result, load_training_rows, make_trainer,
)

SEED = 42
ORIGINAL_SFT_STEP = 250
SFT_BATCH_SIZE = 16
DEFAULT_SFT_STEPS = 32
SFT_HORIZON = 125
EVAL_BATCH_SIZE = 20
HISTORY = ROOT / '历史结果' / 'grpo_screening_4090'
HISTORICAL_SFT = ROOT / '历史结果' / 'results' / 'official_test' / 'sft_seed42_test1319_summary.json'
TEST = ROOT / '历史结果' / 'data' / 'official' / 'gsm8k_test_1319.jsonl'
HOLDOUT = ROOT / 'data' / 'holdout500.jsonl'
HOLDOUT_SHA256 = '1eceb3aea80544c6b270abad209e51632f1108c29288284c9722453ad9e378bb'
HISTORICAL_COUNT = 476
SELECTION_SEED = 20261119
GROUPS = ('random', 'informative')
EVAL_NAMES = ('weak_sft', *GROUPS, 'strong_sft', 'strong_random', 'strong_informative')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate')}


def start_gpu(model):
    import torch
    from transformers import set_seed

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one process per terminal on one GPU')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This workflow requires a BF16 CUDA GPU')
    if importlib.metadata.version('trl') != '0.24.0':
        raise RuntimeError('Install the versions in requirements.txt')
    if digest(model / 'model.safetensors') != EXPECTED['base']:
        raise ValueError('Use the same Base weights as the previous 4090 experiment')
    set_seed(SEED)


def early_sft(a):
    from transformers import TrainerCallback
    from train import build_trainer

    output = a.output / 'sft_rebuild'
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / 'data' / 'sft.jsonl'
    rows = read(source)
    expected_ids = read_json(ROOT / 'split_manifest.json')['splits']['sft']['ids']
    manifest = read_json(source.parent / 'manifest.json')
    if [row['id'] for row in rows] != expected_ids or digest(source) != manifest['files']['sft']['sha256']:
        raise ValueError('Expected the original SFT2000 split')

    class SaveEarlyCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == a.sft_steps:
                control.should_save = True
                control.should_training_stop = True
            return control

    start_gpu(a.model)
    model, tokenizer = load_model(a.model)
    tokenizer.padding_side = 'right'
    trainer = build_trainer('sft', model, tokenizer, rows, dict(
        output_dir=str(output), max_steps=SFT_HORIZON, lr_scheduler_type='linear',
        per_device_train_batch_size=SFT_BATCH_SIZE, gradient_accumulation_steps=1,
        seed=SEED, data_seed=SEED, bf16=True, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=5, save_steps=a.sft_steps,
        save_only_model=True, report_to='none',
    ), world=1)
    trainer.add_callback(SaveEarlyCheckpoint())
    result = trainer.train()
    checkpoint = output / f'checkpoint-{a.sft_steps}'
    if trainer.state.global_step != a.sft_steps or not (checkpoint / 'adapter_model.safetensors').is_file():
        raise RuntimeError(f'Expected an adapter at {checkpoint}')
    record(output / 'run.json', origin='rebuilt', stop_step=a.sft_steps, scheduler_steps=SFT_HORIZON,
           batch_size=SFT_BATCH_SIZE, samples_processed=a.sft_steps * SFT_BATCH_SIZE,
           data_sha256=digest(source), assets=assets(a.model, checkpoint), packages=package_versions(),
           seed=SEED, code=provenance(), metrics=result.metrics)
    print(f'Use --sft {checkpoint}')


def matched_pools(rows, groups, historical_random):
    ids = [row['id'] for row in rows]
    if ids != [group['id'] for group in groups]:
        raise ValueError('Screening must cover RL1000 in its original order')
    if any(len(g['rewards']) != 4 or any(r not in (0, 1) for r in g['rewards']) for g in groups):
        raise ValueError('Expected four binary rewards per question')
    if len(historical_random) != HISTORICAL_COUNT or len(set(historical_random)) != HISTORICAL_COUNT:
        raise ValueError('Expected the 476 historical random IDs')
    by_id = {row['id']: row for row in rows}
    histogram = dict(sorted(Counter(int(sum(g['rewards'])) for g in groups).items()))
    mixed = [row['id'] for row, group in zip(rows, groups, strict=True) if 0 < sum(group['rewards']) < 4]
    # B8 uses two questions per optimizer update. An even K gives two complete passes.
    count = min(len(mixed), HISTORICAL_COUNT)
    count -= count % 2
    if count < 2:
        raise ValueError('Fewer than two mixed questions. Inspect the saved rollouts before training.')
    picked = set(random.Random(SELECTION_SEED).sample(mixed, count))
    return [by_id[i] for i in historical_random[:count]], [by_id[i] for i in mixed if i in picked], histogram


def weak_sft_dir(a):
    return a.sft or rebuilt_sft_dir(a)


def rebuilt_sft_dir(a):
    return a.output / 'sft_rebuild' / f'checkpoint-{a.sft_steps}'


def screen(a):
    from screen12 import screening_groups

    output = a.output / 'screen'
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / 'data' / 'rl.jsonl'
    rows = load_training_rows(source)
    sft = weak_sft_dir(a)
    rebuilt = sft.resolve() == rebuilt_sft_dir(a).resolve()
    state = sft / 'trainer_state.json'
    expected_step = a.sft_steps if rebuilt else ORIGINAL_SFT_STEP
    if state.is_file() and read_json(state)['global_step'] != expected_step:
        raise ValueError(f'Expected SFT optimizer step {expected_step} for this checkpoint')
    start_gpu(a.model)
    identity = assets(a.model, sft)
    if identity['sft']['adapter_model.safetensors'] == EXPECTED['sft']:
        raise ValueError('Screening needs the early checkpoint, not the historical SFT42 final adapter')
    model, tokenizer = load_model(a.model, sft)
    model.cuda().eval()
    output.mkdir(parents=True, exist_ok=False)
    record(output / 'run.json', seed=SEED, assets=identity, data_sha256=digest(source),
           sft_path=str(sft.resolve()),
           sft_origin='rebuilt_b16' if rebuilt else 'supplied_original',
           sft_step=expected_step,
           sft_samples_processed=expected_step * (SFT_BATCH_SIZE if rebuilt else 2),
           packages=package_versions(), code=provenance(),
           sampling=dict(**SAMPLING, prompt_batch_size=a.screen_batch_size,
                         seed_rule='(seed + first question ID in each batch) % 2**32'))
    with (output / 'rollouts.jsonl').open('x', encoding='utf-8') as stream:
        groups = list(screening_groups(model, tokenizer, rows, SEED, stream, a.screen_batch_size))
    historical_random = read_json(HISTORY / 'training_ids.json')['random']
    control, informative, histogram = matched_pools(rows, groups, historical_random)
    for name, subset in zip(GROUPS, (control, informative), strict=True):
        write(output / f'{name}.jsonl', subset)
    overlap = len({r['id'] for r in control} & {r['id'] for r in informative})
    record(output / 'manifest.json', complete=True, source_sha256=digest(source), count=len(control),
           histogram=histogram, format_fail=sum(g['format_fail'] for g in groups),
           truncated=sum(g['truncated'] for g in groups), overlap=overlap,
           historical_count_matched=len(control) == HISTORICAL_COUNT,
           informative_selection_seed=SELECTION_SEED, screen_assets=identity,
           run_sha256=digest(output / 'run.json'), rollouts_sha256=digest(output / 'rollouts.jsonl'),
           files={f'{name}.jsonl': digest(output / f'{name}.jsonl') for name in GROUPS})
    print(f'Correct out of four: {histogram}; {len(control)} questions per arm; overlap={overlap}')


def grpo(a):
    from train_b8 import batch_config

    selection = a.output / 'screen'
    source = selection / f'{a.group}.jsonl'
    rows = load_training_rows(source)
    manifest = read_json(selection / 'manifest.json')
    sft = weak_sft_dir(a)
    identity = assets(a.model, sft)
    if manifest['screen_assets'] != identity:
        raise ValueError('GRPO must start from the same Base and SFT used for screening')
    output = a.output / a.group
    steps = len(rows)
    layout = batch_config(1, 'baseline')
    run = dict(group=a.group, assets=identity, data_sha256=digest(source),
               selection_sha256=digest(selection / 'manifest.json'), steps=steps,
               seed=SEED, config=arm_config('baseline'), layout=layout,
               packages=package_versions(), code=provenance())
    if a.resume:
        if (a.resume.resolve().parent != output.resolve()
                or not all((a.resume / name).is_file() for name in
                           ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                            'adapter_model.safetensors'))):
            raise ValueError('--resume requires a full checkpoint in this training run')
        if read_json(output / 'run.json') != run or (output / 'metrics.json').exists():
            raise ValueError('Resume requires the same configuration and an unfinished run')
    elif output.exists():
        raise FileExistsError(output)
    start_gpu(a.model)
    model, tokenizer = load_model(a.model, sft)
    if max(len(tokenizer(row['prompt'])['input_ids']) for row in rows) > 512:
        raise ValueError('Prompt exceeds the previous GRPO limit of 512 tokens')
    trainer = make_trainer(model, tokenizer, rows, 'baseline', output, steps=steps, seed=SEED,
                           save_steps=25, batch_options=layout)
    if not a.resume:
        output.mkdir(parents=True, exist_ok=True)
        record(output / 'run.json', **run)
    result = trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    if trainer.state.global_step != steps:
        raise RuntimeError(f'Expected {steps} GRPO updates')
    trainer.save_model(str(output / 'final_adapter'))
    trainer.save_state()
    tokenizer.save_pretrained(output / 'final_adapter')
    record(output / 'metrics.json', steps=steps, adapter_sha256=digest(
        output / 'final_adapter' / 'adapter_model.safetensors'), **result.metrics)


def official_test():
    if digest(TEST) != EXPECTED['test'] or digest(ROOT / 'prompt.txt') != EXPECTED['prompt']:
        raise ValueError('Official test or prompt hash differs from the previous 4090 evaluation')
    rows = read(TEST)
    if len(rows) != 1319 or len({row['id'] for row in rows}) != 1319:
        raise ValueError('Expected the 1,319 distinct official questions')
    return rows


def evaluation_rows(dataset):
    if dataset == 'test':
        return [dict(id=row['id'], prompt=PROMPT.format(question=row['question']),
                     gold=row['final_answer']) for row in official_test()]
    if digest(HOLDOUT) != HOLDOUT_SHA256 or digest(ROOT / 'prompt.txt') != EXPECTED['prompt']:
        raise ValueError('The reserved holdout500 or project prompt changed')
    rows = read(HOLDOUT)
    if len(rows) != 500 or len({row['id'] for row in rows}) != 500:
        raise ValueError('Expected 500 unique reserved questions')
    return rows


def dataset_hash(dataset):
    return EXPECTED['test'] if dataset == 'test' else HOLDOUT_SHA256


def evaluate(a):
    rows = evaluation_rows(a.dataset)
    start_gpu(a.model)
    sft = weak_sft_dir(a)
    adapter = a.grpo
    if a.name in GROUPS:
        run_dir = a.output / a.name
        adapter = run_dir / 'final_adapter'
        run = read_json(run_dir / 'run.json')
        metrics = read_json(run_dir / 'metrics.json')
        if (assets(a.model, sft) != run['assets'] or metrics['steps'] != run['steps']
                or digest(adapter / 'adapter_model.safetensors') != metrics['adapter_sha256']):
            raise ValueError('Evaluation must use the exact training origin and its completed final adapter')
    elif a.name == 'weak_sft':
        if assets(a.model, sft) != read_json(a.output / 'screen' / 'manifest.json')['screen_assets']:
            raise ValueError('Use the same weak SFT checkpoint as in screening')
    elif a.name.startswith('strong_'):
        if digest(sft / 'adapter_model.safetensors') != EXPECTED['sft']:
            raise ValueError('Use the original SFT42 final adapter for the strong arms')
        if a.name in ('strong_random', 'strong_informative') and not adapter:
            raise ValueError('Pass the corresponding previous GRPO final adapter with --grpo')
        if adapter:
            group = a.name.removeprefix('strong_')
            expected = read_json(HISTORY / 'summary.json')['runs'][group]['test']['grpo_sha256']
            if digest(adapter / 'adapter_model.safetensors') != expected:
                raise ValueError(f'Expected the original {group} GRPO adapter')
    if a.grpo and a.name not in ('strong_random', 'strong_informative'):
        raise ValueError('--grpo is only used for the two historical GRPO arms')
    model, tokenizer = load_model(a.model, sft, adapter)
    model.cuda().eval()
    directory = a.output / a.dataset
    directory.mkdir(parents=True, exist_ok=True)
    prediction = directory / f'{a.name}.jsonl'
    counts = Counter(correct=0, wrong=0, format_fail=0, truncated=0)
    with prediction.open('x', encoding='utf-8') as stream:
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start:start + a.batch_size]
            prompts = [row['prompt'] for row in batch]
            for row, output in zip(batch, generate_batch(model, tokenizer, prompts), strict=True):
                result = evaluation_result(row, *output)
                counts[result['status']] += 1
                counts['truncated'] += result['truncated']
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{a.name}: {start + len(batch)}/{len(rows)}; correct={counts["correct"]}', flush=True)
    record(directory / f'{a.name}.summary.json', total=len(rows), **counts,
           strict_accuracy=counts['correct'] / len(rows), batch_size=a.batch_size, do_sample=False,
           max_new_tokens=MAX_TOKENS, assets=assets(a.model, sft, adapter),
           test_sha256=dataset_hash(a.dataset), prompt_sha256=EXPECTED['prompt'],
           predictions_sha256=digest(prediction), packages=package_versions())


def paired(before, after):
    ids = list(before)
    gained = [i for i in ids if before[i] != 'correct' and after[i] == 'correct']
    lost = [i for i in ids if before[i] == 'correct' and after[i] != 'correct']
    return dict(gained=gained, lost=lost, net=len(gained) - len(lost),
                delta_pp=100 * (len(gained) - len(lost)) / len(ids))


def compare(a):
    expected_rows = [(row['id'], row['gold']) for row in evaluation_rows(a.dataset)]
    directory = a.output / a.dataset
    verdicts = {}
    comparison_keys = ('batch_size', 'do_sample', 'max_new_tokens', 'test_sha256', 'prompt_sha256')
    eval_keys = None
    selection = read_json(a.output / 'screen' / 'manifest.json')
    for name in EVAL_NAMES:
        path = directory / f'{name}.jsonl'
        if name.startswith('strong_') and not path.is_file():
            continue
        rows = read(path)
        summary = read_json(directory / f'{name}.summary.json')
        keys = [summary[key] for key in comparison_keys] + [summary['assets']['base'], summary['packages']]
        identity = {part: summary['assets'][part] for part in ('base', 'sft')}
        if ([(row['id'], row['reference']) for row in rows] != expected_rows
                or summary['test_sha256'] != dataset_hash(a.dataset)
                or digest(path) != summary['predictions_sha256']
                or (eval_keys is not None and keys != eval_keys)
                or (not name.startswith('strong_') and identity != selection['screen_assets'])):
            raise ValueError(f'Evaluation identity differs for {name}')
        eval_keys = keys
        verdicts[name] = {row['id']: row['status'] for row in rows}
    comparisons = dict(
        weak_screen_vs_random=paired(verdicts['random'], verdicts['informative']),
        weak_screen_vs_sft=paired(verdicts['weak_sft'], verdicts['informative']),
        weak_random_vs_sft=paired(verdicts['weak_sft'], verdicts['random']),
    )
    if 'strong_sft' in verdicts:
        comparisons['weak_screen_vs_strong_sft'] = paired(verdicts['strong_sft'], verdicts['informative'])
        comparisons['weak_sft_vs_strong_sft'] = paired(verdicts['strong_sft'], verdicts['weak_sft'])
    if 'strong_random' in verdicts and 'strong_informative' in verdicts:
        comparisons['strong_screen_vs_random'] = paired(verdicts['strong_random'], verdicts['strong_informative'])
        if 'strong_sft' in verdicts:
            comparisons['strong_screen_vs_sft'] = paired(verdicts['strong_sft'], verdicts['strong_informative'])
    historical = a.dataset == 'test' and 'strong_screen_vs_random' not in comparisons
    strong = comparisons.get('strong_screen_vs_random', {}).get('net')
    if historical:
        previous = read_json(HISTORY / 'summary.json')
        strong = previous['paired']['net']
    interaction = None if strong is None else comparisons['weak_screen_vs_random']['net'] - strong
    report = dict(dataset=a.dataset, total=len(expected_rows), correct={name: sum(v == 'correct' for v in group.values())
                  for name, group in verdicts.items()}, comparisons=comparisons,
                  strong_screen_vs_random_net=strong,
                  strong_source='historical' if historical else 'reevaluated' if strong is not None else None,
                  evaluation_batch_size=eval_keys[0],
                  historical_grpo_eval_batch_size=previous['runs']['random']['test']['batch_size'] if historical else None,
                  historical_strong_sft_correct=read_json(HISTORICAL_SFT)['correct'] if a.dataset == 'test' else None,
                  screening_interaction_net=interaction,
                  screening_interaction_pp=None if interaction is None else 100 * interaction / len(expected_rows),
                  questions_per_arm=selection['count'],
                  historical_count_matched=selection['historical_count_matched'],
                  scope=('Single seed; previously evaluated official GSM8K test.' if a.dataset == 'test'
                         else 'Single seed; reserved GSM8K training-split questions for evaluation only.'))
    (directory / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
    print(json.dumps({**report, 'comparisons': {name: p['net'] for name, p in comparisons.items()}},
                     ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'coldstart-b16')
    p.add_argument('--model', type=Path, help='Local Base checkpoint used in the previous 4090 experiment')
    p.add_argument('--sft', type=Path, help='Original checkpoint-250 or build the early SFT in --output.')
    p.add_argument('--sft-steps', type=int, default=DEFAULT_SFT_STEPS,
                   help='Early stopping step for batch-16 SFT; reuse it on subsequent stages.')
    commands = p.add_subparsers(dest='stage', required=True)
    commands.add_parser('sft').set_defaults(run=early_sft)
    screening = commands.add_parser('screen')
    screening.add_argument('--screen-batch-size', type=int, default=12)
    screening.set_defaults(run=screen)
    train = commands.add_parser('train')
    train.add_argument('group', choices=GROUPS)
    train.add_argument('--resume', type=Path)
    train.set_defaults(run=grpo)
    test = commands.add_parser('eval')
    test.add_argument('name', choices=EVAL_NAMES)
    test.add_argument('--grpo', type=Path)
    test.add_argument('--batch-size', type=int, default=EVAL_BATCH_SIZE)
    test.set_defaults(run=evaluate)
    summary = commands.add_parser('compare')
    summary.set_defaults(run=compare)
    for command in (test, summary):
        command.add_argument('--dataset', choices=('test', 'holdout'), default='test')
    a = p.parse_args()
    if not 1 <= a.sft_steps < SFT_HORIZON:
        p.error(f'--sft-steps must be between 1 and {SFT_HORIZON - 1}')
    if a.stage != 'compare' and not a.model:
        p.error('--model is required for this stage')
    if a.stage == 'screen' and a.screen_batch_size < 1:
        p.error('--screen-batch-size must be positive')
    if a.stage == 'eval' and a.batch_size < 1:
        p.error('--batch-size must be positive')
    a.run(a)


if __name__ == '__main__':
    main()
