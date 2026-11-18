"""Parallel-friendly official test evaluation, 8 questions per GPU forward batch."""
import argparse
from collections import Counter
from decimal import InvalidOperation
from pathlib import Path
import json
import os

from common import digest, load_model, read, record
from common import build_result

MAX_TOKENS = 512
EXPECTED = {
    'base': '6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841',
    'sft': '63420f32629bbf452b30b4e5d27060c924f01b37dea013a073bb2956cea2437f',
    'test': 'd9a6a8c13b99ac41d22bfdb5912550fe202f92dca88e3565190bd9ca044ada0e',
    'prompt': '4343a37c9e2e5603f5c63ac0fa9d5767d0f02e102b058f5554cc92edd736f828',
}


def generate_batch(model, tokenizer, prompts, max_tokens=MAX_TOKENS):
    import torch

    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, padding=True, return_tensors='pt').to(model.device)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_tokens = {eos} if isinstance(eos, int) else set(eos or [])
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1,
            num_return_sequences=1, max_new_tokens=max_tokens,
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    for tokens in output[:, inputs['input_ids'].shape[1]:].tolist():
        # Remove batch padding after the first genuine stop token.
        for i, token in enumerate(tokens):
            if token in stop_tokens:
                yield tokenizer.decode(tokens[:i+1], skip_special_tokens=True), i+1, False
                break
        else:
            yield tokenizer.decode(tokens, skip_special_tokens=True), len(tokens), len(tokens) >= max_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['random', 'informative'])
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    root = Path(__file__).resolve().parent
    base = Path(os.environ.get('MODEL') or Path.home()/'models/Qwen3-1.7B-Base').resolve()
    sft = root/'outputs/sft_full/final_adapter'
    run = root/f'outputs/final_grpo/{args.group}42-b8-e2-parallel'
    adapter = run/'final_adapter'
    test = root/'data/official/gsm8k_test_1319.jsonl'
    prompt = root/'prompt.txt'
    destination = root/f'outputs/final_grpo/official-test-b8-e2-eval-b{args.batch_size}'
    if not (run/'metrics-step476.json').is_file():
        raise FileNotFoundError(f'Missing step-476 metrics in {run}')
    for key, path in [('base',base/'model.safetensors'), ('sft',sft/'adapter_model.safetensors'),
                      ('test',test), ('prompt',prompt)]:
        if digest(path) != EXPECTED[key]:
            raise ValueError(f'SHA256 mismatch for {path}')
    rows = read(test)
    if len(rows) != 1319 or len({row['id'] for row in rows}) != 1319:
        raise ValueError('Expected all 1,319 distinct frozen test items')
    template = prompt.read_text(encoding='utf-8')
    prediction_path = destination/f'{args.group}_test1319.jsonl'
    summary_path = destination/f'{args.group}_test1319_summary.json'
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError(f'Preserve existing {args.group} outputs before rerunning')

    model, tokenizer = load_model(base, sft, adapter)
    model.cuda().eval()
    destination.mkdir(parents=True,exist_ok=True)
    counts = Counter(correct=0, wrong=0, format_fail=0, truncated=0)
    with prediction_path.open('x',encoding='utf-8') as stream:
        for start in range(0,len(rows),args.batch_size):
            batch = rows[start:start+args.batch_size]
            outputs = generate_batch(model,tokenizer,[template.format(question=row['question']) for row in batch])
            for row,(text,length,cut) in zip(batch,outputs,strict=True):
                try:
                    result = build_result(row['id'],text,row['final_answer'],length,cut)
                except InvalidOperation:
                    result = build_result(row['id'],'',row['final_answer'],length,cut)
                    result['prediction'] = text
                counts[result['status']] += 1
                counts['truncated'] += cut
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            print(f'{start+len(batch)}/{len(rows)}  correct={counts["correct"]}',flush=True)
    record(summary_path, group=args.group, total=len(rows), **counts,
           strict_accuracy=counts['correct']/len(rows), batch_size=args.batch_size,
           do_sample=False, max_new_tokens=MAX_TOKENS, test_sha256=EXPECTED['test'],
           prompt_sha256=EXPECTED['prompt'], base_sha256=EXPECTED['base'], sft_sha256=EXPECTED['sft'],
           grpo_sha256=digest(adapter/'adapter_model.safetensors'), predictions_sha256=digest(prediction_path))
    print(f'{args.group}: {counts["correct"]}/{len(rows)} ({100 * counts["correct"]/len(rows):.2f}%)')


if __name__ == '__main__':
    main()
