"""Rebuild the historical RL/dev rows from a pinned public TRAIN split."""
import argparse
import json
import re
from pathlib import Path
from restart.common import sha256, write_json, write_jsonl, read_jsonl
from scripts.evaluator import parse_reference

REVISION = '740312add88f781978c0658806c59bc2815b9866'


def normalized_question(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def build(raw_rows, manifest, prompt):
    if len(raw_rows) != manifest['total_samples']:
        raise ValueError('Source row count changed; stop instead of shifting IDs')
    result = {}
    seen = set()
    # Exclude semantic-exact duplicates of any historical SFT question.
    sft_questions = {normalized_question(raw_rows[i]['question'])
                     for i in manifest['splits']['sft']['ids']}
    for split in ('rl', 'dev'):
        rows = []
        for i in manifest['splits'][split]['ids']:
            raw = raw_rows[i]
            q = normalized_question(raw['question'])
            if q in seen or q in sft_questions:
                raise ValueError(f'Duplicate/cross-split question at {split}/{i}; audit explicitly')
            seen.add(q)
            if raw['answer'].count('####') != 1:
                raise ValueError(f'Invalid answer delimiter: {i}')
            gold = raw['answer'].split('####')[-1].strip()
            parse_reference(gold)
            rows.append({'id': i, 'prompt': prompt.format(question=raw['question']), 'gold': gold})
        result[split] = rows
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['hf', 'local'], default='hf')
    p.add_argument('--raw-jsonl', help='Local GSM8K TRAIN, original row order; NOT test')
    a = p.parse_args()
    out = Path('data/restart')
    if out.exists():
        raise FileExistsError('data/restart exists; preserve it or move it explicitly')
    if a.source == 'hf':
        from datasets import load_dataset
        rows = list(load_dataset('openai/gsm8k', 'main', split='train', revision=REVISION))
        provenance = {'dataset': 'openai/gsm8k', 'revision': REVISION, 'split': 'train'}
    else:
        if not a.raw_jsonl:
            p.error('--raw-jsonl required')
        rows = read_jsonl(a.raw_jsonl)
        provenance = {'source': 'local train JSONL', 'sha256': sha256(a.raw_jsonl)}
    manifest = json.loads(Path('split_manifest.json').read_text(encoding='utf-8'))
    prompt = Path('configs/sft_prompt.txt').read_text(encoding='utf-8')
    data = build(rows, manifest, prompt)
    expected = json.loads(Path('restart/configs/assets.json').read_text(encoding='utf-8'))
    # Validate logical content against the ACTUAL local legacy files, not merely numeric IDs.
    for split, values in data.items():
        canonical = '\n'.join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in values) + '\n'
        import hashlib
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest != expected['data_canonical_sha256'][split]:
            raise ValueError(f'{split} differs from original local rows; do not train')
    for split, values in data.items():
        write_jsonl(out / f'{split}.jsonl', values)
    write_json(out / 'manifest.json', dict(provenance=provenance,
               split_manifest_sha256=sha256('split_manifest.json'),
               prompt_sha256=sha256('configs/sft_prompt.txt'),
               files={s: {'count': len(v), 'sha256': sha256(out / f'{s}.jsonl')}
                      for s, v in data.items()},
               evaluation_scope='Historical dev only; official test already exposed; no new holdout claim'))
    print('Prepared historical RL1000/dev500; no training, no official-test download.')


if __name__ == '__main__':
    main()
