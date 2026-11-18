"""固定历史 ID 划分，只读取 GSM8K train；不接触官方 test。"""
import argparse
import json
import re
from pathlib import Path
from common import parse_reference
from common import ROOT, PROMPT, read, write, digest, record

REVISION = '740312add88f781978c0658806c59bc2815b9866'

def build(raw, manifest):
    if len(raw) != manifest['total_samples']:
        raise ValueError('Raw train count changed; original IDs would no longer be valid')
    result, seen = {}, set()
    for split in ('sft', 'rl', 'dev'):
        rows = []
        for i in manifest['splits'][split]['ids']:
            question, answer = raw[i]['question'].strip(), raw[i]['answer']
            key = ' '.join(question.split()).casefold()
            if key in seen or answer.count('####') != 1:
                raise ValueError(f'Duplicate question / bad answer at {split}/{i}')
            seen.add(key)
            gold = answer.split('####')[-1].strip()
            parse_reference(gold)
            row = dict(id=i, prompt=PROMPT.format(question=question), gold=gold)
            if split == 'sft':
                completion = re.sub(r'<<[^>]+>>', '', answer)
                row['completion'] = re.sub(r'####\s*(.+)$', r'Final answer: \1', completion).strip()
            rows.append(row)
        result[split] = rows
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-jsonl', help='Optional original-order GSM8K train JSONL')
    p.add_argument('--output', default='data')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a NEW directory')
    if a.raw_jsonl:
        raw = read(a.raw_jsonl)
        source = dict(path=str(Path(a.raw_jsonl).resolve()), sha256=digest(a.raw_jsonl))
    else:
        from datasets import load_dataset
        raw = list(load_dataset('openai/gsm8k', 'main', split='train', revision=REVISION))
        source = dict(dataset='openai/gsm8k', revision=REVISION, split='train')
    manifest = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    data = build(raw, manifest)
    for name, rows in data.items():
        write(out/f'{name}.jsonl', rows)
    record(out/'manifest.json', source=source, split_sha256=digest(ROOT/'split_manifest.json'),
           prompt_sha256=digest(ROOT/'prompt.txt'),
           files={s:dict(count=len(r), sha256=digest(out/f'{s}.jsonl')) for s,r in data.items()})
    print({s:len(r) for s,r in data.items()})

if __name__ == '__main__':
    main()
