"""用同一个 SFT 对原 RL 池每题采样四次，保留答对 1–3 次的题。"""
import argparse
import random
from pathlib import Path
from common import correctness_reward
from common import read, write, record, digest, provenance, assets, load_model, generate

def select(source, results, seed):
    if len(results) != len(source) or [r['id'] for r in results] != [r['id'] for r in source]:
        raise ValueError('Screening must cover original pool exactly once in original order')
    if any(len(r['rewards']) != 4 or any(x not in (0, 1) for x in r['rewards']) for r in results):
        raise ValueError('Expected four binary rewards per question')
    selected = [s for s,r in zip(source, results) if 0 < sum(r['rewards']) < 4]
    control = random.Random(seed).sample(source, len(selected))
    return selected, control

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--sft', required=True)
    p.add_argument('--data', required=True, help='Original RL pool, never dev/test')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Use a NEW screening directory')
    source = read(a.data)
    # Check IDs against the original split; never select dev questions by accident.
    import json
    from common import ROOT
    manifest = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    if [r['id'] for r in source] != manifest['splits']['rl']['ids']:
        raise ValueError('Expected the full original RL split in original order')
    data_manifest = json.loads((Path(a.data).parent/'manifest.json').read_text(encoding='utf-8'))
    if digest(a.data) != data_manifest['files']['rl']['sha256']:
        raise ValueError('RL pool content changed since preparation')
    from transformers import set_seed
    set_seed(a.seed)
    model, tokenizer = load_model(a.model, a.sft)
    model.cuda().eval()
    out.mkdir(parents=True)
    record(out/'run.json', model=a.model, sft=a.sft, seed=a.seed, generations=4,
           temperature=0.8, top_p=0.95, max_tokens=256, data_sha256=digest(a.data),
           assets=assets(a.model, a.sft), code=provenance())
    results = []
    # Flush every question; failed runs retain raw evidence and produce NO train subset.
    with (out/'rollouts.jsonl').open('x', encoding='utf-8') as f:
        for row in source:
            outputs = generate(model, tokenizer, row['prompt'], count=4, sample=True, max_tokens=256)
            result = dict(id=row['id'], outputs=outputs,
                          rewards=[correctness_reward(text,row['gold']) for text,_,_ in outputs])
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            results.append(result)
    selected, control = select(source, results, a.seed)
    if not selected:
        raise ValueError('No partially correct questions; do not train an empty subset')
    write(out/'informative.jsonl', selected)
    write(out/'random.jsonl', control)
    record(out/'manifest.json', complete=True, source_sha256=digest(a.data), count=len(selected),
        run_sha256=digest(out/'run.json'), rollouts_sha256=digest(out/'rollouts.jsonl'),
        files={n:digest(out/n) for n in ['informative.jsonl','random.jsonl']})
    print(f'Selected {len(selected)}/{len(source)}; random control has the SAME count')

if __name__ == '__main__':
    main()
