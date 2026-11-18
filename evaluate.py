"""同一个 dev、greedy、512 tokens、历史严格评分器。"""
import argparse
from collections import Counter
from pathlib import Path
from common import build_result
from common import read, record, digest, provenance, assets, load_model, generate

def paired(before, after):
    a = {r['id']:r for r in before}
    b = {r['id']:r for r in after}
    if not a or len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('Empty, duplicate or mismatched evaluation IDs')
    improved, regressed = [], []
    for key in a:
        if a[key]['reference'] != b[key]['reference']:
            raise ValueError('Reference mismatch')
        ca, cb = a[key]['status'] == 'correct', b[key]['status'] == 'correct'
        if cb and not ca:
            improved.append(key)
        if ca and not cb:
            regressed.append(key)
    return dict(n=len(a), improved=improved, regressed=regressed,
                net_correct=len(improved)-len(regressed),
                delta_percentage_points=100*(len(improved)-len(regressed))/len(a))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model')
    p.add_argument('--sft')
    p.add_argument('--grpo')
    p.add_argument('--data')
    p.add_argument('--before')
    p.add_argument('--after')
    p.add_argument('--output', help='NEW prediction JSONL path')
    p.add_argument('--audit-history', action='store_true', help='Verify original frozen runs from Git history')
    a = p.parse_args()
    if a.audit_history:
        import io, subprocess, sys, tarfile, tempfile
        from common import ROOT
        revision = (ROOT/'历史结果/source_revision.txt').read_text(encoding='utf-8').strip()
        archive = subprocess.check_output(['git', '-C', str(ROOT), 'archive', '--format=tar', revision])
        with tempfile.TemporaryDirectory(prefix='qwen-frozen-audit-') as directory:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for item in tar.getmembers():
                    if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                        raise ValueError('Unsafe historical archive path')
                tar.extractall(directory)
            for module in ('scripts.audit_final_evaluation', 'scripts.audit_v2_evidence'):
                subprocess.run([sys.executable, '-m', module], cwd=directory, check=True)
        return
    if not a.output:
        p.error('--output is required for generation or paired comparison')
    out = Path(a.output)
    if a.before or a.after:
        if not (a.before and a.after):
            p.error('--before and --after must be used together')
        import json
        left, right = [json.loads(Path(x).with_suffix('.run.json').read_text(encoding='utf-8'))
                       for x in (a.before, a.after)]
        for key in ('data_sha256', 'do_sample', 'max_new_tokens'):
            if left[key] != right[key]:
                raise ValueError('Evaluation conditions differ: ' + key)
        if left['code']['common.py'] != right['code']['common.py']:
            raise ValueError('Scoring/generation source differs')
        result = paired(read(a.before), read(a.after))
        out.parent.mkdir(parents=True, exist_ok=True)
        record(out, **result)
        print(result)
        return
    if not a.model or not a.data:
        p.error('Generation requires --model and --data')
    if out.exists() or out.with_suffix('.run.json').exists() or out.with_suffix('.summary.json').exists():
        raise FileExistsError('Use a NEW evaluation output')
    rows = read(a.data)
    import json
    from common import ROOT
    split = json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    if [r['id'] for r in rows] != split['splits']['dev']['ids']:
        raise ValueError('Teaching evaluation accepts the original dev split only')
    manifest = json.loads((Path(a.data).parent/'manifest.json').read_text(encoding='utf-8'))
    if digest(a.data) != manifest['files']['dev']['sha256']:
        raise ValueError('Dev content changed since preparation')
    model, tokenizer = load_model(a.model, a.sft, a.grpo)
    model.cuda().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    record(out.with_suffix('.run.json'), model=a.model, sft=a.sft, grpo=a.grpo,
        data_sha256=digest(a.data), code=provenance(), assets=assets(a.model, a.sft, a.grpo),
        do_sample=False, max_new_tokens=512,
        scope='Exploratory dev; historical official test already exposed')
    counts, truncated, tokens = Counter(), 0, 0
    with out.open('x', encoding='utf-8') as f:
        for row in rows:
            text, length, cut = generate(model, tokenizer, row['prompt'])[0]
            result = build_result(row['id'], text, row['gold'], length, cut)
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            counts[result['status']] += 1
            truncated += cut
            tokens += length
    record(out.with_suffix('.summary.json'), n=len(rows), counts=dict(counts), truncated=truncated,
           generated_tokens=tokens, prediction_sha256=digest(out))
    print(dict(counts), 'n=', len(rows), 'truncated=', truncated)

if __name__ == '__main__':
    main()
