import argparse
import json
from restart.common import read_jsonl, write_json


def paired(before, after):
    a = {r['id']:r for r in before}
    b = {r['id']:r for r in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('Mismatched/duplicate evaluation IDs')
    improved, regressed = [], []
    for key in a:
        if a[key]['gold'] != b[key]['gold']:
            raise ValueError('Gold mismatch')
        ca, cb = a[key]['status'] == 'correct', b[key]['status'] == 'correct'
        if cb and not ca:
            improved.append(key)
        if ca and not cb:
            regressed.append(key)
    if not a:
        raise ValueError('Empty evaluation')
    return dict(n=len(a), improved=improved, regressed=regressed,
                net_correct=len(improved)-len(regressed),
                delta_percentage_points=100*(len(improved)-len(regressed))/len(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sa = json.load(open(a.before + '.summary.json', encoding='utf-8'))['provenance']
    sb = json.load(open(a.after + '.summary.json', encoding='utf-8'))['provenance']
    for key in ('dev_sha256', 'parser_sha256', 'sft_sha256', 'decoding'):
        if sa[key] != sb[key]:
            raise ValueError('Noncomparable evaluation: '+key)
    report = paired(read_jsonl(a.before), read_jsonl(a.after))
    write_json(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
