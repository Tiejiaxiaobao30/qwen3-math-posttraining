"""Single-GPU greedy evaluation on the fixed historical dev500 ONLY."""
import argparse
import json
from pathlib import Path
from collections import Counter
from restart.common import read_jsonl, write_jsonl, write_json, sha256, verify_assets
from restart.rewards import components


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--sft-adapter', required=True)
    p.add_argument('--grpo-adapter', help='Omit to evaluate SFT baseline')
    p.add_argument('--output', required=True)
    p.add_argument('--swanlab-mode', choices=['cloud', 'offline', 'disabled'], default='cloud')
    a = p.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    verify_assets(a.model_dir, a.sft_adapter)
    path = Path('data/restart/dev.jsonl')
    manifest = json.loads(Path('data/restart/manifest.json').read_text(encoding='utf-8'))
    if sha256(path) != manifest['files']['dev']['sha256']:
        raise ValueError('Dev data hash changed')
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                  local_files_only=True, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
    if a.grpo_adapter:
        model = PeftModel.from_pretrained(model, a.grpo_adapter)
    model = model.to('cuda').eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    rows = read_jsonl(path)
    results = []
    for row in rows:
        inputs = tokenizer(row['prompt'], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                                     pad_token_id=tokenizer.eos_token_id)
        ids = output[0, inputs['input_ids'].shape[1]:].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        correct, fmt = components(text, row['gold'])
        results.append(dict(id=row['id'], gold=row['gold'], prediction=text,
            status='correct' if correct else ('wrong' if fmt else 'format_fail'),
            generated_tokens=len(ids), truncated=len(ids) == 512 and ids[-1] != tokenizer.eos_token_id))
        if len(results) % 25 == 0:
            print(f'{len(results)}/{len(rows)}', flush=True)
    counts = Counter(r['status'] for r in results)
    metrics = dict(counts, n=len(results), accuracy=counts['correct']/len(results),
        truncated=sum(r['truncated'] for r in results),
        mean_generated_tokens=sum(r['generated_tokens'] for r in results)/len(results))
    provenance = dict(dev_sha256=sha256(path), parser_sha256=sha256('scripts/evaluator.py'),
        sft_sha256=sha256(Path(a.sft_adapter)/'adapter_model.safetensors'),
        grpo_sha256=sha256(Path(a.grpo_adapter)/'adapter_model.safetensors') if a.grpo_adapter else None,
        decoding={'do_sample':False,'max_new_tokens':512}, scope='Historical development, not fresh held-out test')
    write_jsonl(a.output, results)
    write_json(a.output + '.summary.json', dict(metrics=metrics, provenance=provenance))
    print(json.dumps(metrics, indent=2))
    if a.swanlab_mode != 'disabled':
        import swanlab
        swanlab.init(project='qwen3-reward-restart', experiment_name='eval-' + Path(a.output).stem,
                     config=provenance, mode=a.swanlab_mode)
        swanlab.log({'dev/'+k:v for k,v in metrics.items()})
        swanlab.finish()


if __name__ == '__main__':
    main()
