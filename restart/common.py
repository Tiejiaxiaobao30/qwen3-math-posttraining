import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def runtime_layout(world_size, global_batch=12, generations=4):
    if world_size not in (1, 2, 3):
        raise ValueError('Supported world sizes: 1, 2, 3')
    if global_batch % world_size or global_batch % generations:
        raise ValueError('Global completion batch must divide world size and G')
    return dict(per_device_train_batch_size=1,
                gradient_accumulation_steps=global_batch // world_size,
                steps_per_generation=global_batch // world_size)


def verify_assets(model_dir, adapter_dir):
    model_dir, adapter_dir = Path(model_dir), Path(adapter_dir)
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    adapter = json.loads((adapter_dir / 'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3' or config.get('hidden_size') != 2048:
        raise ValueError('Expected Qwen3-1.7B-Base config; do not substitute Instruct/other sizes')
    if 'Qwen3-1.7B-Base' not in adapter.get('base_model_name_or_path', ''):
        raise ValueError('SFT adapter base does not match Qwen3-1.7B-Base')
    expected = json.loads(Path('restart/configs/assets.json').read_text(encoding='utf-8'))
    for name, digest in expected['sft_files'].items():
        if sha256(adapter_dir / name) != digest:
            raise ValueError(f'SFT asset fingerprint mismatch: {name}')
    if not list(model_dir.glob('*.safetensors')):
        raise FileNotFoundError('Base model weights missing')
    return expected
