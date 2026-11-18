"""少量共享工具；模型依赖只在真正运行时导入。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT = (ROOT / 'prompt.txt').read_text(encoding='utf-8')

def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def assets(model, sft=None, grpo=None):
    # Hash actual local weights, not just the directory name or declared revision.
    result = {}
    for name, directory in [('base', model), ('sft', sft), ('grpo', grpo)]:
        if directory:
            path = Path(directory)
            files = sorted(p for p in path.iterdir() if p.suffix in ('.json', '.safetensors', '.bin'))
            result[name] = {p.name: digest(p) for p in files}
    return result

def record(path, **values):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(values, f, ensure_ascii=False, indent=2)

def provenance():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            [*sorted(ROOT.glob('*.py')), ROOT/'prompt.txt', ROOT/'split_manifest.json']}

def load_model(model_dir, sft=None, grpo=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if sft:
        model = PeftModel.from_pretrained(model, sft).merge_and_unload()
    if grpo:
        if not sft:
            raise ValueError('GRPO adapter requires the SAME merged SFT starting point')
        model = PeftModel.from_pretrained(model, grpo)
    return model, tokenizer

def generate(model, tokenizer, prompt, count=1, sample=False, max_tokens=512):
    import torch
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    kwargs = dict(do_sample=sample, num_return_sequences=count, max_new_tokens=max_tokens,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if sample:
        kwargs.update(temperature=0.8, top_p=0.95)
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    sequences = [ids[inputs['input_ids'].shape[1]:].tolist() for ids in output]
    rows = []
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    for ids in sequences:
        # Ignore padding after first EOS when counting generated tokens.
        end = next((i+1 for i,t in enumerate(ids) if t in eos), len(ids))
        tokens = ids[:end]
        rows.append((tokenizer.decode(tokens, skip_special_tokens=True), len(tokens),
                     len(tokens) >= max_tokens and tokens[-1] not in eos))
    return rows


# Historical strict scoring logic; original source preserved in 历史备份/旧实验记录.
import re
from decimal import Decimal


FINAL_ANSWER_PATTERN = re.compile(
    r"Final answer:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE
)


def extract_answer(text):
    if not text:
        return None

    if text.lower().count("final answer:") != 1:
        return None

    match = FINAL_ANSWER_PATTERN.search(text)

    if match is None:
        return None

    number_text = match.group(1).replace(",", "")

    return Decimal(number_text)


def parse_reference(text):
    number_text = text.replace(",", "").strip()
    return Decimal(number_text)


def score(prediction, reference):
    pred_value = extract_answer(prediction)

    if pred_value is None:
        return "format_fail"

    ref_value = parse_reference(reference)

    if pred_value == ref_value:
        return "correct"

    return "wrong"


def build_result(
    sample_id,
    prediction,
    reference,
    generated_tokens,
    truncated,
):
    parsed = extract_answer(prediction)
    status = score(prediction, reference)

    return {
        "id": sample_id,
        "reference": reference,
        "prediction": prediction,
        "parsed_answer": (
            str(parsed) if parsed is not None else None
        ),
        "status": status,
        "generated_tokens": generated_tokens,
        "truncated": truncated,
    }




def correctness_reward(response, gold):
    """
    Strict correctness reward.

    1.0:
        response follows the strict Final answer format
        AND parsed answer equals gold.

    0.0:
        format cannot be parsed
        OR numerical answer is wrong.
    """
    predicted = extract_answer(response)

    if predicted is None:
        return 0.0

    gold_value = parse_reference(str(gold))

    return 1.0 if predicted == gold_value else 0.0
