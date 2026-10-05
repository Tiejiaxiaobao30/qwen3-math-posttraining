# Evaluation Protocol

## Dataset

- Dataset: GSM8K
- Evaluation split: fixed dev
- Number of samples: 500
- Split source: `split_manifest.json`
- Dev data: `data/processed/dev.jsonl`
- Official GSM8K test is not used for development or model selection.

## Model

- Base model: `Qwen/Qwen3-1.7B-Base`
- Source: ModelScope
- Revision: `43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3`
- Precision: BF16

## Prompt

Prompt source:

`configs/sft_prompt.txt`

Prompt:

    Solve the following math problem step by step.
    End with exactly: Final answer: <number>

    {question}

## Generation

- `do_sample = False`
- `max_new_tokens = 512`
- Greedy deterministic generation
- `pad_token_id` falls back to `eos_token_id`

## Strict answer protocol

A valid answer must end with exactly one:

    Final answer: <number>

Accepted numeric forms include:

- integer: `72`
- negative: `-5`
- decimal: `3.5`
- thousands separator: `1,234`
- optional dollar sign: `$72`

The evaluator rejects:

- missing `Final answer:`
- multiple `Final answer:` occurrences
- missing numeric value
- extra text after the numeric answer

## Status

Each sample is assigned exactly one primary status:

- `correct`: parsed successfully and equals reference
- `wrong`: parsed successfully but differs from reference
- `format_fail`: strict parser cannot extract an answer

Truncation is recorded independently:

- `truncated = true / false`

Therefore:

    correct + wrong + format_fail = total

## Metrics

- Strict accuracy = `correct / total`
- Format success rate = `(correct + wrong) / total`
- Truncation rate = `truncated / total`

## Base dev baseline

- Total: 500
- Correct: 267
- Wrong: 64
- Format fail: 169
- Truncated: 1
- Strict accuracy: 53.40%
- Format success rate: 66.20%
- Truncation rate: 0.20%

Raw results:

`results/base_dev_500.jsonl`

Metrics:

`results/base_dev_metrics.json`
