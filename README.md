# Qwen3 Math Post-Training

Post-training project for mathematical reasoning based on `Qwen/Qwen3-1.7B-Base`.

## Current Progress

- Base model baseline on GSM8K
- Fixed 20-sample smoke set
- Reproducible model revision and environment
- GSM8K data inventory with SHA256 fingerprints

## Model

- Model: `Qwen/Qwen3-1.7B-Base`
- Source: ModelScope
- Precision: BF16

## Dataset

- Dataset: GSM8K
- Config: `main`
- Train: 7473 samples
- Test: 1319 samples
- Smoke baseline: 20 fixed samples from the training split

See `data_inventory.json` for dataset metadata and fingerprints.

## Smoke Baseline

Strict output protocol:

    Final answer: <number>

Result:

- Correct: 10 / 20
- Wrong: 2 / 20
- Format fail: 8 / 20
- Strict smoke accuracy: 50.0%

This is a smoke baseline on the training split, not the final GSM8K test accuracy.

## Run

Example baseline script:

    python scripts/run_smoke_baseline.py

## Roadmap

Base baseline -> LoRA SFT -> fixed evaluation -> GRPO -> controlled comparison -> vLLM deployment
