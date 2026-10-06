# Day09 - SFT Adapter Reload Audit

## Purpose

Verify that the final LoRA adapter can be reloaded with the original Qwen3-1.7B-Base model in a fresh Python process.

## Artifacts

- Base model: /mnt/workspace/models/Qwen3-1.7B-Base
- Adapter: outputs/sft_full/final_adapter
- Adapter SHA256: 63420f32629bbf452b30b4e5d27060c924f01b37dea013a073bb2956cea2437f

## Reload command

python -m scripts.run_dev_baseline --limit 3 --adapter outputs/sft_full/final_adapter

## Smoke result

- samples: 3
- correct: 2
- wrong: 1
- format_fail: 0
- truncated: 0
- strict_accuracy: 66.67%
- format_success_rate: 100.00%

The 3-sample accuracy is not a formal quality metric. This run only verifies that Base + final_adapter can be restored in a fresh process and used for inference.

## Checkpoint vs adapter reload

- checkpoint resume: restores model plus optimizer, scheduler, trainer state and global step for continued training
- final_adapter reload: restores the trained LoRA result for inference/evaluation
