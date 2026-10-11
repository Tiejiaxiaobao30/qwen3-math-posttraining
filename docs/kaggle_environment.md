# Kaggle evaluation environment and compatibility notes

## What is actually preserved

The six official logs, original evaluator and original summaries are under `reports/final/logs/`, `scripts/eval_official_test.py`, and `results/official_test/`. Historical `/kaggle/working/...` and `/kaggle/input/...` paths describe that run and are not a public API. Pass your own local paths through the evaluator CLI. Run modules from the repository root so `scripts.evaluator` resolves.

The local `environment.txt` records the earlier A10/DSW environment (Python 3.11.11, torch 2.6.0+cu124); the local `environment-backup.txt` also predates Kaggle. Neither is a lock file for the final Kaggle runs. A complete Kaggle `pip freeze` and exact PEFT/Transformers versions are not in the final evidence ZIP. No new GPU reproduction has been executed during closeout.

## Reported torchao / PEFT issue

The closeout request reports a compatibility conflict involving **torchao 0.10.0 and PEFT** in Kaggle. The imported final logs do not contain the original traceback or a complete environment snapshot, so the precise failing version combination and the historical workaround cannot be independently reconstructed here.

For this BF16, non-quantized evaluation path, a practical workaround is to remove optional torchao from a disposable Kaggle environment, then restart the kernel before importing PEFT/Transformers:

```bash
python -m pip uninstall -y torchao
```

This is a proposed recovery step for the reported incident, not a claim that it was re-tested locally or that all PEFT/torchao versions are incompatible. If another notebook requires torchao quantization, use a separate compatible environment instead. Do not fix imports by editing installed package source or by inserting temporary absolute model paths into project scripts. Save the traceback and before/after package versions if this issue is investigated again.

`requirements-kaggle.txt` lists direct dependencies only. It intentionally does not invent a final-run version lock or replace Kaggle's CUDA-matched torch installation. Capture Python, GPU/driver, torch/CUDA, Transformers, PEFT, Accelerate, and all package versions for any future serving benchmark, as a separate artifact.

## Model composition and safe reproduction

- Base: the recorded Qwen3-1.7B-Base revision; model weights stay outside Git.
- SFT: Base + the matching seed's SFT adapter.
- GRPO: merge the matching SFT adapter, then attach its second-stage GRPO adapter. Loading the GRPO adapter directly onto Base is a different policy.
- Extra-SFT: merge seed42 SFT, then attach the Extra-SFT adapter; the original evaluator uses `--mode grpo` for this two-stage load path.
- Frozen official scripts retain the historical PEFT metadata cleanup after `merge_and_unload()`. It is version-sensitive and is preserved as run evidence rather than asserted to be necessary for every future PEFT release.

Inspect arguments with `python -m scripts.eval_official_test --help`. Supply `--base-model`, `--test-file`, `--prompt-file`, `--run-name`, and applicable adapters; use a **new** `--results-dir`. The archived evaluator can overwrite same-named outputs and supports partial `--limit` runs; only runs with all 1319 examples belong in the final table. The audit never loads models and is the intended closeout check. No further test-guided tuning is authorized by the experimental policy.
