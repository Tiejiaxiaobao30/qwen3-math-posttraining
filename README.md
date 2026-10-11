# Qwen3 Math Post-Training

Mathematical reasoning post-training with `Qwen/Qwen3-1.7B-Base`: LoRA SFT, online binary-correctness-reward GRPO, development ablations, an Extra-SFT control, and two-seed evaluation on GSM8K.

## Final sealed official-test results

| Model | Correct / 1319 | Strict accuracy | Wrong | Format fail | Truncated |
|---|---:|---:|---:|---:|---:|
| Base | 665 / 1319 | 50.42% | 264 | 390 | 9 |
| SFT seed42 | 902 / 1319 | 68.39% | 408 | 9 | 8 |
| GRPO seed42 | 893 / 1319 | 67.70% | 413 | 13 | 11 |
| Extra-SFT control | 898 / 1319 | 68.08% | 413 | 8 | 6 |
| SFT seed123 | 897 / 1319 | 68.01% | 412 | 10 | 6 |
| GRPO seed123 | 882 / 1319 | 66.87% | 422 | 15 | 10 |

**The +0.40 percentage-point GRPO gain on fixed dev500 did not generalize to the sealed official test.** Both GRPO policies scored below their corresponding SFT baseline. Exact count-based test changes are −0.68pp (seed42, −9 answers) and −1.14pp (seed123, −15 answers). The two-seed mean is SFT **68.20%**, GRPO **67.29%** (−0.91pp); these seeds share the same 1319 questions.

| Seed | Dev500 SFT | Dev500 GRPO | Dev change | Official-test change |
|---|---:|---:|---:|---:|
| 42 | 78.00% (390/500) | 78.40% (392/500) | +0.40pp | −0.68pp |
| 123 | 76.00% (380/500) | 76.40% (382/500) | +0.40pp | −1.14pp |

SFT is the main effective stage in this experiment: seed42 improves from Base 50.42% to 68.39% (+17.97pp). Extra-SFT reaches 68.08% (−0.30pp versus SFT seed42) under its tested configuration. The control uses the same RL1000 question pool but is **not compute-matched** to GRPO. These are descriptive results for this setup, not evidence that GRPO is universally ineffective or that the small differences are statistically significant.

## Experimental discipline

- Development: fixed dev500 from GSM8K train; SFT2000 and RL1000 preparation are recorded in `split_manifest.json` and the data-pipeline scripts.
- Final prompt, strict parser, dev data, five adapter fingerprints, decoding, and six evaluated models were recorded in the [pre-test freeze manifest](results/frozen_eval_manifest.json).
- Official test was opened after freeze according to the experiment record; **no post-test tuning**, further training, hyperparameter search, or third seed. Both seeds and the control are reported.
- Greedy generation (`do_sample=False`), 512 new-token cap, BF16, exactly one final `Final answer: <number>`; format failures count as incorrect. Truncation is an independent flag.
- Manifest `SEALED` is the preserved pre-test snapshot. Its timestamp has no timezone, and the local audit cannot independently establish the original execution chronology or absence of earlier test access.

## Reviewable evidence

- [Final report and limitations](reports/final/official_test_report.md)
- [Paired SFT→GRPO analysis and failure counts](reports/final/verification.json)
- [Artifact hashes and archive provenance](reports/final/artifact_provenance.json)
- [Six predictions and summaries, with separately named one-example smoke files](results/official_test/)
- [Original official logs](reports/final/logs/)
- [Kaggle environment and portability notes](docs/kaggle_environment.md)
- [Evaluation protocol](eval_protocol.md), [model revision](model_revision.txt), and historical [development reports](reports/)

Historical day/ablation reports describe the state when written (including "test sealed" and planned next steps); the final report above supersedes their current-status conclusions. No reward-ablation result is claimed without corresponding evidence.

## Recheck the stored results (CPU, Python standard library)

From the repository root:

```bash
python -m scripts.audit_final_evaluation
python -m tests.test_reward
# Optional with pytest installed: python -m pytest tests/test_evaluator.py -q
```

The existing reward audit has 9 cases; the evaluator suite has 17 checks.

The audit rescored all 7914 full-test predictions with the frozen parser, checked IDs, references, counts, decoding metadata, source-file hashes, and four stored dev runs. It optionally verifies the dev data hash when the ignored local `data/processed/dev.jsonl` exists. It performs no generation or tuning. `--output NEW_PATH.json` writes a new audit artifact and refuses to overwrite an existing one.

The [official evaluator](scripts/eval_official_test.py) is preserved byte-for-byte from the run backup and accepts model, adapter, test, prompt, device, and result-directory arguments. Invoke as `python -m scripts.eval_official_test --help` in an appropriate GPU environment. Original summaries retain historical machine paths as evidence, not installation requirements. Do not rerun into the archived results directory.

Seed123 configuration files use a relative `models/Qwen3-1.7B-Base` path; set it to your local model directory before any separate reproduction experiment. Dev evaluators expose `--base-model`, `--dev-file`, `--prompt-file`, `--device`, and `--results-dir`; their original sources are preserved in `reports/final/kaggle_sources/`. No post-test training is part of this release.

## Artifact policy and remaining work

Model weights, checkpoints, tokenizer bundles, ZIP/TAR backups, and the 8GB full backup remain outside Git. The repository includes the frozen GSM8K main test JSONL copy used by these runs (source: GSM8K main via ModelScope, recorded in `data_inventory.json`) and small evaluation evidence.

Still pending: **vLLM deployment benchmark** (TTFT, latency, throughput, fixed prompts/load/hardware/software and model composition) and final evidence-based resume bullets/interview notes. No serving performance is claimed yet.
