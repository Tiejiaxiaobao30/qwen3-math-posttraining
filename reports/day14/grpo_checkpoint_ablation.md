# Day14｜GRPO Checkpoint Ablation

Only training horizon is varied; all other GRPO hyperparameters remain fixed.

| Model | Correct | Strict Accuracy | Gain vs SFT |
| --- | ---: | ---: | ---: |
| SFT | 390/500 | 78.00% | 0.00 pp |
| GRPO-250 | 379/500 | 75.80% | -2.20 pp |
| GRPO-500 | 386/500 | 77.20% | -0.80 pp |
| GRPO-750 | 385/500 | 77.00% | -1.00 pp |
| GRPO-1000 | 392/500 | 78.40% | +0.40 pp |

## Best checkpoint

- Step: 1000
- Strict accuracy: 78.40%
- Gain vs SFT: +0.40 pp
