# GRPO Hyperparameter Ablation

## Goal

Starting from the first successful GRPO configuration, evaluate whether the limited improvement over SFT is mainly caused by ordinary optimization hyperparameters.

Baseline:

| Model | Strict Accuracy |
| --- | ---: |
| Base | 53.40% |
| SFT | 78.00% |
| Initial GRPO | 78.40% |

All experiments use the same fixed dev500 evaluation protocol.

---

## 1. Training-horizon ablation

Only the GRPO training step is varied.

| Model | Correct | Accuracy |
| --- | ---: | ---: |
| SFT | 390/500 | 78.00% |
| GRPO-250 | 379/500 | 75.80% |
| GRPO-500 | 386/500 | 77.20% |
| GRPO-750 | 385/500 | 77.00% |
| GRPO-1000 | 392/500 | 78.40% |

Best checkpoint:

```text
1000 steps
78.40%
```

Conclusion:

Within the tested 0-1000 step range, there is no evidence that the 1000-step policy is over-trained. Earlier checkpoints perform worse and dev accuracy is not strictly monotonic during training.

---

## 2. Beta / KL ablation

Training steps are fixed at 1000 and all other hyperparameters remain unchanged.

| beta | Correct | Accuracy | Gain vs SFT |
| ---: | ---: | ---: | ---: |
| 0.02 | 386/500 | 77.20% | -0.80 pp |
| 0.04 | 392/500 | 78.40% | +0.40 pp |
| 0.08 | 381/500 | 76.20% | -1.80 pp |

Best beta:

```text
beta = 0.04
78.40%
```

Conclusion:

Neither weaker nor stronger KL regularization improved the current GRPO result. Among the tested values, the original beta=0.04 remains best.

---

## 3. Learning-rate ablation

Training steps and beta are fixed:

```text
steps = 1000
beta = 0.04
```

Results:

| Learning Rate | Correct | Accuracy | Gain vs SFT |
| ---: | ---: | ---: | ---: |
| 5e-7 | 377/500 | 75.40% | -2.60 pp |
| 1e-6 | 392/500 | 78.40% | +0.40 pp |
| 2e-6 | 387/500 | 77.40% | -0.60 pp |

Best learning rate:

```text
1e-6
78.40%
```

Conclusion:

The original learning rate remains best. Both smaller and larger update magnitudes reduce dev performance.

---

## 4. Overall conclusion

Three controlled ablations were performed:

```text
training horizon
beta / KL
learning rate
```

The original GRPO configuration remains the best tested setup:

```text
max_steps = 1000
beta = 0.04
learning_rate = 1e-6
```

with:

```text
dev500 strict accuracy = 78.40%
```

Therefore, the current performance bottleneck is unlikely to be solved by simple tuning of these optimization hyperparameters.

The next phase will shift from optimizer-level tuning toward training-signal quality:

- reward design
- rollout reward variance
- RL data difficulty
- mixed-group / informative-prompt selection
- regression control

This changes the research question from:

```text
"Which optimizer hyperparameter is best?"
```

to:

```text
"Is the GRPO learning signal itself informative enough?"
```

---

## 5. Experimental takeaway

The ablation results show that the initial GRPO configuration was already near the best point within the tested optimization hyperparameter range.

The limited SFT → GRPO improvement:

```text
78.00% → 78.40%
```

cannot be explained simply by:

- insufficient training steps
- overly weak or strong KL regularization
- overly small or large learning rate

This suggests that further gains are more likely to come from improving the quality and information density of the RL training signal rather than continuing to sweep standard optimizer hyperparameters.

Day14-Day16 hyperparameter ablation complete.
