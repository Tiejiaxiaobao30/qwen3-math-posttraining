# Extra-SFT Fair Control

## Goal

Evaluate whether the improvement from SFT to GRPO can be explained simply by performing additional training on the same RL prompt pool.

The comparison starts from the same SFT policy.

Two branches are considered:

```text
SFT policy
├── Extra-SFT on RL1000 reference solutions
└── GRPO on RL1000 prompts with reward feedback
```

The official GSM8K test remains sealed. All comparisons in this report use the fixed dev500 split.

---

## Experimental setup

### Shared starting point

Base model:

```text
Qwen3-1.7B-Base
```

Original SFT:

```text
SFT training samples = 2000
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
target modules =
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Original SFT dev500 result:

```text
correct = 390 / 500
strict accuracy = 78.00%
format success = 99.60%
```

### Extra-SFT control

The original SFT adapter is first merged into the base model.

A fresh LoRA adapter is then trained using the same 1000-question RL prompt pool, but with GSM8K reference reasoning as supervised targets.

```text
training samples = 1000
batch size = 2
max steps = 500
learning rate = 2e-4
max length = 512
seed = 42
```

500 optimizer steps correspond to one pass over the 1000 Extra-SFT examples with batch size 2.

This should not be interpreted as compute-equivalent to GRPO. Runtime and training mechanism differ.

---

## Fixed dev500 results

| Model | Correct | Wrong | Format Fail | Truncated | Strict Accuracy | Format Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 267 | 64 | 169 | 1 | 53.40% | 66.20% |
| SFT | 390 | 108 | 2 | 0 | 78.00% | 99.60% |
| Extra-SFT | 382 | 115 | 3 | 1 | 76.40% | 99.40% |
| GRPO | 392 | 104 | 4 | 1 | 78.40% | 99.20% |

---

## Main comparison

### SFT → Extra-SFT

```text
78.00% → 76.40%
change = -1.60 percentage points
net = -8 correct examples / 500
```

Additional supervised training on the RL1000 reference solutions did not improve fixed-dev accuracy under the tested configuration.

### SFT → GRPO

```text
78.00% → 78.40%
change = +0.40 percentage points
net = +2 correct examples / 500
```

### Extra-SFT → GRPO

```text
76.40% → 78.40%
difference = +2.00 percentage points
net = +10 correct examples / 500
```

---

## Interpretation

The Extra-SFT result provides an important control for the GRPO experiment.

The GRPO improvement cannot be explained simply by saying that the model received another training stage on the same 1000-question pool.

Under the tested configurations:

```text
SFT        = 78.00%
Extra-SFT  = 76.40%
GRPO       = 78.40%
```

Extra-SFT reduced dev accuracy while preserving nearly perfect output-format compliance.

This suggests that the Extra-SFT regression is primarily related to answer correctness rather than a collapse of the output protocol.

However, this experiment does not prove that GRPO is universally better than continued supervised fine-tuning.

Important limitations remain:

1. only one seed has been evaluated so far;
2. Extra-SFT and GRPO are not compute-equivalent training procedures;
3. the observed GRPO gain over SFT is only +0.40 percentage points;
4. conclusions are specific to the tested model, dataset split, reward, and training configuration.

The appropriate conclusion is therefore:

> Under the tested setup, GRPO outperformed an additional supervised fine-tuning control using the same RL1000 question pool on the fixed dev500 evaluation.

---

## Why Extra-SFT will not be tuned further

Extra-SFT is used as a pre-declared control rather than as a new hyperparameter-search branch.

After observing the 76.40% result, its learning rate, number of steps, and LoRA settings will not be repeatedly tuned to chase a better dev score.

Doing so would weaken its role as a controlled comparison and introduce additional dev-set selection.

---

## Current project result

```text
Base
53.40%
   |
   | LoRA SFT
   v
SFT
78.00%
   |
   +---- Extra-SFT fair control
   |       76.40%
   |
   +---- GRPO
           78.40%
```

Previous GRPO ablations additionally showed:

```text
checkpoint:
250  = 75.80%
500  = 77.20%
750  = 77.00%
1000 = 78.40%

beta:
0.02 = 77.20%
0.04 = 78.40%
0.08 = 76.20%

learning rate:
5e-7 = 75.40%
1e-6 = 78.40%
2e-6 = 77.40%
```

The next stage therefore moves away from ordinary optimizer tuning.

---

## Next experiment

Next:

```text
reward ablation
+
second-seed validation
```

Planned comparison:

```text
correctness-only reward
vs
correctness + format reward
```

After the reward design and second seed are evaluated, the final prompt, parser, reward, checkpoint, decoding settings, and evaluation configuration will be frozen.

Only then will the sealed GSM8K official test be evaluated.
