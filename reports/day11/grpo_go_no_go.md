# Day11 - GRPO Go / No-Go

## 1. Purpose

Before starting GRPO parameter updates, verify that the current SFT policy,
RL training pool, and rollout configuration produce usable relative reward signal.

This is a rollout admission check, not GRPO training.

## 2. Rollout configuration

- policy: Qwen3-1.7B-Base + final SFT LoRA adapter
- RL prompts: 50
- generations per prompt (G): 4
- total rollouts: 200
- temperature: 0.8
- top_p: 0.95
- max_new_tokens: 256
- seed: 42

## 3. Group-level reward signal

- mixed groups: 20/50 (40.00%)
- all-correct groups: 26/50 (52.00%)
- all-wrong groups: 4/50 (8.00%)
- groups with positive reward variance: 20/50

A mixed group contains both rewarded and unrewarded completions, for example:

`[1, 0, 1, 1]`

Such groups provide a non-zero relative correctness signal for GRPO.

All-correct and all-wrong groups have zero variance under the current binary
correctness reward and therefore provide little or no within-group correctness signal.

## 4. Rollout-level observations

- correctness-rewarded rollouts: 151/200 (75.50%)
- format-valid rollouts: 199/200 (99.50%)
- mean generated tokens: 90
- max-length hits: 2/200 (1.00%)
- harmful max-length hits with no parsed final answer: 1/200 (0.50%)

Two rollouts reached the configured generation limit in this run.
At least one still produced a parseable final answer, so max-length hit and
harmful truncation are tracked separately.

## 5. Admission decision

**GRPO admission: GO**

Reasoning:

- 40.00% of prompt groups showed mixed correctness rewards.
- 20/50 groups had non-zero reward variance.
- Harmful max-length truncation was 0.50%.
- The rollout pipeline successfully produced and scored 200 real SFT-policy completions.

This demonstrates that the current policy/data/sampling setup contains usable
relative reward signal and is suitable for a small GRPO training smoke test.

This decision is specific to the current project configuration and should not
be interpreted as a universal GRPO threshold.

## 6. Next step

Proceed to a small GRPO optimization smoke test before any longer RL run.

Day12 must verify:

- policy parameters actually update
- reward logging is correct
- memory usage is acceptable
- checkpoints can be saved
- training does not produce NaN/OOM
