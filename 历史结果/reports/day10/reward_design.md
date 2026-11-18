# Day10 - Reward Design and RL Data Admission

## 1. RL data flow

The RL training pool contains 1000 samples.

Each raw sample contains:

- id
- question
- final_answer

The final answer is renamed conceptually as `gold`.

The model receives only:

prompt = prompt_template.format(question=question)

The gold answer is kept outside the model prompt and is used only by the reward function.

## 2. Shared parser

Reward reuses the same strict answer parser as the formal evaluator in `scripts/evaluator.py`.

This prevents training and evaluation from using inconsistent answer rules.

The strict protocol requires exactly one valid final-answer marker.

## 3. Correctness reward

`correctness_reward(response, gold)`

Returns:

- 1.0 when the strict parser succeeds and the parsed numerical answer equals gold
- 0.0 otherwise

This is therefore a strict correctness reward: an implicitly correct number with invalid output format does not receive correctness reward.

## 4. Format reward

`format_reward(response)`

Returns:

- 1.0 when the strict evaluator can extract a valid final answer
- 0.0 otherwise

A response may therefore receive format reward while still receiving zero correctness reward.

Example:

Final answer: 20

with gold = 16:

- correctness_reward = 0
- format_reward = 1

## 5. Reward-hacking checks

Adversarial tests include:

- wrong answer with valid format
- correct number with invalid format
- duplicate Final answer markers
- copied gold number in reasoning with wrong final answer
- empty output
- comma-formatted numbers
- negative values
- decimal equivalence
- dollar-prefixed values
- units after the final number
- trailing text
- multiple conflicting final answers

The current automated reward audit passed all configured adversarial cases.

## 6. RL prompt admission

`data/processed/rl_manifest.jsonl` contains 1000 entries with:

- id
- prompt
- gold

The prompt is rendered using the same template used by SFT and formal evaluation.

A fixed-seed audit of 50 entries found:

- no prompt-rendering failures
- no gold mismatches
- no unfilled `{question}` placeholders

Occurrences where the gold text naturally appears inside the original question are recorded but are not treated as construction-time leakage.

## 7. Reward policy for GRPO

Primary signal:

- correctness reward

Format reward is kept as a separate signal so that its effect can be studied independently.

The project will not interpret increased format compliance as equivalent to increased mathematical reasoning ability.

A correctness-only versus correctness-plus-format comparison is reserved as a reward ablation rather than silently changing the reward definition.

## 8. Admission decision

The current reward/parser and RL manifest have passed the automated admission checks required before rollout experiments.

This does not prove that the reward is universally perfect; new failure modes found during rollout must be added as regression tests before changing the reward policy.
