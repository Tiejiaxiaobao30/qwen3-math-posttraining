# Day13｜Qwen3-1.7B 正式 GRPO 实验

## 1. 实验目标

在 SFT policy 基础上进行真实在线 GRPO post-training，并完成训练、重载、评估与误差分析闭环。

模型路径：

Base
→ 加载 SFT LoRA
→ merge 得到 W_sft
→ 挂载 fresh GRPO LoRA
→ 在线 GRPO

最终 policy：

W_sft + ΔW_grpo

---

## 2. 正式 GRPO 配置

- RL prompts: 1000
- max_steps: 1000
- per_device_train_batch_size: 4
- gradient_accumulation_steps: 1
- num_generations: 4
- learning_rate: 1e-6
- temperature: 0.8
- top_p: 0.95
- max_completion_length: 256
- beta: 0.04
- num_iterations: 1
- loss_type: bnpo
- reward: strict final-answer correctness

单卡情况下：

4 batch slots / G=4
= 1 unique prompt per optimizer step

num_iterations=1，因此训练闭环为：

rollout
→ reward
→ advantage
→ backward
→ optimizer.step
→ updated policy
→ new rollout

---

## 3. 1000-step 训练结果

- global_step: 1000
- optimizer steps recorded: 1000
- nonzero-gradient steps: 994
- changed LoRA tensors: 392
- changed parameter values: 17,432,180
- max_abs_change: approximately 1.34e-4
- peak GPU allocated: approximately 11.20 GB
- peak GPU reserved: approximately 17.94 GB

说明 GRPO LoRA 发生了真实参数更新。

---

## 4. 数值稳定性

训练日志仅在 step 120 出现一次孤立 diagnostic NaN，涉及：

- KL
- clip-ratio statistics

该 step 的 loss、grad_norm、reward 等仍为有限值，训练随后正常运行至 step 1000。

最终 adapter 独立检查：

- tensors: 392
- values: 17,432,576
- non-finite tensors: 0

结论：

最终 GRPO adapter 参数全部 finite。

---

## 5. 固定 dev500 结果

评估设置与 Base / SFT baseline 对齐：

- same dev500
- same prompt
- same strict evaluator
- do_sample=False
- max_new_tokens=512

| Model | Correct | Strict Accuracy |
| --- | ---: | ---: |
| Base | 267 / 500 | 53.40% |
| SFT | 390 / 500 | 78.00% |
| GRPO | 392 / 500 | 78.40% |

GRPO：

- correct: 392
- wrong: 104
- format_fail: 4
- truncated: 1
- format success: 99.20%
- truncation rate: 0.20%

变化：

- Base → SFT: +24.60 percentage points
- SFT → GRPO: +0.40 percentage points

因此只能表述为小幅正向变化，而非显著提升。

---

## 6. SFT → GRPO paired transition

- correct → correct: 379
- correct → wrong: 10
- correct → format_fail: 1
- wrong → correct: 13
- wrong → wrong: 94
- wrong → format_fail: 1
- format_fail → correct: 0
- format_fail → wrong: 0
- format_fail → format_fail: 2

关键结果：

- improved to correct: 13
- regressed from correct: 11
- net gain: +2

因此 78.00% → 78.40% 背后存在明显 policy behavior redistribution，而不是所有样本都稳定提升。

---

## 7. 24-case manual audit

人工分析全部 24 个 correctness-flip cases。

### Improved cases 中观察到

GRPO 修复了部分：

- arithmetic errors
- repeated-period reasoning
- time-unit reasoning
- percentage base selection
- ratio application
- capacity reasoning
- unit conversion
- state update
- multi-stage cost aggregation
- entity/count relationships

### Regressed cases 中观察到

主要问题包括：

- condition omission
- state-update omission
- entity/relation misbinding
- arithmetic/algebra regression
- temporal reference confusion
- fraction/complement confusion
- format regression

其中多条件状态保持和实体关系绑定是较明显的 regression 来源。

---

## 8. Reward 局限

当前 reward 仅判断：

strict parser succeeds
AND
final answer == gold

它不会直接检查：

- reasoning chain 是否正确
- 中间计算是否合理
- 是否覆盖全部条件
- entity binding 是否正确
- reasoning consistency

人工审计还发现疑似 noisy-label / reasoning inconsistency case。

因此：

final-answer correctness reward
!=
guaranteed reasoning quality

---

## 9. 当前结论

本轮正式 GRPO：

SFT 78.00%
→ GRPO 78.40%

GRPO 确实产生了有效在线 RL 学习信号，也修复了一部分 SFT 错题，但同时引入新的 regression。

更准确的结论是：

GRPO 对 policy 行为进行了有效重塑，在部分数学结构上获得收益，但 reasoning stability、关系保持与 regression control 仍需要进一步优化。

后续实验重点：

- beta / KL constraint
- training steps
- reward design
- data quality
- regression control

---

## 10. Reproducibility artifacts

Training:

- configs/grpo_smoke.json
- configs/grpo_full.json
- scripts/train_grpo_smoke.py

Reload:

- scripts/reload_grpo_smoke.py

Evaluation:

- scripts/eval_grpo_dev.py

Analysis:

- scripts/analyze_sft_grpo_transitions.py
- scripts/build_sft_grpo_flip_audit.py

Results:

- results/grpo_full_metrics.json
- results/grpo_final_adapter.sha256
- results/grpo_dev_500.jsonl
- results/grpo_dev500_summary.json
- results/sft_grpo_flip_24.jsonl

Final adapter:

outputs/grpo_full/final_adapter

模型权重本身不提交 Git，以 SHA256 记录身份。

---

## Day13 Status

- Online GRPO smoke: PASS
- 1000-step GRPO: PASS
- nonzero gradient audit: PASS
- LoRA parameter update: PASS
- adapter reload: PASS
- adapter finiteness: PASS
- fixed dev500 evaluation: PASS
- paired transition analysis: PASS
- 24-case manual audit: PASS

Day13 complete.
