# 最终 GSM8K official-test 报告

## 结论

**GRPO 在固定 dev500 上两个 seed 均获得 +0.40pp，但这一小幅收益没有泛化到 freeze 后启封的 GSM8K official test。** 两个 seed 的 GRPO test accuracy 都低于对应 SFT。SFT 是本次实验中主要有效的后训练阶段；当前 binary-correctness-reward GRPO 不支持“提升最终泛化表现”的表述。

## 六组完整结果

| Model | Correct / 1319 | Strict accuracy | Wrong | Format fail | Truncated |
|---|---:|---:|---:|---:|---:|
| Base | 665 / 1319 | 50.42% | 264 | 390 | 9 |
| SFT seed42 | 902 / 1319 | 68.39% | 408 | 9 | 8 |
| GRPO seed42 | 893 / 1319 | 67.70% | 413 | 13 | 11 |
| Extra-SFT control | 898 / 1319 | 68.08% | 413 | 8 | 6 |
| SFT seed123 | 897 / 1319 | 68.01% | 412 | 10 | 6 |
| GRPO seed123 | 882 / 1319 | 66.87% | 422 | 15 | 10 |

Strict accuracy = correct / 1319；wrong 与 format_fail 都算错误；truncated 与这三类状态独立，不能重复相加。所有运行使用同一题目顺序、prompt、strict parser、greedy decoding 和 max_new_tokens=512。

## Dev 与 test 对照

| Seed | Dev SFT→GRPO | Test SFT→GRPO | Test 精确计数差值 |
|---|---|---|---|
| 42 | 78.00%→78.40% (+0.40pp, +2/500) | 68.39%→67.70% | −9/1319 = −0.68pp |
| 123 | 76.00%→76.40% (+0.40pp, +2/500) | 68.01%→66.87% | −15/1319 = −1.14pp |

两 seed 平均：SFT 68.20%，GRPO 67.29%，变化 −0.91pp。该平均描述两次训练对同一 test 的表现，不能把它解释为 2638 个独立测试样本。

差值一律由未舍入的分数计算：seed42 为 −0.682335pp。直接用展示值 67.70−68.39 得到的 −0.69pp 是二次舍入差异。Extra-SFT 对 SFT seed42 的变化为 −4/1319 = −0.30pp；Base→SFT seed42 为 +237/1319 = +17.97pp。

## Paired analysis 与失败计数

按相同样本 ID 对齐，correct 与非 correct（wrong/format_fail）作二分类：

| Seed | 两者都对 | SFT错→GRPO对 | SFT对→GRPO错 | 两者都错 | 净变化 |
|---|---:|---:|---:|---:|---:|
| 42 | 857 | 36 | 45 | 381 | −9 |
| 123 | 850 | 32 | 47 | 390 | −15 |

样本 ID 完整保存在 [verification.json](verification.json) 的 `flip_ids`。这是事后描述性分析，不用于调参或挑选新的 checkpoint。

Base format_fail 为 390/1319；SFT seed42 降到 9/1319，所以 SFT 的 strict-accuracy 提升包含输出协议遵守能力提升，不能全部归因为数学推理提升。GRPO 的 format_fail 分别为 13 与 15，高于对应 SFT 的 9 与 10；格式失误增加与净退化同时出现，但这些聚合计数不能证明退化的因果机制。没有将所有 wrong 人工标注为某种推理错误，也不声称完成了逐题语义归因。

## Freeze、启封与评估方法

原始 [frozen_eval_manifest.json](../../results/frozen_eval_manifest.json) 声明 `FROZEN_BEFORE_OFFICIAL_TEST`，freeze_time 为 `2026-10-10T23:03:54.797809`（未记录时区）。其 `SEALED` 是冻结时的状态快照，完整保留。

冻结记录涵盖六个预声明模型、prompt/parser/dev500 SHA256、五个 adapter SHA256、greedy generation、512 token 上限及 no-post-test-tuning policy。当前本地 prompt/parser/dev500 均与其哈希完全一致。test 文件 SHA256 为：

```text
d9a6a8c13b99ac41d22bfdb5912550fe202f92dca88e3565190bd9ca044ada0e
```

实验记录规定 test 在 freeze 后启封；此后 no more training、no more dev tuning、no more hyperparameter search、no third seed、no post-test tuning。此次只整理与复核存量证据，没有重新训练、生成答案或修改 frozen parser/prompt。

Base/SFT 单题 smoke 文件保留作来源证据，但不计入六组正式结果。Extra-SFT summary 的 `mode=grpo` 指 evaluator 的“两阶段 adapter 装载路径”：先 merge SFT，再挂载 Extra-SFT adapter；不表示 Extra-SFT 使用 GRPO 训练。该原始字段未被改写。

## 来源与复核

- `qwen3_math_official_test_final.zip`：六组 summary/predictions、两组单题 smoke、freeze manifest、official evaluator、test 固定副本、六份日志。
- `seed123_final_backup.zip`：GRPO seed123 dev500 与 evaluator；adapter 权重留在原 ZIP，未入库。
- 本地第二 seed 文件：SFT seed123 dev500、SFT/GRPO seed123 configs、SFT dev evaluator。旧环境快照不是 Kaggle lock，未冒充正式依赖文件。
- [artifact_provenance.json](artifact_provenance.json) 记录原 ZIP 和逐个导入文件的字节数、SHA256、来源。官方评估证据保持 byte-for-byte；portable dev scripts 的改动与原源码分开保存。

在仓库根目录运行 `python -m scripts.audit_final_evaluation` 可重新检查六组完整 predictions（7914 条）、四组 dev predictions（2000 条）、ID/顺序/reference/parsed answer/status、summary 计数与比例、生成参数、来源哈希，并重算 paired transitions。dev JSONL 不随此次提交发布；本地存在时额外核对其 freeze hash。

## 局限

1. 仅两个训练 seed、单一模型规模与 GSM8K；dev 已用于 checkpoint、beta、LR 等开发选择，微小 dev 增益不是独立泛化证据。
2. 未做显著性检验或置信区间；结果说明此设置的观察值，没有证明 GRPO 普遍有害或退化显著。
3. Extra-SFT 对照使用相同 RL1000 question pool，但训练机制、步数与计算量不同，不是严格 compute-equivalent comparison。
4. Strict metric 同时衡量格式和答案；Base 的格式失败较多。greedy decoding 也不代表采样多解时的表现。
5. 原 manifest 与日志不是外部签名时间证据。当前核验不能独立证明历史 freeze 先后、没有更早接触 test，或远端环境完全可复现。
6. 本次直接核验 seed123 ZIP 内 GRPO adapter 与 freeze hash；其余四个 adapter 的哈希保留为原 freeze 记录，未重新扫描大备份验证。完整 Kaggle pip freeze 未随 final ZIP 提供；依赖说明明确区分用户报告与实测。
7. vLLM benchmark、完整人工失败归因与最终简历/面试稿仍待完成。

后续仅做 serving benchmark 和项目表述收口；不根据本次 test 结果继续优化这组模型、prompt、parser、reward 或 checkpoint。
