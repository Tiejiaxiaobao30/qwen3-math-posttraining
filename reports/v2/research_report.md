# V2：G4 / G8 后续实验（2026-10-20 备份）

V2 G8 在官方 1319 题上比 V2 G4 多答对 9 题（+0.68 个百分点），但比 SFT seed42 少答对 4 题（−0.30 个百分点）；dev500 则比 G4 少答对 6 题。现有证据不支持稳定泛化提升，也不支持 G8 更省算力的因果结论。

## 范围与 V1 历史

这是在 V1 官方测试已打开后的**探索性后续实验**。V1 的 `results/frozen_eval_manifest.json`、六组官方结果、两种种子、原报告与 Git 历史全部保留。V1 文档中的“不再训练 / no post-test tuning / SEALED”等表述限定于原冻结实验，不能扩展到 V2。V2 没有独立预注册冻结或新的未见测试集；不能称为新的 sealed-test 确认实验，也不能从压缩包单独证明没有测试反馈影响选择。

V2 先合并 SFT42 adapter，再挂新的 GRPO LoRA。备份中的 SFT 权重 SHA256 与 V1 冻结清单一致：`63420f32629bbf452b30b4e5d27060c924f01b37dea013a073bb2956cea2437f`。dev500、官方测试和提示词的哈希与 V1 一致。备份中既有配置、脚本和官方结果的文本内容与 V1 一致（Windows 工作区比较时忽略 CRLF/LF）；因此没有虚构训练脚本修改。新增代码仅为 CPU 证据审计。

## 结果（严格答案准确率）

| 模型 | dev500 | official test1319 | 相对 SFT42 官方变化 |
|---|---:|---:|---:|
| SFT seed42（V1 基线） | 390/500 = 78.00% | 902/1319 = 68.39% | — |
| V1 GRPO G4 seed42 | 392/500 = 78.40% | 893/1319 = 67.70% | −0.68 pp |
| V2 G4 seed42 | 384/500 = 76.80% | 889/1319 = 67.40% | −0.99 pp |
| V2 G8 seed42 | 378/500 = 75.60% | 898/1319 = 68.08% | −0.30 pp |

以上百分比均由正确题数计算。G 表示训练时每个问题的采样组大小，**不是 pass@G**；官方评测是 `do_sample=false`、`max_new_tokens=512` 的单次贪心输出。格式错误计为不正确，截断是独立标记。V2 G4 / G8 官方格式错误为 9 / 8，截断均为 7；dev 格式错误均为 2，无截断。

逐题配对（按 ID 和 reference 对齐）：

| G4 → G8 | dev500 | official test1319 |
|---|---:|---:|
| 两组均正确 | 366 | 858 |
| 错变对（含原格式失败） | 12 | 40 |
| 对变错（含新格式失败） | 18 | 31 |
| 两组均不正确 | 104 | 390 |
| 净正确题数变化 | −6 | +9 |

官方测试是同一组 1319 题，不能把两个模型的预测当作 2638 个独立样本。这里只报告描述性差异，不声称统计显著。V2 G8 与既有 Extra-SFT 恰好同为 898/1319，也不能据此推断逐题表现相同。

## 训练设置、耗时与混杂因素

两组均为 seed/data_seed 42、500 optimizer steps、单卡配置 batch 4 × gradient accumulation 2、学习率 1e−6、LoRA r16/alpha32/dropout0.05、温度 0.8、top_p 0.95、completion 上限 256、beta 0.04、bnpo、num_iterations=1、scale_rewards=true、mask_truncated_completions=true。保存的两份完整配置仅 `num_generations` 和输出目录不同。

| 训练记录 | G4 | G8 |
|---|---:|---:|
| optimizer steps | 500 | 500 |
| train_runtime（秒） | 3350.7089 | 2731.2307 |
| 有非零梯度的步骤 | 500 | 499 |
| 权重变化张量数 | 392 | 392 |
| peak allocated（GiB） | 10.9262 | 11.0563 |
| peak reserved（GiB） | 13.3965 | 19.8086 |

两组日志均未记录 NaN。G8 的一条 optimizer-step 记录没有非零梯度，不等于缺失了一个 optimizer step。保留的梯度审计和训练指标可复核这些计数。脚本日志标题仍写 `START 10-STEP GRPO`，这是沿用的静态标题；真实配置、global_step 和 500 条梯度记录均为 500，不能用标题判断训练长度。

必须保留的限制：

- **组大小同时改变问题分组和覆盖。** 固定 batch、梯度累积与 optimizer steps 并不保证相同独立问题数。按单卡有效 batch=8 条完成样本的配置推算，G4 每批可对应约 2 个问题组，G8 约 1 个；这是配置层面的推算，备份未保存完整 sampler trace，不能作为实测独立问题数。不能将差异纯归因为“更多组内采样”。
- **完成长度、奖励方差与计算负载可能不同。** 两次 runtime 的记录差为 −619.4782 秒（G8 较短约 18.49%），但缺少相同问题顺序、相同生成 token 总预算、重复计时、缓存/热身和系统负载控制。只报告一次运行的耗时，不声称 G8 有 18.49% 算法加速或更高算力效率。
- **V1→V2 不是单变量比较。** V1 G4 配置为 1000 steps、accumulation=1；V2 是 500 steps、accumulation=2，且运行平台/软件环境不同。不能将其差异单独归为 G、训练时长或某项实现修复。
- **一个种子、各一次运行。** 相同 seed 不保证改变 G 后的采样轨迹相同；没有跨种子重复，不能支持稳健排序。dev 和官方测试方向相反，更不能据官方测试择优后声称泛化收益。
- **环境快照不是完整可复现镜像。** 快照记录 RTX 4090、Python 3.12.3、torch 2.8.0 cu128、transformers 4.55.4、trl 0.21.0、peft 0.17.1。它是备份时采集的环境信息，不能独立证明每个训练步骤的硬件/依赖状态。模型路径保留原 AutoDL 绝对路径，复现时需另行设置本地路径；不应覆盖这些历史结果。

## 证据与复核

- [原始配置](../../configs/v2_g4.json)、[G8 配置](../../configs/v2_g8.json)及两份 smoke 配置按原字节导入。
- [统计与转换计数](verification.json)、[文件来源与 SHA256](provenance.json)、[环境快照](environment_snapshot.txt)。
- [训练指标与 500 步梯度审计](training/)、[原始训练日志](source_logs/)。
- [轻量结果](../../results/v2/)包含四份原始 summary 和两个逐题 CSV。CSV 仅保留 ID、reference、parsed answer、状态、token 数与截断标记，不含生成长文本。

```bash
# CPU / Python 标准库；不生成预测、不重新训练
python -m scripts.audit_final_evaluation
python -m scripts.audit_v2_evidence
# 本地完整备份存在时：重新评分全部 3638 条 V2 预测，并校验权重谱系等来源
python -m scripts.audit_v2_evidence --archive E:/qwen/qwen3-v2-evidence-20261020.tar.gz
```

仅仓库审计能复核轻量记录、哈希、计数和配对转换；重新解析生成全文必须使用本地备份。导入时已使用备份完成全文重评分，3638/3638 条与冻结 parser 结果一致，同时通过 V1 原有审计。备份的 gzip 数据流已完整读取，SHA256 为 `404488230140ce611d64e3a90004214239ea25fe81128dc768965b4d6b6290ef`，大小 207652808 字节。

权重、tokenizer、压缩包、完整逐题生成和大体积原始数据仅留在本地完整备份，不上传 Git，也未创建 Release/LFS。该压缩包不包含 Base 模型权重；因此“备份完好”不等于完整离线运行环境已打包。后续重新推理还需要获取对应 Base 模型与依赖。
