# 单卡 4090 同题量 GRPO 筛题实验

2026-11-18 在一张 RTX 4090 上训练。和同题量随机组相比，筛题组在训练中将混合组比例提高了 **17.23 个百分点**，在历史官方 test1319 上净少答对 **8 题（−0.61 个百分点）**。

|指标|随机组|筛题组|
|---|---:|---:|
|训练题数|476|476|
|在线混合组|459/952（48.21%）|623/952（65.44%）|
|在线全对组|416/952（43.70%）|235/952（24.68%）|
|在线全错组|77/952（8.09%）|94/952（9.87%）|
|官方 test 正确|904/1319（68.54%）|896/1319（67.93%）|
|合法格式但数值错误|402|407|
|格式失败|13|16|
|达到长度上限且未结束|11|14|

按同一道题配对，筛题组新增答对 31 题，丢失 39 题。39 道丢分题中，37 题变为数值错误，2 题变为格式失败。截断可与其他状态重叠。结果支持「筛题能提高训练时出现组内正确性差异的频率」。本次单种子对照没有观察到正确率收益，未评估种子稳定性，也未在当前 Base 上单独重测 SFT42，因而无法给出相对本轮 SFT 起点的净收益。这里的官方 test 在历史实验中已经曝光。

## 实际配置

1. 冻结 Qwen3-1.7B-Base + SFT42，用 `screen12.py` 一次处理 12 道题，每题采样四次。RL1000 中有 421 题全对、103 题全错、476 题有对有错。随机组从同一池均匀不放回抽 476 题，和筛题组重叠 228 题。
2. 两组各自在同一 Base 和 SFT42 上新建 GRPO LoRA，在一张 4090 上并行。`train_b8.py` 配置为每个微批 8 条回答、梯度累积 1、G=4。两组各 476 次更新，约两遍，训练 seed42。共同配置为 BNPO、组内标准差缩放、学习率 `1e-6`、KL `0.04`、最大生成 256 tokens。实测软件版本见结果 JSON。
3. 两组各取第 476 步的 `final_adapter`，用 `eval_test_b8.py --batch-size 16` 并行评测。每题一次 greedy 生成，最多 512 tokens。以严格评分和所有 1,319 题为分母。

```bash
# 完整的 Git 克隆。仅在需要还原官方题集时执行。
git archive 100b7ce data/official/gsm8k_test_1319.jsonl | tar -x

# 从项目根目录运行。MODEL 和 SFT 指向已准备的 Base 与 SFT42 adapter。
python screen12.py --screen --model "$MODEL" --sft "$SFT" \
  --output outputs/final_grpo/screen42-b12 --seed 42

# 分别用 random 和 informative 执行。下例为随机组。
pool=random
python train_b8.py --arm baseline --model "$MODEL" --sft "$SFT" \
  --data "outputs/final_grpo/screen42-b12/$pool.jsonl" \
  --output "outputs/final_grpo/${pool}42-b8-e2-parallel" \
  --steps 476 --save-steps 25 --seed 42

MODEL="$MODEL" python eval_test_b8.py "$pool" --batch-size 16
```

准备 Base 和 SFT42 时使用与汇总一致的文件指纹。公开版 `eval_test_b8.py` 直接导入项目根目录的同逻辑评分函数，读取根目录的同哈希提示词。实际云端脚本原件保存在本地归档。复跑环境和模型资产见 [实验设计](../../阅读顺序与GRPO实验设计.md) 与 [项目命令](../../运行命令.md)。后续实验设计文档中的默认 200 步方案早于本次实际的 476 步配置。

## 可复核文件

|文件|内容|
|---|---|
|[summary.json](summary.json)|筛题分布、训练配置、环境、测试结果、逐题得失和来源 SHA256|
|[training_ids.json](training_ids.json)|两组实际采用的 476 道训练题 ID|
|[test_outcomes.csv](test_outcomes.csv)|同题对齐的两组判定、生成长度与截断标记|
|[random_rollout_metrics.csv](random_rollout_metrics.csv)、[informative_rollout_metrics.csv](informative_rollout_metrics.csv)|两组各 476 条训练采样统计|

导入时核对了归档、训练数据、权重和预测哈希，重算了 4,000 条筛题回答的奖励及 2,638 条测试预测的评分。原始生成文本、完整日志、最终 adapter、云端运行脚本原件和压缩包位于本地 `E:/qwen_gsm8k/outputs/final_grpo/`。公开表保留复算所需的题号与结果字段。
