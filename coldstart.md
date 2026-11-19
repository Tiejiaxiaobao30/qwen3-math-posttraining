# 早期 SFT 与同题量 GRPO · 复现

[项目总览](README.md) · [环境与 Base 准备](运行命令.md) · [本轮实跑结果和配置](历史结果/grpo_coldstart_4090/README.md)

## 生成并评测第 2 步起点

从项目根目录执行。使用 `requirements.txt`，准备与上一轮相同的 BF16 CUDA 环境、完整 Base 和 `python data.py` 构建的数据。官方 test 文件使用 `历史结果/data/official/gsm8k_test_1319.jsonl`，代码会核验题集和提示词指纹。

```bash
export MODEL="$HOME/models/Qwen3-1.7B-Base"
export OUT="outputs/coldstart-step2-repro"

python -u coldstart.py --model "$MODEL" --output "$OUT" --sft-steps 2 sft
python -u coldstart.py --model "$MODEL" --output "$OUT" --sft-steps 2 screen --screen-batch-size 30
```

SFT 使用原 2,000 题数据、batch 16，按完整一轮 125 步调度，在第 2 步保存。其他已录制试验点是 8 步（筛题批次 25）和 32 步（筛题批次 12）。指定步数和输出目录后，后续各条命令沿用两者。

筛题每题 G4，保留 1–3 条答对的题。两组通常各 476 道，随机组复用完整 SFT 对照的旧名单。混合题不足时两组同时缩到相同的偶数题量，保持每次更新两道题、训练两轮。

```bash
# 分别在两个终端运行，使用相同的 MODEL 和 OUT。显存不足时依次运行。
python -u coldstart.py --model "$MODEL" --output "$OUT" --sft-steps 2 train random
python -u coldstart.py --model "$MODEL" --output "$OUT" --sft-steps 2 train informative
```

训练结束后评测各自最终 adapter 和第 2 步 SFT 基线。评测默认为 greedy、最多 512 个新 token、每批 20 道。默认数据为官方 test1319。

```bash
for name in weak_sft random informative; do
  python -u coldstart.py --model "$MODEL" --output "$OUT" --sft-steps 2 eval "$name" || exit
done
python coldstart.py --output "$OUT" --sft-steps 2 compare
```

## 用归档权重复测

本地原始包位于 `outputs/coldstart/archive/qwen-coldstart-results-lite-20261119.tar.gz`。解包到独立目录，沿用归档中的 `sft_rebuild/checkpoint-2`、`random/final_adapter`、`informative/final_adapter` 和筛题记录。使用已完成目录评测会保护原预测文件，建议把已完成目录复制或解包到新的输出位置后运行，保留原始 test 子目录用于核验。两组 GRPO 都先合并第 2 步 SFT，再加载各组 GRPO adapter。精简归档保留最终权重及生成记录，完整的中间续训 checkpoint 只存在于原运行端保留的训练目录中。

500 题留出集与 SFT2000、RL1000、dev500 无交集。若需复核，将同哈希的保留文件放在 `data/holdout500.jsonl`。本地源文件在 `历史备份/旧数据/final_grpo_20261113/holdout.jsonl`。每条 `eval` 命令追加 `--dataset holdout`，随后在 `compare` 也追加同样参数。比较两种起点时，用以下三个历史权重入口补齐同一份题集：

```bash
STRONG=outputs/sft_full/final_adapter
PREVIOUS=outputs/final_grpo
python -u coldstart.py --model "$MODEL" --sft "$STRONG" --output "$OUT" --sft-steps 2 eval strong_sft --dataset holdout
for group in random informative; do
  python -u coldstart.py --model "$MODEL" --sft "$STRONG" --output "$OUT" --sft-steps 2 \
    eval "strong_$group" --dataset holdout \
    --grpo "$PREVIOUS/${group}42-b8-e2-parallel/final_adapter" || exit
done
python coldstart.py --output "$OUT" --sft-steps 2 compare --dataset holdout
```

官方 test 和留出集分别输出。当前仓库报告只包含已完成的官方 test1319。
