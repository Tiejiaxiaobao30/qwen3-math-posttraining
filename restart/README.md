# Qwen3 GRPO 奖励消融重启（2026-11-10）

基于已有 Qwen3-1.7B-Base + SFT LoRA，比较“严格答案正确性”与“正确性 + 小权重格式奖励”的在线 GRPO。统一数据、采样、训练步数和评测，支持 SwanLab、单机 1/2/3 卡、断点恢复和逐题改善/退化分析。

**本轮交付是代码准备；没有启动训练，没有新模型效果。** 历史两个 seed 的 GRPO 在官方 test 均低于对应 SFT，详见仓库根 README。该负结果不能证明具体退化原因。新配置不是历史运行环境的精确复刻，也不承诺提高数学推理正确率。

## 1. 奖励为什么这样改

旧入口实际只调用 `strict_correctness_reward`，已有的 `format_reward` 未进入总奖励。本轮从已有报告观察到格式失败，但它只是相关现象。将“格式信号是否有帮助”作为待验证假设：

| 回答 | strict | format_aux |
| --- | ---: | ---: |
| 答案正确且格式合规 | 1.0 | 1.1 |
| 答案错误但格式合规 | 0.0 | 0.1 |
| 格式不合规 | 0.0 | 0.0 |

总奖励 `r = correctness + w * format`，两组唯一主动干预为 `w=0 / 0.1`。两组均计算/记录两项原始奖励。训练底座直接调用固定版本 TRL 的 GRPOTrainer；本仓库新增的是实验编排、资产与数据校验、奖励接线、诊断和 AutoDL 入口，并未自行重新实现 GRPO 优化器。

这种设计会给“错但格式正确”少量奖励，可能改善协议遵守，也可能诱导格式捷径；用相同的严格正确率评测，不能凭总 reward 上升宣布成功。它不会修复全错且格式相同、或全对组的零相对奖励；KL 仍可能提供梯度。“全同分”不是“所有训练梯度一定为零”。不加入推理步骤数、关键词命中或答案字符串出现在 reasoning 中的奖励。

先看开发集错误和在线 `audit/*`，如格式已经接近饱和，干预可能没有作用。首轮不再同时改 G、长度、loss、KL 或标准差缩放。更换方案要开新协议/新 run，不能覆盖本轮记录。

## 2. 资产：SFT Adapter 确实存在

本地来源：`E:/qwen/qwen3-math/outputs/sft_full/final_adapter`。`adapter_model.safetensors` 为 69,782,384 字节，r=16、alpha=32，基座记录为 `Qwen3-1.7B-Base`。它不是完整 Base 权重。

GitHub 保存代码和 SHA256 清单，**没有保存 Base 或 SFT 权重**。本轮额外准备 `E:/qwen/restart-assets/qwen3-sft-final-adapter.zip` 供你上传 AutoDL，包内目录名为 `final_adapter/`。不要下载成 Instruct，也不要把 SFT adapter 当完整模型路径。

模型从 [ModelScope Qwen3-1.7B-Base](https://modelscope.cn/models/Qwen/Qwen3-1.7B-Base) 下载，固定沿用历史 `model_revision.txt` 的 `43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3`。脚本验证模型类型、hidden_size、SFT 基座声明和 SFT 两个关键文件的哈希；形状检查不等于模型全权重来源证明。`run.json` 保存模型 revision 声明，实际资产来源仍需保留下载缓存/日志。没有在本机执行下载大模型、合并和加载验证。

## 3. AutoDL 安装与准备

启动一个 **Linux、Python 3.11、支持 BF16 的 NVIDIA GPU** 环境。安装脚本使用 PyTorch 2.6.0 CUDA 12.4 wheel；驱动必须支持此运行时，用 `nvidia-smi` 检查。具体显存和速度尚未实测，先运行 inspect 和 20 步试跑，再决定是否扩大。三卡 DDP 每张卡都放一份模型，显存不会简单合并为一个大池。

以下从仓库根目录执行。已下载仓库则跳过 clone/cd 两行。代码位于本次专用分支：

```bash
cd /root/autodl-tmp
git clone --branch feat/reward-restart-autodl https://github.com/Tiejiaxiaobao30/qwen3-math-posttraining.git
cd qwen3-math-posttraining
conda create -n qwen-restart python=3.11 -y
conda activate qwen-restart
bash restart/setup.sh

# 下载历史同型号/同 revision Base；已有相同模型则跳过下载并改 MODEL_DIR
python -m restart.download_model
export MODEL_DIR=/root/autodl-tmp/models/Qwen3-1.7B-Base

# 用 AutoDL 文件上传功能把本地权重 ZIP 放到 /root/autodl-tmp/
mkdir -p /root/autodl-tmp/assets
unzip /root/autodl-tmp/qwen3-sft-final-adapter.zip -d /root/autodl-tmp/assets
export SFT_ADAPTER=/root/autodl-tmp/assets/final_adapter

# 此命令只下载公开 train split，不下载旧官方 test，不训练
python -m restart.prepare_data

# 登录时交互输入自己的 key，不要写进脚本、README 或 Git
swanlab login
```

依赖完整清单在 [requirements.txt](requirements.txt)：TRL、Transformers、PEFT、Accelerate、Datasets、SwanLab、ModelScope 等已固定直接版本。不是完整传递依赖 lock；安装后保存 `restart-environment.local.txt`，不要把其视为历史环境。国内访问 Hugging Face 失败时，可以在能联网的机器执行数据准备后上传整个 `data/restart/`；或按下一节提供原序 TRAIN JSONL。不会静默改成另一个数据集。

## 4. 数据与验证边界

选用 [Hugging Face openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)，固定 revision `740312add88f781978c0658806c59bc2815b9866`、main/train。已核对公开 train 为 7473 条，question/answer 字段可用；按已有 `split_manifest.json` 复建原 RL1000/dev500，不重新随机切分。也查询了 [ModelScope 镜像](https://modelscope.cn/datasets/modelscope/gsm8k)，主链使用可直接核对内容的 HF 版本。

脚本检查原 ID、规范化题目跨 SFT/RL/dev 重复、数值答案，以及重建后的完整内容指纹。Gold 只传给奖励，不拼进 prompt。已在本机用固定 HF parquet 实际重建并通过旧文件内容指纹核对。若镜像顺序或内容不同，会拒绝继续；不得删掉校验绕过。

如果你已有从 ModelScope 下载、保持原始 TRAIN 顺序的标准 `{question, answer}` JSONL：

```bash
python -m restart.prepare_data --source local --raw-jsonl /root/autodl-tmp/gsm8k-train.jsonl
```

`data/restart/` 已存在时脚本拒绝覆盖，保留原数据即可。数据文件不上传 Git。

**dev500 已用于旧实验开发；官方 test 也已曝光。** 新项目明确重启旧冻结实验之外的探索，旧官方测试结果和原冻结文件保持原样，不将这 1319 题再称作新封存测试。本轮只交开发集结论；真正独立的泛化验证仍需另行确定、核对污染并冻结一个未用于选择的新集合。这里没有把 reserve 题池擅自宣称为新独立金标。

## 5. 启动：先核对，再短训，再对照

只建模和 Trainer、验证资产/配置，**不采样、不执行 train**：

```bash
WORLD=1 ARM=strict OUT=outputs/restart/inspect bash restart/run.sh --inspect-only
```

在单卡上先试 20 步，确认 SwanLab 有曲线、checkpoint 可保存、显存和耗时可接受。这是你之后运行的命令，本次助手没有执行：

```bash
WORLD=1 ARM=strict STEPS=20 OUT=outputs/restart/smoke-strict bash restart/run.sh
```

单卡正式对照，两个 run 都从同一份 SFT 重新初始化：

```bash
WORLD=1 ARM=strict SEED=42 bash restart/run.sh
WORLD=1 ARM=format_aux SEED=42 bash restart/run.sh
```

同一台 AutoDL 实例有三张卡时：

```bash
WORLD=3 ARM=strict SEED=42 bash restart/run.sh
WORLD=3 ARM=format_aux SEED=42 bash restart/run.sh
```

这是三卡 DDP；不能直接用于三台独立实例。若你租了三台各单卡机器，可各自准备模型/SFT/数据，分别运行 strict-seed42、format_aux-seed42、strict-seed123，之后补 format_aux-seed123；第三个任务不替代两个奖励组的同 seed 配对。不要在同一输出目录混用机器或不同种子。

| 参数 | 单卡 | 三卡 |
| --- | ---: | ---: |
| 每卡 completion microbatch | 1 | 1 |
| 梯度累积 | 12 | 4 |
| 全局 completion batch / update | 12 | 12 |
| G（同题采样数） | 4 | 4 |
| 每 update 唯一 prompt 数（未计重复） | 3 | 3 |
| 默认更新步数 | 200 | 200 |

不是把三卡全局 batch=3 直接交给 G=4；generation_batch=12 才能整除。与旧历史全局 batch=4 不完全相同，所以新 strict 是必要的同条件基线，不能用旧数值替代。单卡/三卡虽然匹配全局批量，随机数流、采样顺序和浮点规约仍可能不同；正式两组使用相同 world size。200 步只是首轮探索，不宣称收敛。

短跑发现内存不足或严重截断，先记录问题，两个正式组必须统一调整后重新开始。不要只给某一组增加长度或减少样本。

断点恢复示例（真实存在的 checkpoint-N）：

```bash
WORLD=3 ARM=strict SEED=42 bash restart/run.sh \
  --resume outputs/restart/strict-seed42-gpus3/checkpoint-100
```

保留完整 `checkpoint-N`，它包含 optimizer/scheduler/RNG/Trainer 状态；`final_adapter/` 只用于推理，不是完整恢复点。脚本拒绝改变奖励/seed/world size/数据/实验配置/代码版本的恢复。SwanLab 恢复会创建一个新记录段；不声称自动连接原云端 run。输出目录非空且未传 resume 时拒绝覆盖。

## 6. SwanLab 看什么

打开训练终端打印的 SwanLab 实验链接，项目名 `qwen3-reward-restart`。三卡仅主进程上传同一 run。训练 `loss`、KL、reward、reward_std、`frac_reward_zero_std`、生成长度/截断等由 TRL 输出；新增 `audit/correctness`、`audit/format_fail`、`audit/all_wrong_groups`、`audit/all_correct_groups` 和主进程显存。

诊断按跨卡聚合后的完整 G=4 组计算，避免只看某个 rank 上的不完整组。reward 权重不同，总 reward 不能作为两组共同质量指标。正确率要看相同 dev greedy 评测；在线采样正确率只是训练信号诊断。原始 rollout 文本默认不上传，训练统计/配置会上传；不要把敏感字段加入配置。输出 `run.json`、`metrics.json`、`log_history.json` 和 checkpoint 可本地审阅。

训练统计看不到 dev 曲线是正常的：本实现不每步生成 500 道评测，评测独立执行并上传 `eval-*` 记录。无网时设置 `SWANLAB_MODE=offline` 可本地记录，但不能称已经在云端可见；恢复网络后按 SwanLab 官方说明同步。

## 7. 评测和逐题对照

训练完成后先对 SFT、strict、format_aux 依次生成相同 dev500，greedy、512 new tokens、旧严格 parser。**不要用训练采样 reward 代替评测。** 以下以三卡训练产物为例；单卡将 `gpus3` 改成 `gpus1`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m restart.evaluate --model-dir "$MODEL_DIR" \
  --sft-adapter "$SFT_ADAPTER" --output outputs/restart/eval-sft.jsonl

CUDA_VISIBLE_DEVICES=0 python -m restart.evaluate --model-dir "$MODEL_DIR" \
  --sft-adapter "$SFT_ADAPTER" \
  --grpo-adapter outputs/restart/strict-seed42-gpus3/final_adapter \
  --output outputs/restart/eval-strict.jsonl

CUDA_VISIBLE_DEVICES=0 python -m restart.evaluate --model-dir "$MODEL_DIR" \
  --sft-adapter "$SFT_ADAPTER" \
  --grpo-adapter outputs/restart/format_aux-seed42-gpus3/final_adapter \
  --output outputs/restart/eval-format_aux.jsonl

python -m restart.compare --before outputs/restart/eval-sft.jsonl \
  --after outputs/restart/eval-strict.jsonl --output outputs/restart/sft-vs-strict.json
python -m restart.compare --before outputs/restart/eval-strict.jsonl \
  --after outputs/restart/eval-format_aux.jsonl --output outputs/restart/reward-ablation.json
python -m restart.compare --before outputs/restart/eval-sft.jsonl \
  --after outputs/restart/eval-format_aux.jsonl --output outputs/restart/sft-vs-format_aux.json
```

先汇报 correct/n、wrong、format_fail、truncated、输出长度，再看改善/退化 ID。只有格式改善而严格正确率未变，结论就写格式改善；没有提升如实保留。先固定本次结论再换 seed=123 重复两组；若复用 seed42 SFT 起点，42/123 是 GRPO 随机种子，不是两套独立 SFT 种子。

## 8. 来源、借鉴和本轮验收

- [TRL v0.24.0 GRPOTrainer](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py)、[GRPOConfig](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py)：调用训练、采样、奖励加权、KL、DDP 和保存/恢复逻辑。本轮阅读源码核对了 generation batch 及全局 reward gather。不是复制最新 main 的可变 API。
- [Open-R1 rewards（固定 commit）](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py)：借鉴分离 accuracy/format、显式加权和反例检查的设计；没有照搬 `<think>/<answer>` 模板，也未声称复现其结果。上游已停止维护，执行依赖使用 TRL。
- [SwanLab TRL 官方说明](https://docs.swanlab.cn/en/guide_cloud/integration/integration-huggingface-trl.html)：采用显式 `SwanLabCallback`，训练和独立评测均有记录入口。
- [GSM8K 原作者仓库](https://github.com/openai/grade-school-math)、[HF 数据卡](https://huggingface.co/datasets/openai/gsm8k)：数据来源/许可随上游；本轮仅下载 train，未引入 NuminaMath 等混合数据制造第二个变量。

已执行的检查见 [VALIDATION.md](VALIDATION.md)。CPU 检查和 API 构建不等价于 CUDA 加载、DDP、SwanLab 云上传、训练稳定性或模型收益；这些都要由实际 AutoDL 日志验证。

可用于 GitHub 的项目简介：**基于 Qwen3-1.7B 的数学后训练实验：保留两种子 GRPO 负结果，复用 SFT LoRA，构建可复跑的奖励消融、SwanLab 监控及单机三卡训练流程。新奖励效果待训练验证。**
