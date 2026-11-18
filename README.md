# Qwen GSM8K：只使用这一个项目

本地位置：`E:/qwen gsm8k`。所有运行代码和配置就在根目录，不再经过projects/restart/teaching嵌套。

```text
qwen gsm8k/
  data.py             处理GSM8K，固定SFT2000/RL1000/dev500
  train.py            SFT或GRPO训练
  sample.py           SFT每题G4筛题，输出同数量随机对照
  evaluate.py         同条件dev评测、逐题改善/退化对比
  common.py           IO、模型合并、生成、严格评分器
  prompt.txt          唯一提示词
  split_manifest.json 固定样本划分
  requirements.txt
  test_core.py
  data/               已处理数据（本地已生成）
  models/             完整Base模型（尚未下载）
  outputs/            训练checkpoint、最终adapter、筛题与评测输出
  logs/               原始训练日志
  历史备份/            必要历史评测证据与恢复备份，日常不用进入
```

旧代码不是整套搬过来继续使用。数据清洗、SFT/GRPO、严格评分、G4筛题和配对评测已合进上面五个文件；平台专用入口、重复评测、一次性制表、旧4090/TRL0.21封装从正常代码中删除。逐文件取舍见`代码取舍说明.md`。历史冻结核验仍需的少数源码属于证据，不是第二套训练入口。

## 模型与数据

已有SFT42：`outputs/sft_full/final_adapter`；对应GRPO42：`outputs/grpo_full/final_adapter`。seed123最终adapter在`outputs/sft_seed123/final_adapter`、`outputs/grpo_seed123/final_adapter`。这些是LoRA，不能当完整Base。

`data/sft.jsonl`是prompt/completion/gold；`data/rl.jsonl`、`data/dev.jsonl`只有prompt/gold/id。数据已从公开train缓存重建，与原RL逐条核对。Gold只传给loss/reward，不拼进模型prompt。新环境第一次准备：

```bash
# 在项目根目录执行；已有data/时不要覆盖
python data.py
# 或提供保持原始顺序的GSM8K train JSONL
python data.py --raw-jsonl /path/gsm8k-train.jsonl
```

固定HF train revision及历史ID，不下载official test。此处dev500曾用于历史开发，不能当新封存测试集。

## 云端环境

以下为同一台Linux三卡BF16 GPU、Python3.11命令。先将当前项目源码上传/从仓库获取，并将本地data/、需要的SFT adapter上传到同样相对目录。大型产出、权重、私有恢复ZIP不提交Git。

```bash
conda create -n qwen-gsm8k python=3.11 -y
conda activate qwen-gsm8k
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip freeze > environment-cloud.txt
nvidia-smi
export MODEL=/root/autodl-tmp/models/Qwen3-1.7B-Base
export SFT=outputs/sft_full/final_adapter
```

本地没有完整Base权重；已有相同Base可指定路径。没有时，下载到空目录并保存来源日志：

```bash
pip install modelscope==1.30.0
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B-Base', revision='43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3', local_dir='/root/autodl-tmp/models/Qwen3-1.7B-Base')"
```

这是历史声明的ModelScope revision；实际资产指纹记录到run，revision文本本身不证明权重来源。新云端Python/Torch组合、三卡CUDA和完整1.7B训练尚未验证，先inspect再20步检查。

## SFT → 筛题 → GRPO → 评测

```bash
# 如复用已有SFT，跳过此段；从头训练时先20步，再另开正式目录
torchrun --standalone --nproc_per_node=3 train.py sft --model "$MODEL" --data data/sft.jsonl --output outputs/sft-inspect --inspect-only
torchrun --standalone --nproc_per_node=3 train.py sft --model "$MODEL" --data data/sft.jsonl --output outputs/sft-smoke --steps 20
torchrun --standalone --nproc_per_node=3 train.py sft --model "$MODEL" --data data/sft.jsonl --output outputs/sft42 --steps 1000 --seed 42
export SFT=outputs/sft42/final_adapter

# 筛题：每题采4次，答对1–3次的原RL题构成informative；random取同数量
CUDA_VISIBLE_DEVICES=0 python sample.py --model "$MODEL" --sft "$SFT" --data data/rl.jsonl --output outputs/screen42 --seed 42

# GRPO先构建、20步试跑，再独立启动同条件对照
torchrun --standalone --nproc_per_node=3 train.py grpo --model "$MODEL" --sft "$SFT" --data outputs/screen42/random.jsonl --output outputs/grpo-inspect --inspect-only
torchrun --standalone --nproc_per_node=3 train.py grpo --model "$MODEL" --sft "$SFT" --data outputs/screen42/random.jsonl --output outputs/grpo-smoke --steps 20
torchrun --standalone --nproc_per_node=3 train.py grpo --model "$MODEL" --sft "$SFT" --data outputs/screen42/random.jsonl --output outputs/random42 --steps 200 --seed 42
torchrun --standalone --nproc_per_node=3 train.py grpo --model "$MODEL" --sft "$SFT" --data outputs/screen42/informative.jsonl --output outputs/informative42 --steps 200 --seed 42

CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "$MODEL" --sft "$SFT" --data data/dev.jsonl --output outputs/eval/sft.jsonl
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "$MODEL" --sft "$SFT" --grpo outputs/random42/final_adapter --data data/dev.jsonl --output outputs/eval/random.jsonl
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "$MODEL" --sft "$SFT" --grpo outputs/informative42/final_adapter --data data/dev.jsonl --output outputs/eval/informative.jsonl
python evaluate.py --before outputs/eval/random.jsonl --after outputs/eval/informative.jsonl --output outputs/eval/paired.json
```

GRPO先合并同一SFT，再创建全新LoRA；评测也先合并该SFT再载入GRPO adapter。不要把GRPO adapter直接挂到Base上。

1/2/3卡的全局completion batch均为12，G4，每次update三题；新SFT全局batch12与历史batch2不同，这是新探索协议，不能称原实验精确复现。两组需同SFT、样本数、seed、步数、卡数与超参。SFT模型和GRPO随机种子含义分开解释。筛题固定256 tokens，评测greedy512；不能用训练reward代替dev正确率。

训练每50步保存完整checkpoint，不自动删除。续训在相同命令后加`--resume outputs/random42/checkpoint-100`，保持steps/seed/数据/代码/资产相同。final_adapter只用于推理，不能当optimizer恢复点。标准Trainer续训不承诺跨中断逐token完全相同的rollout轨迹。筛题逐题保存原文，但本版不支持断点采样；中断后用新目录重跑，不输出不完整训练子集。

所有新输出拒绝覆盖。评测记录原文、correct/wrong/format_fail、截断、长度与指纹；配对比较报告改善/退化ID和净变化。仅报告探索性dev结论，无新holdout承诺。

## 验证与历史边界

```bash
python -m pytest test_core.py -q
# 核验历史冻结记录时才需要进入这里
cd 历史备份/旧实验记录
python -m scripts.audit_final_evaluation
python -m scripts.audit_v2_evidence
```

历史V1两个seed的GRPO低于相应SFT；V2属于official test曝光后的探索，单次G8相对G4的变化不证明稳定增益。原结果、报告、冻结parser/prompt与验证需要的源码保留原字节；只有存放位置变化。移除旧平台入口不代表删除历史负结果。

迁移前恢复ZIP、完整Git历史bundle、旧checkpoint完整TAR留在历史备份，防止整理误丢成果；不作为日常代码，不上传远端。代码取舍是本轮逐项审核后的结果，不是把全部旧版本保留成并列工程。
